from __future__ import annotations

import hashlib
import importlib
import faulthandler
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image


_CACHE: dict[tuple[str, str, str, int], tuple[Any, Any, Any]] = {}


def _mark(message: str) -> None:
    print(f"[geo4d_adapter] {message}", flush=True)


def _ensure_geo4d_path(root: str) -> None:
    sys.path = [p for p in sys.path if p != root]
    sys.path.insert(0, root)
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            module = sys.modules.get(name)
            module_file = str(getattr(module, "__file__", ""))
            if root not in module_file:
                del sys.modules[name]
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(Path(root) / "utils")]  # type: ignore[attr-defined]
    sys.modules["utils"] = utils_pkg


def _install_optional_dependency_shims() -> None:
    if "ipdb" not in sys.modules:
        import types

        ipdb = types.ModuleType("ipdb")
        ipdb.set_trace = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["ipdb"] = ipdb


def _install_pytorch3d_compat() -> None:
    try:
        import pytorch3d  # noqa: F401
        return
    except Exception:
        pass

    import types
    import torch

    class _PerspectiveCameras:
        def __init__(self, R=None, T=None, focal_length=None, principal_point=None, device=None, **_: Any):
            if device is None:
                device = "cpu"
            if R is None:
                if focal_length is None:
                    batch = 1
                elif isinstance(focal_length, torch.Tensor):
                    batch = int(focal_length.shape[0]) if focal_length.ndim > 0 else 1
                else:
                    batch = len(focal_length) if hasattr(focal_length, "__len__") else 1
                R = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
            else:
                R = torch.as_tensor(R, device=device).float()
                if R.ndim == 2:
                    R = R.unsqueeze(0)
            if T is None:
                T = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)
            else:
                T = torch.as_tensor(T, device=R.device, dtype=R.dtype)
                if T.ndim == 1:
                    T = T.unsqueeze(0)
            self.R = R
            self.T = T
            self.focal_length = focal_length
            self.principal_point = principal_point
            self.device = self.R.device

        def __len__(self) -> int:
            return int(self.R.shape[0])

        def clone(self):
            out = _PerspectiveCameras(R=self.R.clone(), T=self.T.clone(), device=self.device)
            out.focal_length = self.focal_length
            out.principal_point = self.principal_point
            return out

        def get_camera_center(self):
            return -torch.matmul(self.R.transpose(1, 2), self.T.unsqueeze(-1)).squeeze(-1)

    class _RayBundle:
        def __init__(self, origins=None, directions=None, lengths=None, xys=None, **kwargs: Any):
            self.origins = origins
            self.directions = directions
            self.lengths = lengths
            self.xys = xys
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Transform:
        def __init__(self, matrix):
            self._matrix = matrix

        def inverse(self):
            return _Transform(torch.linalg.inv(self._matrix))

        def compose(self, other):
            return _Transform(torch.matmul(self._matrix, other._matrix))

        def get_matrix(self):
            return self._matrix

    class _Rotate(_Transform):
        def __init__(self, R):
            R = torch.as_tensor(R).float()
            if R.ndim == 2:
                R = R.unsqueeze(0)
            matrix = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
            matrix[:, :3, :3] = R
            super().__init__(matrix)

    class _Translate(_Transform):
        def __init__(self, T):
            T = torch.as_tensor(T).float()
            if T.ndim == 1:
                T = T.unsqueeze(0)
            matrix = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
            matrix[:, 3, :3] = T
            super().__init__(matrix)

    root = types.ModuleType("pytorch3d")
    renderer = types.ModuleType("pytorch3d.renderer")
    transforms = types.ModuleType("pytorch3d.transforms")
    renderer.PerspectiveCameras = _PerspectiveCameras  # type: ignore[attr-defined]
    renderer.RayBundle = _RayBundle  # type: ignore[attr-defined]
    transforms.Rotate = _Rotate  # type: ignore[attr-defined]
    transforms.Translate = _Translate  # type: ignore[attr-defined]
    root.renderer = renderer  # type: ignore[attr-defined]
    root.transforms = transforms  # type: ignore[attr-defined]
    sys.modules["pytorch3d"] = root
    sys.modules["pytorch3d.renderer"] = renderer
    sys.modules["pytorch3d.transforms"] = transforms


def _load_geo4d_model(context: dict[str, Any]):
    root = str(context["geo4d_root"])
    ckpt_path = str(context["ckpt_path"])
    config_path = str(context.get("config_path") or (Path(root) / "configs" / "inference_geo4d.yaml"))
    gpu_no = int(context.get("gpu_no", 0))
    key = (root, ckpt_path, config_path, gpu_no)
    if key in _CACHE:
        _mark("using cached model")
        return _CACHE[key]

    _ensure_geo4d_path(root)
    _install_optional_dependency_shims()
    _install_pytorch3d_compat()
    _mark(f"load model begin root={root} ckpt={ckpt_path} config={config_path}")
    faulthandler.dump_traceback_later(120, repeat=True)
    from omegaconf import OmegaConf
    from lvdm.basics import disabled_train
    from utils.utils import instantiate_from_config
    from scripts.evaluation.test_geo4d import load_model_checkpoint

    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_no)
        model.perframe_ae = bool(context.get("perframe_ae", True))
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Geo4D checkpoint not found: {ckpt_path}")
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()
        _patch_open_clip_compat(model)

        pointmap_vae = None
        if "vae_path" in config:
            vae_path = Path(str(config["vae_path"]))
            if not vae_path.is_absolute():
                vae_path = Path(root) / vae_path
            pointmap_vae_config = config.pop("pointmap_vae_config", OmegaConf.create())
            pointmap_vae = instantiate_from_config(pointmap_vae_config).eval().cuda(gpu_no)
            pointmap_vae.train = disabled_train
            for param in pointmap_vae.parameters():
                param.requires_grad = False
            if not vae_path.is_file():
                raise FileNotFoundError(f"Geo4D VAE checkpoint not found: {vae_path}")
            vae_weights = importlib.import_module("torch").load(str(vae_path), map_location="cpu")["state_dict"]
            new_state = OrderedDict(
                (k[6:], v) for k, v in vae_weights.items() if k.startswith("model.")
            )
            pointmap_vae.load_state_dict(new_state, strict=True)
    finally:
        os.chdir(old_cwd)
        faulthandler.cancel_dump_traceback_later()

    _CACHE[key] = (model, config, pointmap_vae)
    _mark("load model done")
    return _CACHE[key]


def _patch_open_clip_compat(module: Any) -> None:
    for child in module.modules() if hasattr(module, "modules") else []:
        open_clip_model = getattr(child, "model", None)
        if open_clip_model is not None and hasattr(open_clip_model, "attn_mask"):
            open_clip_model.attn_mask = None
        visual = getattr(child, "visual", None)
        if visual is None:
            continue
        if not hasattr(visual, "input_patchnorm"):
            visual.input_patchnorm = False


def _read_tum_trajectory(path: str | Path) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    poses = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Expected TUM line with 8 values, got {len(parts)} in {path}")
            _, tx, ty, tz, qw, qx, qy, qz = [float(v) for v in parts]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = [tx, ty, tz]
            poses.append(c2w)
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return np.stack(poses, axis=0)


def _write_video(image_paths: list[str], video_path: Path, height: int, width: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
        frames.append(np.asarray(image, dtype=np.uint8))
    imageio.mimwrite(video_path, frames, fps=8, macro_block_size=1)


def _sequence_key(sequence_name: str, image_paths: list[str]) -> str:
    digest = hashlib.sha1("\n".join(image_paths).encode("utf-8")).hexdigest()[:10]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sequence_name).strip("_") or "sequence"
    return f"{safe}_{digest}"


def _infer_video_c2w(
    *,
    model: Any,
    config: Any,
    pointmap_vae: Any,
    video_path: Path,
    savedir: Path,
    context: dict[str, Any],
) -> np.ndarray:
    _install_optional_dependency_shims()
    _install_pytorch3d_compat()
    import torch
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from utils.funcs import load_video_batch
    from scripts.evaluation.test_geo4d import (
        denormalize_pc_bbox2,
        get_far_away_mask,
        get_sky_mask,
        image_guided_synthesis,
        post_optimization,
        raymap_to_camera_matrix,
    )

    seed_everything(int(context.get("seed", 123)))
    gpu_no = int(context.get("gpu_no", 0))
    height = int(context.get("height", 320))
    width = int(context.get("width", 512))
    ddim_steps = int(context.get("ddim_steps", 5))
    stride = int(context.get("stride", 4))
    old_cwd = os.getcwd()
    os.chdir(str(context["geo4d_root"]))
    try:
        video_frames, fps_list = load_video_batch(
            [str(video_path)],
            frame_stride=1,
            video_size=(height, width),
            video_frames=-1,
        )
        _, _, num_frames, frame_h, frame_w = video_frames.shape
        if num_frames < 16:
            pad = video_frames[:, :, -1:, :, :].expand(-1, -1, 16 - num_frames, -1, -1)
            video_frames = torch.cat([video_frames, pad], dim=2)
            num_frames = 16

        views = [{"img": video_frames[0, :, i, :, :], "idx": (i,)} for i in range(num_frames)]
        channels = model.model.diffusion_model.out_channels
        noise_shape = [1, channels, 16, frame_h // 8, frame_w // 8]
        videos_all = video_frames.cuda(gpu_no)
        prompts = ["Output a video that assigns each 3D location in the world a consistent color."]

        slice_list = [slice(start, start + 16, 1) for start in range(0, num_frames - 16 + 1, stride)]
        if not slice_list or slice_list[-1] != slice(num_frames - 16, num_frames, 1):
            slice_list.append(slice(num_frames - 16, num_frames, 1))

        pred_list = []
        view_list = []
        pnt_valid_mask = torch.ones((num_frames, frame_h, frame_w, 1), device=f"cuda:{gpu_no}") > 0
        for sl in slice_list:
            videos = videos_all[:, :, sl, :, :].clone()
            view_list.append(views[sl])
            batch_samples = image_guided_synthesis(
                model,
                prompts,
                videos,
                noise_shape,
                n_samples=1,
                ddim_steps=ddim_steps,
                ddim_eta=0.0,
                unconditional_guidance_scale=1.0,
                cfg_img=None,
                fs=fps_list[0],
                text_input=True,
                multiple_cond_cfg=False,
                loop=False,
                interp=False,
                timestep_spacing="uniform_trailing",
                guidance_rescale=0.7,
                pointmap_vae=pointmap_vae,
            )
            if batch_samples.shape[1] != 1:
                raise ValueError(f"Geo4D returned {batch_samples.shape[1]} variants; expected 1")
            batch_samples = batch_samples[:, 0]

            inverse_depthmap = None
            traj = None
            if model.modality == "pc_ray_cross_depth":
                raymap = batch_samples[:, 4:7]
                crossmap = batch_samples[:, 7:10]
                traj = raymap_to_camera_matrix(raymap, crossmap)
                inverse_depthmap = batch_samples[:, 10:11]
                inverse_depthmap = rearrange(inverse_depthmap, "b c t h w -> (b t) c h w")
                inverse_depthmap = rearrange(inverse_depthmap, "t c h w -> t h w c")
                inverse_depthmap = (inverse_depthmap + 1.0) / 2.0

            point_samples = batch_samples[:, :4]
            x_recon = rearrange(point_samples, "b c t h w -> (b t) c h w")
            confidence = torch.nn.Softplus()(x_recon[:, [-1], :, :])
            confidence = rearrange(confidence, "t c h w -> t h w c")
            if pointmap_vae is None:
                confidence = torch.ones_like(confidence)
            x_recon = x_recon[:, :-1, :, :]
            x_recon_hw = rearrange(x_recon, "t c h w -> t h w c")
            invalid_pts = get_sky_mask(x_recon_hw, sky_value=1.05, eps=0.35)
            invalid_pts = invalid_pts | get_far_away_mask(x_recon_hw, far_away_value=1.99)
            confidence[invalid_pts] = 999.0
            pnt_valid_mask[sl] = pnt_valid_mask[sl] * (~invalid_pts)
            inverse_confidence = 1 / confidence
            inverse_confidence[invalid_pts] = 0.0
            x_recon_hw = denormalize_pc_bbox2(x_recon_hw, alpha=2.0, beta=2.0)

            pred_pts = {"pts3d": x_recon_hw, "conf": inverse_confidence}
            if inverse_depthmap is not None:
                pred_pts["inverse_depthmap"] = inverse_depthmap
            if traj is not None:
                pred_pts["traj"] = traj
            pred_list.append(pred_pts)

        scene = post_optimization(
            view_list,
            pred_list,
            config.postprocess,
            conf_optimize=True,
            init_method="group",
            lr=0.03,
            opt_raydir=False,
        )
        seq_dir = savedir / video_path.stem
        seq_dir.mkdir(parents=True, exist_ok=True)
        traj_path = seq_dir / "pred_traj.txt"
        scene.save_tum_poses(str(traj_path))
        return _read_tum_trajectory(traj_path)
    finally:
        os.chdir(old_cwd)


def predict_sequence_c2w(image_paths: list[str], row: Any, context: dict[str, Any]) -> np.ndarray:
    """Run Geo4D on all input frames and return `(N,4,4)` c2w cameras."""
    model, config, pointmap_vae = _load_geo4d_model(context)
    height = int(context.get("height", 320))
    width = int(context.get("width", 512))
    scratch_root = Path(context["scratch_dir"]) / "geo4d_raw"
    key = _sequence_key(getattr(row, "sequence_name", "sequence"), image_paths)
    video_path = scratch_root / "videos" / f"{key}.mp4"
    _write_video(image_paths, video_path, height=height, width=width)
    pred_c2w = _infer_video_c2w(
        model=model,
        config=config,
        pointmap_vae=pointmap_vae,
        video_path=video_path,
        savedir=scratch_root / "outputs",
        context=context,
    )
    if pred_c2w.shape[0] < len(image_paths):
        raise ValueError(f"Geo4D returned {pred_c2w.shape[0]} poses for {len(image_paths)} frames")
    return pred_c2w[: len(image_paths)]


def predict_pair_c2w(
    image_paths: list[str],
    pair: tuple[int, int],
    row: Any,
    context: dict[str, Any],
) -> np.ndarray:
    """Run Geo4D with only two unique images, padded to its 16-frame clip size."""
    if len(image_paths) != 2:
        raise ValueError(f"Geo4D pair adapter expects exactly 2 images, got {len(image_paths)}")
    padded_paths = [image_paths[0]] * 8 + [image_paths[1]] * 8
    pred_c2w = predict_sequence_c2w(padded_paths, row=row, context=context)
    return pred_c2w[[0, -1]]

from __future__ import annotations

import sys
from typing import Any
from pathlib import Path
import importlib.util
import importlib.machinery
import faulthandler
import types
from collections import defaultdict

import numpy as np
from PIL import Image


_CACHE: dict[tuple[Any, ...], Any] = {}


def _mark(message: str) -> None:
    print(f"[video_world_model_adapter] {message}", flush=True)


def _wrap_marker(owner: Any, attr: str, label: str) -> None:
    original = getattr(owner, attr)
    if getattr(original, "_vwm_adapter_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any):
        detail = ""
        if label == "hf_hub_download":
            repo_id = kwargs.get("repo_id", args[0] if args else None)
            filename = kwargs.get("filename", args[1] if len(args) > 1 else None)
            detail = f" repo_id={repo_id} filename={filename}"
        _mark(f"{label} begin{detail}")
        result = original(*args, **kwargs)
        _mark(f"{label} done{detail}")
        return result

    wrapped._vwm_adapter_wrapped = True  # type: ignore[attr-defined]
    setattr(owner, attr, wrapped)


def _ensure_lightweight_wan_packages(root: str) -> None:
    wan_dir = Path(root) / "algorithms" / "wan"
    package_specs = {
        "algorithms.wan": wan_dir,
        "algorithms.wan.modules": wan_dir / "modules",
        "algorithms.wan.utils": wan_dir / "utils",
    }
    for package_name, package_path in package_specs.items():
        module = sys.modules.get(package_name)
        if module is not None:
            continue
        module = types.ModuleType(package_name)
        module.__file__ = str(package_path / "__init__.py")
        module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        module.__package__ = package_name
        module.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        sys.modules[package_name] = module


def _install_lightweight_re10k_module() -> None:
    if "datasets.re10k" in sys.modules:
        return
    import cv2
    import torch
    import torch.nn.functional as F

    datasets_module = sys.modules.get("datasets")
    if datasets_module is None:
        datasets_module = types.ModuleType("datasets")
        datasets_module.__path__ = []  # type: ignore[attr-defined]
        datasets_module.__package__ = "datasets"
        datasets_module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None, is_package=True)
        sys.modules["datasets"] = datasets_module

    def _intersect_skew_lines(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        directions = F.normalize(directions, dim=-1)
        eye = torch.eye(3, dtype=points.dtype, device=points.device).unsqueeze(0)
        proj = eye - directions.unsqueeze(-1) * directions.unsqueeze(-2)
        lhs = proj.sum(dim=0)
        rhs = torch.bmm(proj, points.unsqueeze(-1)).sum(dim=0).squeeze(-1)
        return torch.linalg.lstsq(lhs, rhs).solution

    def raymap_to_camera(
        raymaps: torch.Tensor,
        raymap_scale: float = 1.0,
        max_points: int = 5000,
        ransac_reproj_threshold: float = 1.0,
        min_abs_z: float = 1e-6,
    ):
        if raymaps.dim() == 3:
            raymaps = raymaps.unsqueeze(0)
            squeeze_output = True
        elif raymaps.dim() == 4:
            squeeze_output = False
        else:
            raise ValueError(f"Expected raymaps with 3 or 4 dims, got shape {raymaps.shape}.")
        if raymaps.shape[1] != 6:
            raise ValueError(f"Expected channel dim to be 6, got {raymaps.shape[1]}.")

        raymaps = raymaps.clone()
        raymaps[:, 3:] = raymaps[:, 3:] * raymap_scale
        batch_size, _, height, width = raymaps.shape
        device = raymaps.device
        dtype = raymaps.dtype

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        pixel_x = (xx + 0.5).reshape(-1)
        pixel_y = (yy + 0.5).reshape(-1)

        intrinsics_list = []
        extrinsics_list = []
        for batch_idx in range(batch_size):
            rays_d = F.normalize(raymaps[batch_idx, :3], dim=0)
            rays_m = raymaps[batch_idx, 3:]
            rays_d_flat = rays_d.permute(1, 2, 0).reshape(-1, 3)
            rays_m_flat = rays_m.permute(1, 2, 0).reshape(-1, 3)
            ray_points = torch.cross(rays_d_flat, rays_m_flat, dim=-1)
            cam_center = _intersect_skew_lines(ray_points, rays_d_flat)

            valid = (
                torch.isfinite(rays_d_flat).all(dim=-1)
                & torch.isfinite(rays_m_flat).all(dim=-1)
                & (rays_d_flat[:, 2].abs() > min_abs_z)
            )
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() < 8:
                raise ValueError(f"Not enough valid rays to estimate homography: {valid_idx.numel()}.")
            if valid_idx.numel() > max_points:
                sample_idx = torch.linspace(
                    0,
                    valid_idx.numel() - 1,
                    steps=max_points,
                    dtype=torch.float32,
                    device=device,
                ).long()
                valid_idx = valid_idx[sample_idx]

            rays_d_sel = rays_d_flat[valid_idx]
            src = torch.stack(
                [rays_d_sel[:, 0] / rays_d_sel[:, 2], rays_d_sel[:, 1] / rays_d_sel[:, 2]],
                dim=-1,
            )
            dst = torch.stack([pixel_x[valid_idx], pixel_y[valid_idx]], dim=-1)
            src_np = src.detach().cpu().numpy().astype(np.float32)
            dst_np = dst.detach().cpu().numpy().astype(np.float32)

            homography, _ = cv2.findHomography(
                src_np,
                dst_np,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
            )
            if homography is None:
                homography, _ = cv2.findHomography(src_np, dst_np, method=0)
            if homography is None:
                raise ValueError("Homography estimation failed from ray directions.")

            projection = np.concatenate(
                [homography.astype(np.float64), np.zeros((3, 1), dtype=np.float64)],
                axis=1,
            )
            k_np, r_w2c_np, *_ = cv2.decomposeProjectionMatrix(projection)
            k_np = k_np / (k_np[2, 2] + 1e-12)
            if k_np[0, 0] < 0:
                k_np[:, 0] *= -1
                r_w2c_np[0, :] *= -1
            if k_np[1, 1] < 0:
                k_np[:, 1] *= -1
                r_w2c_np[1, :] *= -1
            u_mat, _, vt_mat = np.linalg.svd(r_w2c_np)
            r_w2c_np = u_mat @ vt_mat
            if np.linalg.det(r_w2c_np) < 0:
                u_mat[:, -1] *= -1
                r_w2c_np = u_mat @ vt_mat

            intrinsics = torch.from_numpy(k_np).to(device=device, dtype=dtype)
            r_c2w = torch.from_numpy(r_w2c_np.T).to(device=device, dtype=dtype)
            extrinsics = torch.eye(4, dtype=dtype, device=device)
            extrinsics[:3, :3] = r_c2w
            extrinsics[:3, 3] = cam_center
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics_out = torch.stack(intrinsics_list, dim=0)
        extrinsics_out = torch.stack(extrinsics_list, dim=0)
        if squeeze_output:
            intrinsics_out = intrinsics_out.squeeze(0)
            extrinsics_out = extrinsics_out.squeeze(0)
        return intrinsics_out, extrinsics_out

    re10k_module = types.ModuleType("datasets.re10k")
    re10k_module.raymap_to_camera = raymap_to_camera  # type: ignore[attr-defined]
    re10k_module.__package__ = "datasets"
    re10k_module.__spec__ = importlib.machinery.ModuleSpec("datasets.re10k", loader=None)
    sys.modules["datasets.re10k"] = re10k_module
    setattr(datasets_module, "re10k", re10k_module)


def _patch_diffusers_package_scan() -> None:
    """Avoid slow importlib.metadata package-map scans on shared filesystems.

    diffusers only needs this map to resolve occasional package-to-distribution
    aliases for version checks. Returning an empty defaultdict preserves normal
    importlib.util.find_spec and version(pkg_name) behavior while skipping a
    filesystem-heavy scan that can stall on the cluster.
    """
    try:
        import importlib.metadata as importlib_metadata

        importlib_metadata.packages_distributions = lambda: defaultdict(list)  # type: ignore[assignment]
    except Exception:
        pass
    try:
        import importlib_metadata as importlib_metadata_backport

        importlib_metadata_backport.packages_distributions = lambda: defaultdict(list)  # type: ignore[assignment]
    except Exception:
        pass


def _install_vwm_diagnostics(root: str) -> Any:
    _ensure_lightweight_wan_packages(root)
    _install_lightweight_re10k_module()
    _patch_diffusers_package_scan()
    import algorithms.wan.wan_t2v_ray_depth_mot as wan_mod

    _wrap_marker(wan_mod, "hf_hub_download", "hf_hub_download")
    _wrap_marker(wan_mod, "umt5_xxl", "umt5_xxl")
    _wrap_marker(wan_mod, "HuggingfaceTokenizer", "HuggingfaceTokenizer")
    _wrap_marker(wan_mod, "video_vae_factory", "video_vae_factory")
    _wrap_marker(wan_mod, "video_vae_flf_factory_v2", "video_vae_flf_factory_v2")
    _wrap_marker(wan_mod.WanModel, "from_config", "WanModel.from_config")
    _wrap_marker(wan_mod.WanModel, "from_t2v_pretrained", "WanModel.from_t2v_pretrained")
    _wrap_marker(wan_mod.BaseWanModel, "from_config", "BaseWanModel.from_config")
    if not getattr(wan_mod.WanTextToVideoRayDepthMoT._load_tuned_state_dict, "_vwm_adapter_wrapped", False):
        original_load_tuned = wan_mod.WanTextToVideoRayDepthMoT._load_tuned_state_dict

        def wrapped_load_tuned(self, *args: Any, **kwargs: Any):
            _mark(f"load tuned checkpoint begin path={self.cfg.model.tuned_ckpt_path}")
            result = original_load_tuned(self, *args, **kwargs)
            _mark(f"load tuned checkpoint done keys={len(result)}")
            return result

        wrapped_load_tuned._vwm_adapter_wrapped = True  # type: ignore[attr-defined]
        wan_mod.WanTextToVideoRayDepthMoT._load_tuned_state_dict = wrapped_load_tuned
    return wan_mod


def _center_crop_to_aspect(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    src_width, src_height = image.size
    target_aspect = target_width / target_height
    src_aspect = src_width / src_height
    if src_aspect > target_aspect:
        crop_width = int(round(src_height * target_aspect))
        left = max(0, (src_width - crop_width) // 2)
        return image.crop((left, 0, left + crop_width, src_height))
    crop_height = int(round(src_width / target_aspect))
    top = max(0, (src_height - crop_height) // 2)
    return image.crop((0, top, src_width, top + crop_height))


def _load_rgb(path: str, height: int, width: int, preprocess_mode: str = "direct_resize"):
    import torch

    image = Image.open(path).convert("RGB")
    if preprocess_mode == "center_crop_resize":
        image = _center_crop_to_aspect(image, target_width=width, target_height=height)
    elif preprocess_mode != "direct_resize":
        raise ValueError(f"Unknown VWM image_preprocess_mode={preprocess_mode}")
    image = image.resize((width, height), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _load_model(context: dict[str, Any]):
    root = str(context["video_world_model_root"])
    device = str(context.get("device", "cuda"))
    ckpt_path = str(context["ckpt_path"])
    sample_steps = int(context.get("sample_steps", 40))
    diffusion_mode = str(context.get("diffusion_mode", "ray_depth_prediction_firstlast"))
    lang_guidance = float(context.get("lang_guidance", 0.0))
    hist_guidance = float(context.get("hist_guidance", 1.0))
    override_n_frames = context.get("override_n_frames")
    override_n_frames = int(override_n_frames) if override_n_frames is not None else None
    override_height = context.get("override_height")
    override_width = context.get("override_width")
    override_height = int(override_height) if override_height is not None else None
    override_width = int(override_width) if override_width is not None else None
    key = (
        root,
        ckpt_path,
        device,
        sample_steps,
        diffusion_mode,
        lang_guidance,
        hist_guidance,
        override_n_frames,
        override_height,
        override_width,
    )
    if key in _CACHE:
        _mark("using cached model")
        return _CACHE[key]

    _mark(f"load model begin root={root} ckpt={ckpt_path} device={device}")
    if root not in sys.path:
        sys.path.insert(0, root)
    hydra_config_dir = str(context.get("hydra_config_dir") or (Path(ckpt_path).parent.parent / ".hydra"))

    _mark("import lightweight eval module begin")
    module_path = Path(root) / "scripts" / "eval_ours_geo4d_benchmarks.py"
    spec = importlib.util.spec_from_file_location("_vwm_eval_ours_geo4d_benchmarks", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    _mark("import lightweight eval module done")

    _mark(f"load hydra config begin hydra_config_dir={hydra_config_dir}")
    from omegaconf import OmegaConf

    full_cfg = OmegaConf.load(Path(hydra_config_dir) / "config.yaml")
    if override_n_frames is not None or override_height is not None or override_width is not None:
        OmegaConf.set_struct(full_cfg, False)
    if override_n_frames is not None:
        full_cfg.dataset.n_frames = override_n_frames
    if override_height is not None:
        full_cfg.dataset.height = override_height
    if override_width is not None:
        full_cfg.dataset.width = override_width
    algo_cfg = full_cfg.algorithm
    OmegaConf.set_struct(algo_cfg, False)
    if override_n_frames is not None:
        algo_cfg.n_frames = override_n_frames
    if override_height is not None:
        algo_cfg.height = override_height
    if override_width is not None:
        algo_cfg.width = override_width
    algo_cfg.model.tuned_ckpt_path = ckpt_path
    # Pairwise geometry inference uses an empty prompt. Supplying an empty
    # prompt embedding avoids loading the multi-GB T5 text encoder.
    algo_cfg.load_prompt_embed = True
    algo_cfg.text_encoder.compile = False
    algo_cfg.vae.compile = False
    algo_cfg.model.compile = False
    algo_cfg.gradient_checkpointing_rate = 0.0
    algo_cfg.sample_steps = sample_steps
    _mark("load hydra config done")

    _mark("import WanTextToVideoRayDepthMoT begin")
    wan_mod = _install_vwm_diagnostics(root)
    WanTextToVideoRayDepthMoT = wan_mod.WanTextToVideoRayDepthMoT
    _mark("import WanTextToVideoRayDepthMoT done")
    _mark("construct model begin")
    model = WanTextToVideoRayDepthMoT(algo_cfg)
    _mark("construct model done")
    _mark("configure_model begin")
    model.configure_model()
    _mark("configure_model done")
    _mark(f"move model to {device} begin")
    model = model.train(False).to(device)
    model.vae_scale = [model.vae_mean, model.vae_inv_std]
    _mark("move model done")
    model.hist_guidance = hist_guidance
    model.lang_guidance = lang_guidance
    model.sample_steps = sample_steps
    _CACHE[key] = (base, model, None)
    return _CACHE[key]


def predict_pair_c2w(
    image_paths: list[str],
    pair: tuple[int, int],
    row: Any,
    context: dict[str, Any],
):
    """Return `(2,4,4)` predicted c2w for one Pi3 manifest image pair."""
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    _mark("predict pair import torch begin")
    import torch
    _mark("predict pair import torch done")

    base, model, _ = _load_model(context)
    _mark(f"predict pair begin pair={pair}")

    height = int(model.height)
    width = int(model.width)
    preprocess_mode = str(context.get("image_preprocess_mode", "direct_resize"))
    _mark(f"image preprocessing mode={preprocess_mode} size={width}x{height}")
    videos = torch.stack([_load_rgb(path, height, width, preprocess_mode) for path in image_paths], dim=0)
    rgb_input = videos.permute(1, 0, 2, 3).unsqueeze(0)
    _mark("run_inference begin")
    device = str(context.get("device", "cuda"))
    with torch.no_grad():
        _, pred_ray_d, pred_ray_m = _run_inference_no_text(model, rgb_input, device)
    _mark("run_inference done")
    _, pred_c2w = base.raymaps_to_cameras(pred_ray_d, pred_ray_m)
    _mark("raymap_to_camera done")
    pred_c2w = pred_c2w[[0, -1]]
    return pred_c2w.detach().cpu().numpy()


def predict_sequence_c2w(
    image_paths: list[str],
    row: Any,
    context: dict[str, Any],
):
    """Return `(N,4,4)` predicted c2w for one full-frame VWM inference."""
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    _mark("predict sequence import torch begin")
    import torch
    _mark("predict sequence import torch done")

    base, model, _ = _load_model(context)
    _mark(f"predict sequence begin frames={len(image_paths)} sequence={getattr(row, 'sequence_name', '')}")

    height = int(model.height)
    width = int(model.width)
    preprocess_mode = str(context.get("image_preprocess_mode", "direct_resize"))
    _mark(f"image preprocessing mode={preprocess_mode} size={width}x{height}")
    videos = torch.stack([_load_rgb(path, height, width, preprocess_mode) for path in image_paths], dim=0)
    rgb_input = videos.permute(1, 0, 2, 3).unsqueeze(0)
    _mark("run_sequence_inference begin")
    device = str(context.get("device", "cuda"))
    with torch.no_grad():
        _, pred_ray_d, pred_ray_m = _run_sequence_inference_no_text(model, rgb_input, device)
    _mark("run_sequence_inference done")
    _, pred_c2w = base.raymaps_to_cameras(pred_ray_d, pred_ray_m)
    _mark("raymap_to_camera done")
    return pred_c2w.detach().cpu().numpy()


def _run_inference_no_text(model: Any, rgb_input_bcthw: Any, device: str):
    import torch

    pair_rgb = rgb_input_bcthw.to(device=device, dtype=model.dtype)
    bsz, channels, t_pair, height, width = pair_rgb.shape
    if t_pair != 2:
        raise ValueError(f"Pairwise VWM adapter expects exactly 2 frames, got {t_pair}")
    t_pix = int(model.n_frames)
    rgb = torch.zeros((bsz, channels, t_pix, height, width), device=device, dtype=model.dtype)
    rgb[:, :, 0] = pair_rgb[:, :, 0]
    rgb[:, :, -1] = pair_rgb[:, :, 1]
    placeholder_ray = torch.zeros(
        bsz, t_pix, 6, model.lat_h, model.lat_w, device=device, dtype=model.dtype
    )
    placeholder_depth = torch.zeros(
        bsz, t_pix, 1, height, width, device=device, dtype=model.dtype
    )
    prompt_dim = int(model.cfg.text_encoder.text_dim)
    empty_prompt = torch.zeros((0, prompt_dim), device=device, dtype=model.dtype)
    batch = {
        "videos": rgb.permute(0, 2, 1, 3, 4),
        "raymaps": placeholder_ray,
        "depths": placeholder_depth,
        "prompts": [""],
        "prompt_embeds": [empty_prompt],
        "subset_id": ["re10k"],
        "has_raymap": torch.tensor([False], device=device),
        "has_depth": torch.tensor([False], device=device),
    }

    t_lat = model.lat_t // 4
    hist_indexes = torch.tensor([0, t_lat - 1], device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        video_pred = model.sample_seq(batch, hist_indexes=hist_indexes)
    tp = video_pred.shape[1] // 4
    pred_ray_d = video_pred[0, tp : 2 * tp].float().cpu()
    pred_ray_m = video_pred[0, 2 * tp : 3 * tp].float().cpu()
    pred_depth = video_pred[0, 3 * tp :].float().cpu()
    pred_disp = pred_depth.mean(dim=1, keepdim=True)
    return pred_disp, pred_ray_d, pred_ray_m


def _run_sequence_inference_no_text(model: Any, rgb_input_bcthw: Any, device: str):
    import torch

    rgb = rgb_input_bcthw.to(device=device, dtype=model.dtype)
    bsz, channels, t_pix, height, width = rgb.shape
    if t_pix != int(model.n_frames):
        raise ValueError(f"Full-sequence VWM adapter expects {model.n_frames} frames, got {t_pix}")
    if channels != 3:
        raise ValueError(f"Full-sequence VWM adapter expects RGB input, got {channels} channels")
    placeholder_ray = torch.zeros(
        bsz, t_pix, 6, model.lat_h, model.lat_w, device=device, dtype=model.dtype
    )
    placeholder_depth = torch.zeros(
        bsz, t_pix, 1, height, width, device=device, dtype=model.dtype
    )
    prompt_dim = int(model.cfg.text_encoder.text_dim)
    empty_prompt = torch.zeros((0, prompt_dim), device=device, dtype=model.dtype)
    batch = {
        "videos": rgb.permute(0, 2, 1, 3, 4),
        "raymaps": placeholder_ray,
        "depths": placeholder_depth,
        "prompts": [""],
        "prompt_embeds": [empty_prompt],
        "subset_id": ["re10k"],
        "has_raymap": torch.tensor([False], device=device),
        "has_depth": torch.tensor([False], device=device),
    }

    t_lat = model.lat_t // 4
    hist_indexes = torch.arange(t_lat, device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        video_pred = model.sample_seq(batch, hist_indexes=hist_indexes)
    tp = video_pred.shape[1] // 4
    pred_ray_d = video_pred[0, tp : 2 * tp].float().cpu()
    pred_ray_m = video_pred[0, 2 * tp : 3 * tp].float().cpu()
    pred_depth = video_pred[0, 3 * tp :].float().cpu()
    pred_disp = pred_depth.mean(dim=1, keepdim=True)
    return pred_disp, pred_ray_d, pred_ray_m

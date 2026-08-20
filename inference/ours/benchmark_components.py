"""Optional components for the standardized OURS benchmark pipeline.

The default path remains the historical benchmark behavior.  Callers may opt
into the Wan2.2 fine-tuned depth decoder, prediction-only shared-intrinsics
camera recovery, and the frozen sparse BA configuration verified in July 2026.

Confidence is returned only as an artifact.  It is deliberately not consumed
by depth scoring, feature tracking, sparse-track initialization, or BA.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.aria import raymap_to_camera


VERIFIED_BA_CORE_SHA256 = (
    "07d4c86a71c7751f16db5538ff91db84a781b045580dc5c17219af415cc4606c"
)

# Frozen convergence-qualified configuration from
# docs/worklog/20260716-233733-ba-50x2-benchmark.html and
# docs/worklog/20260717-012410-ba-shared-intrinsics.html.
VERIFIED_BA_CONFIG: dict[str, Any] = {
    "feature_method": "sift",
    "max_features": 5000,
    "max_pair_matches": 2500,
    "match_ratio": 0.75,
    "ransac_thresh_px": 1.5,
    "min_track_len": 2,
    "max_tracks": 500,
    "min_depth_units": 0.001,
    "max_depth_units": 1.0e9,
    "reproj_sigma_px": 2.0,
    "pose_rot_prior_sigma_deg": 0.05,
    "pose_trans_prior_sigma_units": 0.002,
    "point_prior_sigma_units": 0.05,
    "loss_f_scale": 1.0e6,
    "max_nfev": 120,
}


@dataclasses.dataclass(frozen=True)
class CameraRecovery:
    """Prediction-only cameras and audit metadata."""

    intrinsics: np.ndarray
    c2w: np.ndarray
    legacy_c2w: np.ndarray
    summary: dict[str, Any]


def save_rng_state(path: str | os.PathLike[str]) -> None:
    """Atomically save global RNG streams after one benchmark item.

    Ordered resume can then skip a completed item without changing the random
    stream seen by the next item. This keeps component ablations paired after
    a Slurm preemption.
    """

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary)
    os.replace(temporary, destination)


def restore_rng_state(path: str | os.PathLike[str]) -> None:
    """Restore global RNG streams saved by :func:`save_rng_state`."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def attach_finetuned_depth_vae_v2(
    model: torch.nn.Module,
    algo_cfg: Any,
    checkpoint: str | os.PathLike[str],
) -> dict[str, Any]:
    """Attach the verified Wan2.2 depth decoder and confidence head.

    The base VAE locator and load order intentionally match
    ``scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py``.
    """

    from huggingface_hub import hf_hub_download

    from algorithms.wan.modules.depth_vae import depth_video_vae_flf_factory

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Depth VAE checkpoint does not exist: {checkpoint_path}")
    vae_version = str(getattr(algo_cfg.vae, "version", ""))
    if vae_version != "wan2_2":
        raise ValueError(
            "The benchmark depth decoder is the 5B/Wan2.2 v2 implementation; "
            f"algorithm.vae.version={vae_version!r}"
        )
    if str(algo_cfg.diffusion_forcing.mode) != "ray_depth_prediction_firstlast":
        raise ValueError(
            "The fine-tuned depth decoder requires "
            "diffusion_forcing.mode=ray_depth_prediction_firstlast"
        )

    vae_locator = Path(str(algo_cfg.vae.ckpt_path))
    vae_model_path = hf_hub_download(
        repo_id=str(vae_locator.parent), filename=vae_locator.name
    )
    depth_vae = depth_video_vae_flf_factory(
        pretrained_path=vae_model_path,
        depth_checkpoint=str(checkpoint_path),
        z_dim=int(algo_cfg.vae.z_dim),
        version=vae_version,
    )
    vae_dtype = next(model.vae.parameters()).dtype
    model.depth_vae = depth_vae.eval().requires_grad_(False).to(dtype=vae_dtype)
    return {
        "enabled": True,
        "version": vae_version,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "base_vae": str(algo_cfg.vae.ckpt_path),
        "confidence_saved_only": True,
        "confidence_filters_depth_metrics": False,
        "confidence_filters_ba": False,
    }


def recover_predicted_cameras(
    pred_raymaps: torch.Tensor,
    *,
    shared_intrinsics: bool,
    model_native_units: bool,
    norm_meta: dict = None,
    shared_max_points_per_frame: int = 600,
    shared_max_nfev: int = 80,
) -> CameraRecovery:
    """Recover cameras from predicted rays without consulting camera GT.

    Historical benchmark output uses ``raymap_to_camera``'s legacy Aria moment
    denormalization.  Sparse BA must instead share units with decoded depth, so
    BA/shared-K paths recover camera centers directly in model-native units.
    Sim(3) pose scoring removes this global unit choice.
    """

    raymaps = pred_raymaps.detach().float()
    legacy_k, legacy_c2w_t = raymap_to_camera(raymaps)
    legacy_c2w = legacy_c2w_t.detach().cpu().numpy().astype(np.float64)
    # ARM-AWARE TRANSLATION. raymap_to_camera assumes channels 3:6 are Plucker moments. That is
    # true for F / global_metric / vggt, but NOT for the other ablation arms: scheme C stores a
    # mu-law companded camera origin, vggt_omega/pi3/da3 store the origin broadcast spatially,
    # and genception stores it only inside the centred "Rothko" rectangle. Decoding any of those
    # as a moment yields a WRONG trajectory with no error raised. K and R still come from the
    # direction channels either way, so only the translation column is replaced here.
    # NOTE pi3: its directions are per-frame LOCAL, so R is frame-invariant and that arm carries
    # no rotation at all -- its rotation metrics are structurally degenerate, not merely poor.
    _meta = dict(norm_meta or {})
    if _meta.get("scale_mode"):
        from datasets._geometry_builder import (
            origins_from_normalized_raymap, ray_encoding_for, PLUCKER,
        )

        class _Cfg(dict):
            def get(self, k, d=None):
                return dict.get(self, k, d)

        _cfg = _Cfg({k: v for k, v in _meta.items() if v is not None})
        if ray_encoding_for(_cfg) != PLUCKER:
            # scale=1.0: Sim(3) pose alignment absorbs the global scalar; the SEMANTICS of the
            # channel are what must be right here.
            t_arm = origins_from_normalized_raymap(raymaps, 1.0, _cfg)
            legacy_c2w[:, :3, 3] = t_arm.detach().cpu().numpy().astype(np.float64)
    if not (shared_intrinsics or model_native_units):
        return CameraRecovery(
            intrinsics=legacy_k.detach().cpu().numpy().astype(np.float64),
            c2w=legacy_c2w,
            legacy_c2w=legacy_c2w,
            summary={
                "mode": "independent_per_frame_legacy",
                "model_native_units": False,
                "prediction_only": True,
            },
        )

    native_k, native_c2w_t = raymap_to_camera(
        raymaps, moments_are_normalized=False
    )
    native_intrinsics = native_k.detach().cpu().numpy().astype(np.float64)
    native_c2w = native_c2w_t.detach().cpu().numpy().astype(np.float64)
    if not shared_intrinsics:
        return CameraRecovery(
            intrinsics=native_intrinsics,
            c2w=native_c2w,
            legacy_c2w=legacy_c2w,
            summary={
                "mode": "independent_per_frame_model_native",
                "model_native_units": True,
                "prediction_only": True,
            },
        )

    if raymaps.ndim != 4 or len(raymaps) < 2:
        raise ValueError(
            "Shared-intrinsics recovery requires at least two TCHW raymap frames"
        )
    from scripts.shared_intrinsics_camera import recover_cameras_shared_intrinsics

    result = recover_cameras_shared_intrinsics(
        raymaps[:, :3].cpu().numpy(),
        raymaps[:, 3:].cpu().numpy(),
        native_intrinsics,
        max_points_per_frame=shared_max_points_per_frame,
        max_nfev=shared_max_nfev,
    )
    return CameraRecovery(
        intrinsics=result.intrinsics.astype(np.float64),
        c2w=result.extrinsics.astype(np.float64),
        legacy_c2w=legacy_c2w,
        summary={
            "mode": "shared_intrinsics_model_native",
            "model_native_units": True,
            "prediction_only": True,
            **result.summary(),
        },
    )


def normalized_disparity_to_depth_units(
    depth_norm: np.ndarray, eps: float = 1.0e-6
) -> np.ndarray:
    """Invert the model disparity encoding in model-native joint units.

    This is the all-finite, unfiltered conversion used by the verified BA
    benchmark.  It intentionally has no confidence mask or percentile clip.
    """

    disp = np.clip(
        np.asarray(depth_norm, dtype=np.float32) * 0.5 + 0.5, 0.0, 1.0
    )
    return (1.0 / np.clip(disp, eps, 1.0) - 1.0).astype(np.float64)


def _load_verified_ba_core(repo_root: Path):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "The exact verified BA core requires Python 3.10; "
            f"got {sys.version}"
        )
    path = (
        repo_root
        / "scripts"
        / "__pycache__"
        / "bundle_adjust_re10k_pred_outputs.cpython-310.pyc"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Verified BA core not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != VERIFIED_BA_CORE_SHA256:
        raise RuntimeError(
            "Verified BA core checksum mismatch: "
            f"expected {VERIFIED_BA_CORE_SHA256}, got {digest} ({path})"
        )
    module_name = "_benchmark_verified_ba_core"
    loader = importlib.machinery.SourcelessFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"Could not load verified BA core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module, path


def _pose_update_summary(initial: np.ndarray, refined: np.ndarray) -> dict[str, float]:
    angles, centers = [], []
    for before, after in zip(initial, refined):
        relative = after[:3, :3] @ before[:3, :3].T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(cosine))))
        centers.append(float(np.linalg.norm(after[:3, 3] - before[:3, 3])))
    return {
        "rotation_update_mean_deg": float(np.mean(angles)),
        "rotation_update_max_deg": float(np.max(angles)),
        "center_update_mean_units": float(np.mean(centers)),
        "center_update_max_units": float(np.max(centers)),
    }


def run_verified_sparse_bundle_adjustment(
    *,
    rgb_tchw: np.ndarray,
    depth_norm: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    output_dir: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run exact verified fixed-depth/intrinsics sparse BA.

    Dense decoded depth is immutable.  Frame 0 is the gauge camera.  Remaining
    camera poses and sparse 3D track positions are jointly optimized.
    """

    root = Path(
        repo_root
        or os.environ.get("VWM_REPO", Path(__file__).resolve().parents[4])
    ).resolve()
    core, core_path = _load_verified_ba_core(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(rgb_tchw, dtype=np.float32)
    depth = normalized_disparity_to_depth_units(depth_norm)
    camera_k = np.asarray(intrinsics, dtype=np.float64)
    camera_c2w = np.asarray(c2w, dtype=np.float64)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB [T,3,H,W], got {rgb.shape}")
    if depth.ndim != 3 or depth.shape[0] != rgb.shape[0]:
        raise ValueError(f"Expected matching depth [T,H,W], got {depth.shape}")
    if camera_k.shape != (len(rgb), 3, 3):
        raise ValueError(f"Expected intrinsics {(len(rgb), 3, 3)}, got {camera_k.shape}")
    if camera_c2w.shape != (len(rgb), 4, 4):
        raise ValueError(f"Expected c2w {(len(rgb), 4, 4)}, got {camera_c2w.shape}")

    camera_k, rescale_xy = core._maybe_rescale_intrinsics(
        camera_k, image_hw=depth.shape[-2:]
    )
    rgb_u8 = np.clip((rgb + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    data = core.SampleData(
        sample_dir=output_path,
        rgb_hwc=np.transpose(rgb_u8, (0, 2, 3, 1)),
        rgb_tchw=rgb,
        depth_m=depth,
        intrinsics=camera_k,
        c2w=camera_c2w,
        frame_indices=np.arange(len(rgb), dtype=np.int64),
    )
    cfg = VERIFIED_BA_CONFIG
    features, resolved_method = core.extract_features(
        data.rgb_hwc, cfg["feature_method"], cfg["max_features"]
    )
    tracks_xy, match_stats = core.build_tracks(
        features,
        feature_method=resolved_method,
        ratio=cfg["match_ratio"],
        ransac_thresh=cfg["ransac_thresh_px"],
        max_pair_matches=cfg["max_pair_matches"],
    )
    ba_tracks = core.prepare_ba_tracks(
        data,
        tracks_xy,
        min_track_len=cfg["min_track_len"],
        max_tracks=cfg["max_tracks"],
        min_depth_m=cfg["min_depth_units"],
        max_depth_m=cfg["max_depth_units"],
    )
    ba_tracks.match_stats = match_stats
    refined_c2w, refined_points, ba_summary = core.run_bundle_adjustment(
        data,
        ba_tracks,
        reproj_sigma_px=cfg["reproj_sigma_px"],
        pose_rot_prior_sigma_deg=cfg["pose_rot_prior_sigma_deg"],
        pose_trans_prior_sigma_m=cfg["pose_trans_prior_sigma_units"],
        point_prior_sigma_m=cfg["point_prior_sigma_units"],
        loss_f_scale=cfg["loss_f_scale"],
        max_nfev=cfg["max_nfev"],
        verbose=0,
    )
    np.save(output_path / "ba_sparse_points.npy", refined_points.astype(np.float64))
    summary = {
        "mode": "verified_sparse_points_fixed_dense_depth_and_intrinsics",
        "optimized_variables": ["camera_poses_except_frame_0", "sparse_3d_points"],
        "fixed_variables": ["frame_0_pose", "dense_decoder_depth", "intrinsics"],
        "confidence_used": False,
        "model_native_units": True,
        "core_path": str(core_path),
        "core_sha256": VERIFIED_BA_CORE_SHA256,
        "parameters": dict(cfg),
        "intrinsics_rescale_xy": [float(value) for value in rescale_xy],
        "features_per_frame": [int(len(item.keypoints_xy)) for item in features],
        "resolved_feature_method": resolved_method,
        "n_raw_tracks": int(ba_tracks.n_raw_tracks),
        "n_used_tracks": int(ba_tracks.n_used_tracks),
        "n_observations": int(len(ba_tracks.points_2d)),
        "n_sparse_points": int(len(refined_points)),
        "matches": [vars(item) for item in match_stats],
        "pose_update": _pose_update_summary(camera_c2w, refined_c2w),
        "solver": ba_summary,
        "artifacts": {"sparse_points": str(output_path / "ba_sparse_points.npy")},
    }
    with (output_path / "ba_summary.json").open("w") as handle:
        json.dump(_json_value(summary), handle, indent=2, allow_nan=True)
    return refined_c2w.astype(np.float64), _json_value(summary)

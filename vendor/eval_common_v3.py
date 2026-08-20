"""
V3 metric wrappers, shared across all baselines + ours.

Pose:  Geo4D `dust3r/utils/vo_eval.py:eval_metrics` (ATE/RPE_trans/RPE_rot via evo).
Depth: Geo4D `dust3r/depth_eval.py:depth_evaluation` with align_with_lad2=True
       (per-video global LAD2 scale-shift fit).

Both metric implementations are vendored verbatim under `geo4d_eval/`.
This module exposes a tiny convenience layer so each per-method evaluator
just has to provide (pred_c2w, gt_c2w) for pose and (pred_T_H_W, gt_T_H_W)
for depth.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Vendored eval code (do not modify).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from geo4d_eval import vo_eval as _vo_eval  # noqa: E402
from geo4d_eval import depth_eval as _depth_eval  # noqa: E402

from evo.core.trajectory import PosePath3D, PoseTrajectory3D  # noqa: E402


# ── Pose ─────────────────────────────────────────────────────────────────────

def _c2w_to_traj(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a stack of c2w 4×4 matrices to TUM-format `(traj_tum, timestamps)`,
    matching `vo_eval.load_replica_traj` output convention:
      traj_tum: (N, 7) with columns (x, y, z, qw, qx, qy, qz)
      timestamps: (N,) frame indices as floats
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    N = len(c2w)
    poses_se3 = [c2w[i] for i in range(N)]
    pose_path = PosePath3D(poses_se3=poses_se3)
    timestamps = np.arange(N, dtype=np.float64)
    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps)
    xyz = traj.positions_xyz
    quat_wxyz = traj.orientations_quat_wxyz
    traj_tum = np.column_stack((xyz, quat_wxyz))
    return traj_tum, timestamps


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    cos_theta = (np.trace(rotation_matrix) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _eval_pose_sequence_first_pose_scale(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    seq: str,
    metric_file: Path,
    reason: Exception,
) -> dict:
    """Fallback for rank-degenerate trajectories where Umeyama cannot run."""
    if len(pred_c2w) < 2:
        raise reason

    align = gt_c2w[0] @ np.linalg.inv(pred_c2w[0])
    pred_aligned = np.einsum("ij,tjk->tik", align, pred_c2w)

    pred_delta = pred_aligned[:, :3, 3] - pred_aligned[0, :3, 3]
    gt_delta = gt_c2w[:, :3, 3] - gt_c2w[0, :3, 3]
    pred_span = float(np.linalg.norm(pred_delta[-1]))
    gt_span = float(np.linalg.norm(gt_delta[-1]))
    scale = gt_span / pred_span if pred_span > 1e-8 and gt_span > 1e-8 else 1.0
    pred_aligned[:, :3, 3] = gt_c2w[0, :3, 3] + scale * pred_delta

    translation_error = pred_aligned[:, :3, 3] - gt_c2w[:, :3, 3]
    ate = float(np.sqrt(np.mean(np.sum(translation_error**2, axis=1))))

    rpe_trans: list[float] = []
    rpe_rot: list[float] = []
    for i in range(len(pred_aligned) - 1):
        pred_rel = np.linalg.inv(pred_aligned[i]) @ pred_aligned[i + 1]
        gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[i + 1]
        rpe_trans.append(float(np.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3])))
        rpe_rot.append(_rotation_angle_deg(pred_rel[:3, :3] @ gt_rel[:3, :3].T))

    out = {
        "ate": float(ate),
        "rpe_trans": float(np.sqrt(np.mean(np.square(rpe_trans)))),
        "rpe_rot": float(np.sqrt(np.mean(np.square(rpe_rot)))),
        "n_frames": int(len(pred_c2w)),
        "alignment": "first_pose_scale_fallback",
    }
    metric_file.write_text(
        "Seq: {seq}\n\n"
        "Geo4D/evo Sim(3) alignment failed; used first-pose + baseline-scale fallback.\n"
        "Reason: {reason}\n"
        "ATE: {ate:.8f}\n"
        "RPE_trans: {rpe_trans:.8f}\n"
        "RPE_rot: {rpe_rot:.8f}\n".format(seq=seq, reason=reason, **out),
        encoding="utf-8",
    )
    return out


def eval_pose_sequence(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    seq: str,
    save_dir: str | os.PathLike,
) -> dict:
    """
    Compute Geo4D-style pose metrics for one sequence and write the
    `<seq>_eval_metric.txt` file expected by `vo_eval.process_directory`.

    Returns a dict {"ate", "rpe_trans", "rpe_rot", "n_frames"}.
    """
    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_c2w   = np.asarray(gt_c2w,   dtype=np.float64)
    assert pred_c2w.shape == gt_c2w.shape, (pred_c2w.shape, gt_c2w.shape)

    pred_traj = _c2w_to_traj(pred_c2w)
    gt_traj   = _c2w_to_traj(gt_c2w)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_file = save_dir / f"{seq}_eval_metric.txt"

    try:
        ate, rpe_trans, rpe_rot = _vo_eval.eval_metrics(
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            seq=seq,
            filename=str(metric_file),
        )
    except Exception as exc:
        if "Degenerate covariance rank" not in str(exc):
            raise
        return _eval_pose_sequence_first_pose_scale(pred_c2w, gt_c2w, seq, metric_file, exc)

    return {
        "ate":       float(ate),
        "rpe_trans": float(rpe_trans),
        "rpe_rot":   float(rpe_rot),
        "n_frames":  int(len(pred_c2w)),
        "alignment": "sim3_umeyama",
    }


# ── Depth ────────────────────────────────────────────────────────────────────

# Geo4D's per-dataset max_depth defaults (from infer_geo4d.py:538-540 + the
# Sintel/Bonn/KITTI/ScanNet++ presets in their evaluation_script.md).
DEPTH_MAX_BY_DATASET: dict[str, float | None] = {
    "vkitti2":        70.0,
    "scannetpp":      70.0,
    "aria":           70.0,
    "scenenet_depth": None,
    "dl3dv":          None,
    # Added: these were missing -> max_depth=None -> sky/far pixels never clipped ->
    # RMSE/SqRel blown up for PI3/DA3. Caps sit above each dataset's valid scene depth
    # (~p99) and below its sky sentinel (TartanAir ~134-140 m, Sintel ~300-374 m).
    "tartanair":      100.0,
    "tartanair_v2":   100.0,
    "sintel":         100.0,
}


def eval_depth_sequence(
    pred_depth_THW: np.ndarray | torch.Tensor,
    gt_depth_THW:   np.ndarray | torch.Tensor,
    dataset:        str | None = None,
    custom_mask:    np.ndarray | torch.Tensor | None = None,
    use_gpu:        bool = True,
    pred_in_disparity: bool = False,
    gt_in_disparity:   bool = False,
    eval_in_disparity: bool = False,
    normalize_unit_per_video: bool = False,
) -> dict:
    """
    Per-video global LAD2 scale-shift alignment, Geo4D-style.

    All inputs are (T, H, W). For consistent affine fit we want both pred and
    gt in *depth* space (positive, larger = farther) before LAD2. If a caller
    natively produces disparity (1/depth or normalised inverse depth, e.g.
    ChronoDepth's [0,1] output), pass `pred_in_disparity=True` and the input
    will be inverted (1/(x+eps)) before alignment. Same for `gt_in_disparity`.

    `depth_evaluation` then flattens across frames and fits one (s, t) over
    the whole video — Geo4D's per-video LAD2.
    """
    pd = pred_depth_THW
    gd = gt_depth_THW
    if isinstance(pd, np.ndarray): pd = torch.from_numpy(pd)
    if isinstance(gd, np.ndarray): gd = torch.from_numpy(gd)
    pd = pd.float()
    gd = gd.float()

    # Convert disparity-space inputs → depth space so LAD2 fits an affine
    # mapping in a single, consistent space across all methods.
    eps = 1e-6
    if pred_in_disparity:
        pd = 1.0 / torch.clamp(pd, min=eps)
    if gt_in_disparity:
        gd = 1.0 / torch.clamp(gd, min=eps)

    # Optional: AFTER both inputs are in depth space, flip both to disparity
    # so LAD2 fits and metrics are computed in *disparity space*. This is a
    # parallel evaluation track to the depth-space one — same per-video LAD2,
    # but the affine mapping s*p + t = g lives in 1/m units.
    if eval_in_disparity:
        pd = 1.0 / torch.clamp(pd, min=eps)
        gd = 1.0 / torch.clamp(gd, min=eps)

    # Optional: per-video min-max → [0, 1] BEFORE LAD2.
    # This stabilises depth-space alignment when raw depth has a huge dynamic
    # range (e.g. 1/disparity blowups near-zero). Applied to pred and gt
    # independently — LAD2 fits the affine offset/scale anyway, so this just
    # conditions the inputs without changing the affine-equivalent solution
    # (when valid pixels span the full input range; near-degenerate scenes
    # still benefit from bounded numerics).
    def _percentile_clip(x: torch.Tensor, q_lo: float = 2.0,
                         q_hi: float = 98.0) -> torch.Tensor:
        """Per-video percentile clip (no rescale). Bounds the input dynamic
        range so a handful of far-pixel outliers (where 1/disp blows up) don't
        dominate LAD2's affine fit; values in [p_lo, p_hi] pass through, the
        rest get clamped to those endpoints."""
        # numpy.percentile on a CPU array is faster + lighter than torch.quantile
        # on a multi-million-element tensor (the latter materialises a sorted copy).
        arr = x.detach().cpu().numpy() if x.is_cuda else x.detach().numpy()
        finite = np.isfinite(arr) & (arr > 0)
        if not finite.any():
            return x
        vals = arr[finite]
        lo = float(np.percentile(vals, q_lo))
        hi = float(np.percentile(vals, q_hi))
        if hi <= lo:
            return x
        return torch.clamp(x, lo, hi)

    if normalize_unit_per_video:
        # Per-video minmax to a bounded **non-zero** window using GT's [p2, p98]
        # as the shared reference range. Both pred and gt land in the same
        # [floor, 1] depth interval so LAD2 fits cleanly; the floor (0.05)
        # avoids zero-valued GT that would either get masked out or drive
        # abs_rel = |p-g|/g toward infinity.
        FLOOR = 0.05
        gd_arr = gd.detach().cpu().numpy()
        finite = np.isfinite(gd_arr) & (gd_arr > 0)
        if finite.any():
            v = gd_arr[finite]
            lo = float(np.percentile(v, 2.0))
            hi = float(np.percentile(v, 98.0))
            if hi > lo:
                def _norm(t: torch.Tensor) -> torch.Tensor:
                    out = (t - lo) / (hi - lo)        # → ~[0, 1] over GT bulk
                    out = torch.clamp(out, 0.0, 1.0)
                    return out * (1.0 - FLOOR) + FLOOR  # → [FLOOR, 1]
                pd = _norm(pd)
                gd = _norm(gd)

    # When both inputs are unit-normalised, the dataset's metric max_depth
    # clip (e.g. 70 m for vKITTI2) no longer applies — values are in [0, 1].
    if normalize_unit_per_video:
        max_depth = None
    else:
        max_depth = DEPTH_MAX_BY_DATASET.get(dataset, None)
    # Geo4D uses lr=1e-2/iters=5000 when masking with `custom_mask` (their
    # outdoor branch); lr=1e-4/iters=1000 otherwise. Mirror that.
    lr        = 1e-2 if custom_mask is not None else 1e-4
    max_iters = 5000 if custom_mask is not None else 1000
    post_clip_max = max_depth

    results, _err_map, _pred_full, _gt_full = _depth_eval.depth_evaluation(
        predicted_depth_original=pd,
        ground_truth_depth_original=gd,
        max_depth=max_depth,
        custom_mask=custom_mask,
        align_with_lad2=True,
        lr=lr,
        max_iters=max_iters,
        use_gpu=use_gpu,
        post_clip_max=post_clip_max,
    )
    # Stringify the seven Geo4D metric keys to JSON-friendly snake_case.
    return {
        "abs_rel":   float(results["Abs Rel"]),
        "sq_rel":    float(results["Sq Rel"]),
        "rmse":      float(results["RMSE"]),
        "log_rmse":  float(results["Log RMSE"]),
        "delta_1":   float(results["δ < 1.25"]),
        "delta_2":   float(results["δ < 1.25^2"]),
        "delta_3":   float(results["δ < 1.25^3"]),
        "valid_pixels": int(results["valid_pixels"]),
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_seq: list[dict], keys: list[str]) -> dict:
    """Arithmetic mean over sequences (matches Geo4D `calculate_averages`)."""
    if not per_seq:
        return {k: 0.0 for k in keys} | {"n_samples": 0}
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in per_seq if k in d]
        out[k] = sum(vals) / max(len(vals), 1)
    out["n_samples"] = len(per_seq)
    return out


POSE_KEYS  = ["ate", "rpe_trans", "rpe_rot"]
DEPTH_KEYS = ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]


# ── Depth-format conversion ──────────────────────────────────────────────────

def disparity_norm_to_metric_depth(
    depth_norm: np.ndarray,
    scale: float,
    eps: float = 1e-3,
    percentile_clip: float | None = 2.0,
    disp_minmax_floor: float = 0.05,
    disp_minmax_ceil: float = 0.95,
) -> np.ndarray:
    """
    Convert the ``[-1, 1]`` normalised disparity stored by our inference
    pipeline (`*_depth_raw.npz["depth"]` / `["scale"]`) into metric depth
    in meters.

    Pipeline (matches scripts/eval_rgb_to_ray_depth_depth.py:53-71):
        disp_01  = (depth_norm + 1) / 2
        d_scaled = 1 / clip(disp_01, lo, 1 - eps) - 1
        d_metric = d_scaled * scale

    `percentile_clip` (default 2.0): clip disp_01's lower bound to its p-th
    percentile across valid pixels of the video (with floor `eps`). This
    stops the `1/disp` inversion from exploding when many pixels have
    near-zero disparity (very far ground-truth points), which otherwise
    blows up the depth dynamic range and makes downstream LAD2 alignment
    numerically dominated by a few outlier far pixels. Set to None for the
    legacy fixed-eps behaviour.
    """
    disp_01 = (np.asarray(depth_norm, dtype=np.float32) + 1.0) * 0.5
    # Percentile-floor on disp_01 keeps the subsequent 1/disp inversion bounded
    # regardless of the raw disparity distribution (degenerate scenes where most
    # pixels cluster near 0 no longer blow up to 1/eps depths). Relative order
    # is preserved, so LAD2 downstream still has the right shape to fit.
    if percentile_clip is not None:
        valid = (disp_01 > 0) & np.isfinite(disp_01)
        if valid.any():
            lo_p = float(np.percentile(disp_01[valid], percentile_clip))
            lo = max(lo_p, eps)
        else:
            lo = eps
    else:
        lo = eps
    disp_01 = np.clip(disp_01, lo, 1.0 - eps)
    d_scaled = 1.0 / disp_01 - 1.0
    return (d_scaled * float(scale)).astype(np.float32)


def load_depth_metric_from_npz(npz_path) -> np.ndarray:
    """
    Read `*_depth_raw.npz` (with keys 'depth' [-1,1] and 'scale' float) and
    return a (T, H, W) float32 metric-depth tensor.
    """
    npz = np.load(npz_path)
    return disparity_norm_to_metric_depth(
        depth_norm=npz["depth"].astype(np.float32),
        scale=float(npz["scale"]),
    )

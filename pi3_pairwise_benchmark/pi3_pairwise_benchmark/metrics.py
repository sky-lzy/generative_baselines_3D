from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairwiseErrors:
    rotation_deg: np.ndarray
    translation_deg: np.ndarray
    pairs: list[tuple[int, int]]


def pairwise_indices(num_frames: int) -> list[tuple[int, int]]:
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    return [(i, j) for i in range(num_frames) for j in range(i + 1, num_frames)]


def invert_se3_stack(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose stack with shape (N,4,4), got {poses.shape}")
    return np.linalg.inv(poses)


def _rotation_angle_deg(rot_gt: np.ndarray, rot_pred: np.ndarray) -> float:
    rel = rot_gt @ rot_pred.T
    cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _translation_angle_deg(t_gt: np.ndarray, t_pred: np.ndarray, ambiguity: bool = True) -> float:
    eps = 1e-15
    n_gt = float(np.linalg.norm(t_gt))
    n_pred = float(np.linalg.norm(t_pred))
    if n_gt < eps or n_pred < eps:
        return 0.0
    t_gt_unit = t_gt / (n_gt + eps)
    t_pred_unit = t_pred / (n_pred + eps)
    loss_t = max(1.0 - float(np.dot(t_pred_unit, t_gt_unit)) ** 2, eps)
    err = float(np.degrees(np.arccos(np.sqrt(1.0 - loss_t))))
    if ambiguity:
        err = min(err, abs(180.0 - err))
    return err


def compute_pairwise_errors(
    pred_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    pairs: Iterable[tuple[int, int]] | None = None,
) -> PairwiseErrors:
    pred_w2c = np.asarray(pred_w2c, dtype=np.float64)
    gt_w2c = np.asarray(gt_w2c, dtype=np.float64)
    if pred_w2c.shape != gt_w2c.shape:
        raise ValueError(f"Pose shape mismatch: pred={pred_w2c.shape}, gt={gt_w2c.shape}")
    if pred_w2c.ndim != 3 or pred_w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose stacks with shape (N,4,4), got {pred_w2c.shape}")

    pair_list = list(pairwise_indices(len(pred_w2c)) if pairs is None else pairs)
    r_errors: list[float] = []
    t_errors: list[float] = []
    for i, j in pair_list:
        rel_gt = gt_w2c[j] @ np.linalg.inv(gt_w2c[i])
        rel_pred = pred_w2c[j] @ np.linalg.inv(pred_w2c[i])
        r_errors.append(_rotation_angle_deg(rel_gt[:3, :3], rel_pred[:3, :3]))
        t_errors.append(_translation_angle_deg(rel_gt[:3, 3], rel_pred[:3, 3]))

    return PairwiseErrors(
        rotation_deg=np.asarray(r_errors, dtype=np.float64),
        translation_deg=np.asarray(t_errors, dtype=np.float64),
        pairs=pair_list,
    )


def calculate_auc_np(
    r_error: np.ndarray,
    t_error: np.ndarray,
    max_threshold: int = 30,
) -> tuple[float, np.ndarray]:
    r_error = np.asarray(r_error, dtype=np.float64)
    t_error = np.asarray(t_error, dtype=np.float64)
    if r_error.shape != t_error.shape:
        raise ValueError(f"Error shape mismatch: r={r_error.shape}, t={t_error.shape}")
    if r_error.size == 0:
        return float("nan"), np.zeros(max_threshold, dtype=np.float64)

    max_errors = np.max(np.stack((r_error, t_error), axis=1), axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized))), normalized


def summarize_pi3_metrics(
    rotation_deg: np.ndarray,
    translation_deg: np.ndarray,
    thresholds: Iterable[int] = (5, 15, 30),
) -> dict[str, float]:
    rotation_deg = np.asarray(rotation_deg, dtype=np.float64)
    translation_deg = np.asarray(translation_deg, dtype=np.float64)
    out: dict[str, float] = {}
    for threshold in thresholds:
        out[f"Racc_{threshold}"] = float(np.mean(rotation_deg < threshold) * 100.0)
        out[f"Tacc_{threshold}"] = float(np.mean(translation_deg < threshold) * 100.0)
        auc, _ = calculate_auc_np(rotation_deg, translation_deg, max_threshold=int(threshold))
        out[f"Auc_{threshold}"] = float(auc * 100.0)
    out["num_pairs"] = float(rotation_deg.size)
    return out


def evaluate_w2c_stack(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict[str, float]:
    errors = compute_pairwise_errors(pred_w2c=pred_w2c, gt_w2c=gt_w2c)
    return summarize_pi3_metrics(errors.rotation_deg, errors.translation_deg)


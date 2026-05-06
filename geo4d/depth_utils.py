"""Depth shape helpers for Geo4D evaluators."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def match_depth_to_gt(pred_depth: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    """Trim padded frames and resize prediction depth to match GT shape."""
    if pred_depth.shape == gt_depth.shape:
        return pred_depth

    n_frames = min(pred_depth.shape[0], gt_depth.shape[0])
    pred = pred_depth[:n_frames]
    gt = gt_depth[:n_frames]
    if pred.shape[-2:] != gt.shape[-2:]:
        tensor = torch.from_numpy(pred).unsqueeze(1).float()
        tensor = F.interpolate(tensor, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = tensor.squeeze(1).numpy()
    return pred

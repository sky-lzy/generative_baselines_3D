"""Camera helpers for the local SEVA evaluation wrapper."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


IntrinsicsScale = str | float | tuple[float, float]
FrameWindow = str | int


def _parse_intrinsics_scale(scale: IntrinsicsScale) -> tuple[float, float] | Literal["auto"]:
    if isinstance(scale, str):
        value = scale.strip().lower()
        if value == "auto":
            return "auto"
        if value in {"none", "off"}:
            return (1.0, 1.0)
        if "," in value:
            sx, sy = value.split(",", maxsplit=1)
            return (float(sx), float(sy))
        parsed = float(value)
        return (parsed, parsed)
    if isinstance(scale, tuple):
        sx, sy = scale
        return (float(sx), float(sy))
    parsed = float(scale)
    return (parsed, parsed)


def _intrinsics_are_normalized(Ks: np.ndarray) -> bool:
    principal = Ks[..., :2, 2]
    return bool(np.all((principal >= 0.0) & (principal <= 1.0)))


def _infer_intrinsics_scale(Ks: np.ndarray, image_hw: tuple[int, int]) -> tuple[float, float]:
    """Infer a uniform pixel-scale fix for intrinsics saved at ray-grid size."""
    if _intrinsics_are_normalized(Ks):
        return (1.0, 1.0)

    image_h, image_w = (float(image_hw[0]), float(image_hw[1]))
    if image_h <= 0 or image_w <= 0:
        return (1.0, 1.0)

    cx = float(np.nanmedian(Ks[..., 0, 2]))
    cy = float(np.nanmedian(Ks[..., 1, 2]))
    if not np.isfinite(cx) or not np.isfinite(cy) or cx <= 0.0 or cy <= 0.0:
        return (1.0, 1.0)

    # Pixel-space intrinsics usually have principal points near the RGB image
    # center. The broken metadata path stores K at the ray-map resolution, so
    # cx/W and cy/H are around 1/16 for an 8x smaller ray grid.
    if 0.20 <= cx / image_w <= 0.80 and 0.20 <= cy / image_h <= 0.80:
        return (1.0, 1.0)

    candidates = np.array([image_w / (2.0 * cx), image_h / (2.0 * cy)], dtype=np.float64)
    if not np.all(np.isfinite(candidates)):
        return (1.0, 1.0)

    uniform = float(round(float(np.median(candidates))))
    if uniform <= 1.0 or uniform > 64.0:
        return (1.0, 1.0)
    if np.max(np.abs(candidates - uniform) / uniform) > 0.25:
        return (1.0, 1.0)
    return (uniform, uniform)


def rescale_intrinsics_to_image(
    Ks: np.ndarray,
    image_hw: tuple[int, int],
    intrinsics_scale: IntrinsicsScale = "auto",
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return K expressed in RGB pixel coordinates.

    SEVA accepts pixel-space K and then normalizes it internally. Some local
    evaluation artifacts store K at the ray-map grid resolution while `gt_rgb`
    is saved at RGB resolution. `intrinsics_scale="auto"` detects that common
    case and applies the uniform integer scale, typically 8x.
    """
    parsed_scale = _parse_intrinsics_scale(intrinsics_scale)
    out = np.array(Ks, dtype=np.float64, copy=True)
    squeeze = out.ndim == 2
    if squeeze:
        out = out[None]
    if out.ndim != 3 or out.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T,3,3) or (3,3), got {Ks.shape}")

    scale_xy = _infer_intrinsics_scale(out, image_hw) if parsed_scale == "auto" else parsed_scale
    sx, sy = scale_xy
    out[..., 0, :] *= sx
    out[..., 1, :] *= sy
    if squeeze:
        out = out[0]
    return out, (float(sx), float(sy))


def resolve_seva_frame_window(num_frames: int, frame_window: FrameWindow = 21) -> int:
    """Return the SEVA per-forward frame window.

    Official SEVA uses `T` as the model context window, defaulting to 21, not as
    the total clip length. Use "full"/"all" or a non-positive integer to recover
    the old one-forward behavior.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    if isinstance(frame_window, str):
        value = frame_window.strip().lower()
        if value in {"full", "all", "clip"}:
            return int(num_frames)
        window = int(value)
    else:
        window = int(frame_window)

    if window <= 0:
        return int(num_frames)
    if window < 3:
        raise ValueError(f"frame_window must be at least 3, got {window}")
    return min(window, int(num_frames))


def resolve_seva_target_size(
    image_hw: tuple[int, int],
    resize_hw: tuple[int, int] | None = None,
    l_short: int | None = 576,
    size_stride: int = 64,
) -> tuple[int, int]:
    """Return target (H, W) using SEVA's aspect-preserving short-side rule."""
    image_h, image_w = (int(image_hw[0]), int(image_hw[1]))
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image_hw must be positive, got {image_hw}")

    if resize_hw is not None:
        target_h, target_w = (int(resize_hw[0]), int(resize_hw[1]))
    elif l_short is None or int(l_short) <= 0:
        target_h, target_w = image_h, image_w
    else:
        short = int(l_short)
        if image_w < image_h:
            target_w = short
            target_h = int(short * image_h / image_w)
        else:
            target_h = short
            target_w = int(short * image_w / image_h)

    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target size must be positive, got {(target_h, target_w)}")
    if size_stride > 1:
        target_h = math.floor(target_h / size_stride + 0.5) * size_stride
        target_w = math.floor(target_w / size_stride + 0.5) * size_stride
    return int(target_h), int(target_w)

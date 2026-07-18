"""Depth-alignment ablation, applied IDENTICALLY to every method for fairness.

Three per-sequence alignments of pred depth to GT depth (pred is metric-proportional):
  1. scale     : aligned = s·pred            (s by L1-robust IRLS; matches π³ scale-only)
  2. affine_lsq: aligned = s·pred + t         (closed-form least-squares, L2)
  3. lads      : aligned = s·pred + t         (L1 robust affine == old-pipeline "LAD2",
                                               solved by IRLS -> the true L1 optimum, so it
                                               does NOT depend on Adam lr/iters converging)

metrics on the shared valid pixels: AbsRel = mean(|a-g|/g), delta1 = mean(max(a/g,g/a)<1.25).
All alignment is fit ONLY on valid pixels; masking (max_depth etc.) is applied before fitting.
"""
import numpy as np


def _valid_mask(pred, gt, max_depth=None, min_depth=1e-3):
    m = np.isfinite(pred) & np.isfinite(gt) & (gt > min_depth) & (pred > 0)
    if max_depth is not None:
        m &= (gt <= max_depth)
    return m


def _fit_scale(p, g, iters=15):
    """L1-robust scale via IRLS (Weiszfeld), init at ratio of medians."""
    s = np.median(g) / max(np.median(p), 1e-8)
    for _ in range(iters):
        r = np.abs(s * p - g)
        w = 1.0 / np.maximum(r, 1e-6)
        s = np.sum(w * p * g) / max(np.sum(w * p * p), 1e-12)
    return float(s), 0.0


def _fit_affine_lsq(p, g):
    """Closed-form least-squares scale+shift."""
    A = np.stack([p, np.ones_like(p)], axis=1)
    st, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(st[0]), float(st[1])


def _fit_affine_lads(p, g, iters=50):
    """L1-robust scale+shift via IRLS (converges to the true LAD/LAD2 optimum)."""
    s, t = _fit_affine_lsq(p, g)
    for _ in range(iters):
        r = np.abs(s * p + t - g)
        w = 1.0 / np.maximum(r, 1e-6)
        W = np.sqrt(w)
        A = np.stack([p * W, W], axis=1)
        b = g * W
        st, *_ = np.linalg.lstsq(A, b, rcond=None)
        ns, nt = float(st[0]), float(st[1])
        if abs(ns - s) < 1e-7 and abs(nt - t) < 1e-7:
            s, t = ns, nt; break
        s, t = ns, nt
    return s, t


ALIGNERS = {"scale": _fit_scale, "affine_lsq": _fit_affine_lsq, "lads": _fit_affine_lads}


def _metrics(aligned, g):
    aligned = np.clip(aligned, 1e-6, None)
    absrel = float(np.mean(np.abs(aligned - g) / g))
    ratio = np.maximum(aligned / g, g / aligned)
    d1 = float(np.mean((ratio < 1.25).astype(np.float64)))
    return absrel, d1


def score_all_alignments(pred, gt, max_depth=None, min_depth=1e-3):
    """pred,gt: (N,H,W) or flat. Returns {alignment: (absrel, delta1, n_valid_pixels)}.
    A single scale/affine is fit over ALL valid pixels of the sequence (per-video), matching
    both π³ (scale) and the old pipeline (per-video affine)."""
    pred = np.asarray(pred, np.float64).reshape(-1)
    gt = np.asarray(gt, np.float64).reshape(-1)
    m = _valid_mask(pred, gt, max_depth, min_depth)
    if m.sum() < 10:
        return {k: (float("nan"), float("nan"), int(m.sum())) for k in ALIGNERS}
    p, g = pred[m], gt[m]
    out = {}
    for name, fit in ALIGNERS.items():
        s, t = fit(p, g)
        absrel, d1 = _metrics(s * p + t, g)
        out[name] = (absrel, d1, int(m.sum()))
    return out

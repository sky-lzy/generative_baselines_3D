#!/usr/bin/env python3
"""ONE scorer for every method in the fair NVS benchmark: PSNR / SSIM / LPIPS.

Both methods go through a single code path here. That is the point: as soon as each side has its own
scoring function, differences in resampling, colour range or compression start masquerading as model
quality. The only per-method branch is *how the predicted frames are read off disk* (below), and both
readers are lossless.

LOSSLESS ON BOTH SIDES
----------------------
  ours : sample_XXXXX/raw_arrays.npz -> rgb_pred (T,3,H,W) float in [-1,1], written by the engine's
         --save_raw BEFORE any video encoding. NOT pred_rgb.mp4, which is H.264 and would put a
         lossy codec on our side only.
  SEVA : <scene>/samples-rgb/*.png -- already lossless.
No codec asymmetry, in either direction.

COMMON METRIC DOMAIN
--------------------
Strategy B puts the square crop on the OUTPUT side (SEVA's own benchmark README does this for the
ViewCrafter splits). So for GT and for every method:

    4:3 frame -> CENTRE SQUARE CROP -> resize to metric_size (area) -> metrics

Ours is 384x288 -> 288x288; SEVA is 768x576 -> 576x576; both then resize to metric_size. GT comes
from the same 384x288 scene image the models were given, through the same crop+resize. Nothing is
upsampled into the metric domain, so no method is credited with detail it did not produce.

Metric conventions match the campaign scorers so numbers stay comparable:
  PSNR  mean over per-frame PSNR (not PSNR of the mean MSE), on [0,1]
  SSIM  scikit-image, channel_axis=2, data_range=1.0, gaussian_weights, sigma=1.5
  LPIPS lpips(alex) on [-1,1]

SELF-TEST (--selftest)
----------------------
Reads ours' raw_arrays.npz and recomputes FULL-FRAME PSNR, then compares against the engine's own
rgb_metrics.json per_frame_psnr for the same frames. They must agree to <1e-4 dB. That proves the
raw-array path is being decoded correctly (right array, right layout, right value range) before any
cropping is introduced -- if this drifts, every number the scorer emits is suspect.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ----------------------------------------------------------------- frame readers (only branch)
def read_ours(sample_dir, ids):
    """(N,3,H,W) float [0,1] from the lossless raw arrays, indexed by CLIP frame id."""
    f = os.path.join(sample_dir, "raw_arrays.npz")
    if not os.path.exists(f):
        return None, "no raw_arrays.npz (run inference with --save_raw)"
    with np.load(f) as z:
        if "rgb_pred" not in z:
            return None, "raw_arrays.npz has no rgb_pred"
        a = z["rgb_pred"]                      # (T,3,H,W) in [-1,1]
    T = a.shape[0]
    bad = [i for i in ids if i >= T]
    if bad:
        return None, f"clip has {T} frames; ids {bad} out of range"
    x = torch.from_numpy(a[list(ids)]).float()
    return (x * 0.5 + 0.5).clamp(0, 1), None


def read_seva(scene_dir, ids):
    """(N,3,H,W) float [0,1] from PNGs. SEVA emits its targets in test_id order, one png each."""
    ps = sorted(glob.glob(os.path.join(scene_dir, "samples-rgb", "*.png")))
    if not ps:
        return None, "no samples-rgb/*.png"
    if len(ps) < len(ids):
        return None, f"{len(ps)} pngs < {len(ids)} targets"
    out = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).float().permute(2, 0, 1) / 255.0
           for p in ps[:len(ids)]]
    return torch.stack(out), None


def read_gt(scene_root, ids):
    frames = json.load(open(os.path.join(scene_root, "transforms.json")))["frames"]
    out = []
    for i in ids:
        p = os.path.join(scene_root, frames[i]["file_path"])
        out.append(torch.from_numpy(np.array(Image.open(p).convert("RGB"))).float().permute(2, 0, 1) / 255.0)
    return torch.stack(out)


# ----------------------------------------------------------------- common domain + metrics
def to_metric_domain(x, size):
    """(N,3,H,W) [0,1] -> centre square crop -> resize `size` (area). Identical for every method."""
    h, w = x.shape[-2:]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    x = x[..., y0:y0 + s, x0:x0 + s]
    if s != size:
        x = F.interpolate(x, size=(size, size), mode="area")
    return x.clamp(0, 1)


def metrics(pred, gt, lp, dev):
    from skimage.metrics import structural_similarity as ssim_fn
    mses = torch.mean((pred - gt) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr = float(torch.mean(-10 * torch.log10(mses)))          # mean of per-frame PSNR
    ssim = float(np.mean([
        ssim_fn(g.permute(1, 2, 0).numpy(), p.permute(1, 2, 0).numpy(), channel_axis=2,
                data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        for p, g in zip(pred, gt)]))
    with torch.no_grad():
        d = lp(pred.to(dev) * 2 - 1, gt.to(dev) * 2 - 1).flatten().cpu()
    return psnr, ssim, float(d.mean())


# ----------------------------------------------------------------- self-test
def selftest(run_dir, scenes_root, scenes, tol=1e-4):
    """Full-frame PSNR from raw_arrays must reproduce the engine's rgb_metrics.json."""
    worst, n = 0.0, 0
    for i, sc in enumerate(scenes):
        sd = os.path.join(run_dir, f"sample_{i:05d}")
        m = os.path.join(sd, "rgb_metrics.json")
        f = os.path.join(sd, "raw_arrays.npz")
        if not (os.path.exists(m) and os.path.exists(f)):
            continue
        pf = json.load(open(m)).get("per_frame_psnr")
        if not pf:
            continue
        with np.load(f) as z:
            p, g = z["rgb_pred"], z["rgb_gt"]
        pu = (torch.from_numpy(p).float() * 0.5 + 0.5).clamp(0, 1)
        gu = (torch.from_numpy(g).float() * 0.5 + 0.5).clamp(0, 1)
        mse = torch.mean((pu - gu) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        mine = (-10 * torch.log10(mse)).numpy()
        k = min(len(mine), len(pf))
        d = float(np.abs(mine[:k] - np.array(pf[:k])).max())
        worst = max(worst, d); n += 1
    if n == 0:
        print("[selftest] no raw_arrays.npz + rgb_metrics.json pairs found -- SKIPPED")
        return True
    print(f"[selftest] {n} scenes | max |raw-array PSNR - engine PSNR| = {worst:.2e} dB "
          f"({'PASS' if worst < tol else 'FAIL'})")
    return worst < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="preds cell dir (ours: sample_*/; seva: <scene>/)")
    ap.add_argument("--method", required=True, choices=["ours", "seva"])
    ap.add_argument("--scenes_root", required=True, help="fair scene-set dir")
    ap.add_argument("--split", type=int, required=True, choices=[1, 2])
    ap.add_argument("--metric_size", type=int, default=256)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--selftest", action="store_true", help="ours only; verify the raw-array path")
    ap.add_argument("--require_complete", action="store_true",
                    help="exit non-zero unless every scene scored (no silent partial CSVs)")
    a = ap.parse_args()

    scenes = sorted(d for d in os.listdir(a.scenes_root)
                    if os.path.isdir(os.path.join(a.scenes_root, d)))
    if a.selftest and a.method == "ours":
        if not selftest(a.run_dir, a.scenes_root, scenes):
            sys.exit("SELFTEST FAILED — raw-array decoding does not match the engine")

    import lpips as lpips_mod
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips_mod.LPIPS(net="alex").to(dev).eval()

    rows, skipped = [], []
    for i, sc in enumerate(scenes):
        sroot = os.path.join(a.scenes_root, sc)
        ids = sorted(json.load(open(f"{sroot}/train_test_split_{a.split}.json"))["test_ids"])
        if a.method == "ours":
            pred, err = read_ours(os.path.join(a.run_dir, f"sample_{i:05d}"), ids)
        else:
            pred, err = read_seva(os.path.join(a.run_dir, sc), ids)
        if pred is None:
            skipped.append((sc, err)); continue
        gt = read_gt(sroot, ids[:len(pred)])
        p = to_metric_domain(pred, a.metric_size)
        g = to_metric_domain(gt, a.metric_size)
        psnr, ssim, lpi = metrics(p, g, lp, dev)
        rows.append((sc, psnr, ssim, lpi, len(p)))
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(scenes)} {sc} psnr {psnr:.2f} ssim {ssim:.4f} lpips {lpi:.4f}",
                  flush=True)

    if skipped:
        print(f"[warn] {len(skipped)} scene(s) skipped; first few: "
              + "; ".join(f"{s}: {e}" for s, e in skipped[:5]))
    if not rows:
        sys.exit("ERROR: nothing scored")

    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "psnr", "ssim", "lpips", "n_views"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", r[4]])
        w.writerow(["AVERAGE", f"{np.mean([r[1] for r in rows]):.4f}",
                    f"{np.mean([r[2] for r in rows]):.4f}",
                    f"{np.mean([r[3] for r in rows]):.4f}", sum(r[4] for r in rows)])
    print(f"[{a.method}] {len(rows)}/{len(scenes)} scenes | "
          f"PSNR {np.mean([r[1] for r in rows]):.4f} | SSIM {np.mean([r[2] for r in rows]):.4f} | "
          f"LPIPS {np.mean([r[3] for r in rows]):.4f} -> {a.out_csv}")
    if a.require_complete and len(rows) != len(scenes):
        sys.exit(f"INCOMPLETE: {len(rows)}/{len(scenes)} scenes scored")


if __name__ == "__main__":
    main()

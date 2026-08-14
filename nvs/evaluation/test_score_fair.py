#!/usr/bin/env python3
"""Correctness tests for score_nvs_fair.py. No GPU, no model — synthetic predictions only.

The scorer is the one place where an unfair comparison can hide, because it is the only code both
methods share. These tests pin down the three ways it could silently favour one side:

  T1 PERFECT RECONSTRUCTION  feed GT back as the prediction -> PSNR must be enormous, SSIM ~1,
     LPIPS ~0. Catches value-range bugs (a [-1,1] vs [0,1] mixup would show up as ~10 dB here).

  T2 PATH PARITY (the important one)  give the OURS reader and the SEVA reader byte-identical image
     content at the same resolution. Both go through crop -> resize -> metrics. The two CSVs must
     agree to ~1e-6. If they do not, the reader path itself is biasing results and every downstream
     number is contaminated.

  T3 RESOLUTION EFFECT  give the SEVA reader the same content upsampled 2x to 768x576 -- exactly
     what SEVA really emits. Any metric difference vs T2 is the pure resampling advantage of
     rendering larger and being downsampled into the metric domain, with model quality held
     constant. This quantifies the residual unfairness the design cannot remove.

Run:  python3 nvs/evaluation/test_score_fair.py [--scenes_root <fair set>] [--n 3]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SCORER = HERE / "score_nvs_fair.py"


def read_csv(p):
    rows = {r["scene"]: r for r in csv.DictReader(open(p))}
    return rows.pop("AVERAGE", None), rows


def build_fake(tmp, scenes_root, scenes, split, seva_scale=1):
    """Write an 'ours' preds dir and a 'seva' preds dir that both contain the GT frames."""
    ours = tmp / f"ours_x{seva_scale}"
    seva = tmp / f"seva_x{seva_scale}"
    for i, sc in enumerate(scenes):
        sroot = Path(scenes_root) / sc
        meta = json.load(open(sroot / "transforms.json"))["frames"]
        ids = sorted(json.load(open(sroot / f"train_test_split_{split}.json"))["test_ids"])
        imgs = [np.array(Image.open(sroot / meta[k]["file_path"]).convert("RGB")) for k in ids]

        # ---- ours: raw_arrays.npz holds the FULL clip in [-1,1], indexed by clip frame id
        T = max(ids) + 1
        h, w = imgs[0].shape[:2]
        full = np.zeros((T, 3, h, w), dtype=np.float32)
        for k, im in zip(ids, imgs):
            full[k] = (im.astype(np.float32) / 255.0 * 2 - 1).transpose(2, 0, 1)
        d = ours / f"sample_{i:05d}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "raw_arrays.npz", rgb_pred=full, rgb_gt=full)

        # ---- seva: one png per target, in test_id order (its real output layout)
        sd = seva / sc / "samples-rgb"
        sd.mkdir(parents=True, exist_ok=True)
        for j, im in enumerate(imgs):
            pim = Image.fromarray(im)
            if seva_scale != 1:
                pim = pim.resize((w * seva_scale, h * seva_scale), Image.LANCZOS)
            pim.save(sd / f"{j:03d}.png")
    return ours, seva


def score(run_dir, method, scenes_root, split, out_csv, metric_size=256):
    cmd = [sys.executable, str(SCORER), "--run_dir", str(run_dir), "--method", method,
           "--scenes_root", str(scenes_root), "--split", str(split),
           "--metric_size", str(metric_size), "--out_csv", str(out_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); sys.exit(f"scorer failed for {method}")
    return read_csv(out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root",
                    default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair/re10k128_50f")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    all_scenes = sorted(d for d in os.listdir(a.scenes_root)
                        if os.path.isdir(os.path.join(a.scenes_root, d)))
    scenes = all_scenes[:a.n]
    # the scorer walks ALL scenes in scenes_root, so give it a root containing only our subset
    tmp = Path(tempfile.mkdtemp(prefix="fairtest_"))
    sub = tmp / "scenes"
    sub.mkdir()
    for sc in scenes:
        os.symlink(Path(a.scenes_root) / sc, sub / sc)

    print(f"scenes: {scenes}\nsplit: {a.split}\n")
    fails = []

    # ---------------- T1 + T2 : same content, same resolution, both readers
    ours, seva = build_fake(tmp, a.scenes_root, scenes, a.split, seva_scale=1)
    ao, ro = score(ours, "ours", sub, a.split, tmp / "ours.csv")
    as_, rs = score(seva, "seva", sub, a.split, tmp / "seva.csv")

    print("T1 perfect reconstruction (prediction == GT)")
    print(f"    ours  PSNR {float(ao['psnr']):9.3f}  SSIM {float(ao['ssim']):.6f}  LPIPS {float(ao['lpips']):.6f}")
    print(f"    seva  PSNR {float(as_['psnr']):9.3f}  SSIM {float(as_['ssim']):.6f}  LPIPS {float(as_['lpips']):.6f}")
    if not (float(ao["psnr"]) > 60 and float(ao["ssim"]) > 0.999 and float(ao["lpips"]) < 1e-3):
        fails.append("T1 ours: perfect reconstruction did not score as perfect")
    if not (float(as_["psnr"]) > 60 and float(as_["ssim"]) > 0.999 and float(as_["lpips"]) < 1e-3):
        fails.append("T1 seva: perfect reconstruction did not score as perfect")

    print("\nT2 reader-path parity (identical content, identical resolution)")
    worst = 0.0
    for sc in scenes:
        for m in ("psnr", "ssim", "lpips"):
            d = abs(float(ro[sc][m]) - float(rs[sc][m]))
            worst = max(worst, d)
    print(f"    max |ours - seva| over all scenes/metrics = {worst:.3e}")
    if worst > 1e-5:
        fails.append(f"T2: reader paths disagree by {worst:.3e} on identical content")

    # ---------------- T3 : SEVA at its real 2x output resolution
    _, seva2 = build_fake(tmp, a.scenes_root, scenes, a.split, seva_scale=2)
    as2, rs2 = score(seva2, "seva", sub, a.split, tmp / "seva2x.csv")
    print("\nT3 resolution effect (SEVA content upsampled 2x -> 768x576, its real output size)")
    print(f"    seva 1x  PSNR {float(as_['psnr']):8.3f}  SSIM {float(as_['ssim']):.6f}  LPIPS {float(as_['lpips']):.6f}")
    print(f"    seva 2x  PSNR {float(as2['psnr']):8.3f}  SSIM {float(as2['ssim']):.6f}  LPIPS {float(as2['lpips']):.6f}")
    print(f"    delta    PSNR {float(as2['psnr']) - float(as_['psnr']):+8.3f}  "
          f"SSIM {float(as2['ssim']) - float(as_['ssim']):+.6f}  "
          f"LPIPS {float(as2['lpips']) - float(as_['lpips']):+.6f}")
    print("    (pure resampling, model quality held constant — the residual the design cannot remove)")

    print("\n" + ("ALL TESTS PASSED" if not fails else "FAILURES:\n  " + "\n  ".join(fails)))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"kept: {tmp}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate a Gen3R / Gen3C prediction directory against score_nvs_fair.py's contract.

Why this exists: when a baseline scores badly it is usually impossible to tell, from the
number alone, whether the MODEL is bad or the PREDICTIONS are malformed (wrong frame count,
wrong dtype, wrong range, off-by-one indexing, fewer scenes than the benchmark). The scorer
deliberately refuses to guess -- it returns an error string per scene and averages what is
left, so a systematically broken cell can still produce a plausible-looking average.

This script re-implements the scorer's read path EXACTLY (same file names, same shape and
index checks) and reports, per scene, whether it would be scored or skipped -- without
needing a GPU, the model, or the ground truth.

    python nvs/baselines/verify_baseline_preds.py --method gen3c --run_dir <preds dir> \
        --split 2 --n_frames 50

Exit code 0 = every scene satisfies the contract; 1 = at least one would be dropped.
"""
import argparse, glob, json, os, sys

import numpy as np

# Frame ids the scorer asks for, mirroring score_nvs_fair.py.
#   split 1 -> predict 1..n_frames-1 ; split 2 -> predict 1..n_frames-2
def scored_ids(n_frames, split):
    return list(range(1, n_frames - (split - 1)))


def check_gen3c(scene_dir, ids):
    f = os.path.join(scene_dir, "pred_rgb.npy")
    if not os.path.exists(f):
        return "no pred_rgb.npy"
    try:
        a = np.load(f, mmap_mode="r")
    except Exception as e:
        return f"unreadable pred_rgb.npy: {e}"
    if a.ndim != 4 or a.shape[-1] != 3:
        return f"unexpected shape {tuple(a.shape)}, want (T,H,W,3)"
    if a.dtype != np.uint8:
        return f"dtype {a.dtype}, scorer divides by 255 so it MUST be uint8"
    bad = [i for i in ids if i >= a.shape[0]]
    if bad:
        return f"emitted {a.shape[0]} frames; ids {bad[:4]}{'...' if len(bad)>4 else ''} out of range"
    return None


def check_gen3r(scene_dir, ids):
    f = os.path.join(scene_dir, "pred_rgb_560.npy")
    if not os.path.exists(f):
        return "no pred_rgb_560.npy"
    try:
        a = np.load(f, mmap_mode="r")
    except Exception as e:
        return f"unreadable pred_rgb_560.npy: {e}"
    if a.ndim != 4 or a.shape[-1] != 3:
        return f"unexpected shape {tuple(a.shape)}, want (T,H,W,3)"
    # Gen3R is architecturally 49 slots: conditions on 0 and 48, predicts 1..47.
    if a.shape[0] != 49:
        return f"emitted {a.shape[0]} frames; Gen3R is fixed at 49"
    bad = [i for i in ids if i >= 49]
    if bad:
        return (f"ids {bad[:4]}{'...' if len(bad)>4 else ''} exceed Gen3R's 49 slots -- "
                f"use --max_frame on the scorer, do NOT silently drop them")
    return None


CHECK = {"gen3c": check_gen3c, "gen3r": check_gen3r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(CHECK))
    ap.add_argument("--run_dir", required=True, help="preds dir holding sample_XXXXX/ or <scene>/ subdirs")
    ap.add_argument("--split", type=int, default=2, choices=[1, 2])
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--expect_scenes", type=int, default=128,
                    help="benchmark size; a short cell is the #1 cause of a misleading average")
    a = ap.parse_args()

    ids = scored_ids(a.n_frames, a.split)
    scenes = sorted(d for d in glob.glob(os.path.join(a.run_dir, "*")) if os.path.isdir(d))
    if not scenes:
        print(f"FAIL: no scene subdirectories under {a.run_dir}")
        return 1

    ok, bad = [], []
    seeds = {}
    for d in scenes:
        err = CHECK[a.method](d, ids)
        (ok if err is None else bad).append((os.path.basename(d), err))
        p = os.path.join(d, "provenance.json")
        if os.path.exists(p):
            try:
                seeds[json.load(open(p)).get("n_seed")] = seeds.get(json.load(open(p)).get("n_seed"), 0) + 1
            except Exception:
                pass

    print(f"method={a.method}  run_dir={a.run_dir}")
    print(f"scored frame ids: {ids[0]}..{ids[-1]}  ({len(ids)} frames, split {a.split})")
    print(f"scenes: {len(scenes)} found, {len(ok)} valid, {len(bad)} would be DROPPED by the scorer")
    if len(scenes) != a.expect_scenes:
        print(f"  WARNING: expected {a.expect_scenes} scenes -- the average would be over the wrong set")
    if seeds:
        print(f"  n_seed distribution (conditioning views): {seeds}")
        if set(seeds) == {1}:
            print("  NOTE: seeded from ONE view. ours/SEVA use two (ncf2), so this cell is a")
            print("        LOWER BOUND and is NOT a like-for-like comparison.")
    for name, err in bad[:10]:
        print(f"   DROP {name}: {err}")
    if len(bad) > 10:
        print(f"   ... and {len(bad)-10} more")
    return 1 if (bad or len(scenes) != a.expect_scenes) else 0


if __name__ == "__main__":
    sys.exit(main())

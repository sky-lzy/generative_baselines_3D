#!/usr/bin/env python3
"""Build the FAIR 128-scene RE10K NVS scene sets (Strategy B, same pixels for every method).

Why this exists
---------------
Comparing our model against SEVA on the stock benchmark is not apples-to-apples on the INPUT side:

  * SEVA runs its pipeline at 576x576 (or 768x576) while we run at 384x288, i.e. 4x the pixels.
  * On the official 4DiM protocol the two also see DIFFERENT CROPS: the spec says "centre crop to
    576" (square), while our loader takes a 4:3 crop. From a 640x360 source that is a 360x360
    square for SEVA versus a 480x360 rectangle for us -- 33% more horizontal field of view for us,
    and ~40% of our render then discarded before scoring.

SEVA's own benchmark README documents two legitimate patterns, and explicitly says preprocessing is
applied to the MODEL INPUT and postprocessing to the MODEL OUTPUT:

  A. "center crop to 576"                        (square; 4DiM / ReconFusion / pixelSplat splits)
  B. "resize the shortest side to 576 (--L_short)" + "center crop" on the output
                                                 (the ViewCrafter splits)

This builder implements **Strategy B**: one aspect-preserving 4:3 image store that BOTH methods
consume, with the square crop moved to the OUTPUT side at scoring time. SEVA is architecturally
fine with this -- it is a fully-convolutional UNet whose attention flattens (h w) at runtime with no
learned absolute position table, so any H,W divisible by 64 works; our own runs on the ViewCrafter
splits came out 1024x576, not square.

Pixel parity
------------
Images are written at **384x288**, OUR budget. SEVA then runs at 576x768, which is an exact 2x
upsample of these same pixels: it operates at its trained scale but receives no information we do
not have. Intrinsics are rebased for the crop+resize here, and SEVA's own transform_img_and_K
rescales K again for its 2x resize -- so the camera model stays correct end-to-end and there is no
silent inference bug from mismatched intrinsics.

Frame windows
-------------
Source scenes are 71 frames (indices 0..70). Two split views share one image store:

  re10k128_50f   clip = frames 0..49  (--n_frames 50, our TRAINING clip length)
                 split_1: train [0]     test 1..49     (single input view)
                 split_2: train [0,49]  test 1..48     (two input views)

  re10k128_4dim  clip = frames 0..69  (--n_frames 70, the 4DiM target spacing)
                 split_1: train [0]     test 10,20,..,60
                 split_2: train [0,69]  test 10,20,..,60

The 4DiM protocol's 7th target is frame 70, which is STRUCTURALLY unreachable for us: the VAE's
first-last-frame chunking needs clip length % 4 == 2, so 71 frames clamps to 70 -> indices 0..69.
Scoring SEVA on 7 targets while we get 6 silently drops our furthest and hardest view (worth ~0.4 dB
in our favour), so both splits here define the SAME 6 reachable targets for both methods.

Usage
-----
  python3 nvs/data/build_fair_scenes.py --out_root <scenes_fair>            # build both sets
  python3 nvs/data/build_fair_scenes.py --verify_only --out_root <...>      # re-run checks only
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SRC = "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes/re10k-4dim"
OUT_W, OUT_H = 384, 288                      # our native 4:3 budget
SETS = {
    # name           n_frames  split_1 (1 input)                     split_2 (2 inputs)
    "re10k128_50f":  (50,  ([0], list(range(1, 50))),  ([0, 49], list(range(1, 49)))),
    "re10k128_4dim": (70,  ([0], list(range(10, 61, 10))), ([0, 69], list(range(10, 61, 10)))),
}


def crop_resize_box(w, h, out_w=OUT_W, out_h=OUT_H):
    """Centre crop `w x h` to the out aspect, then uniform-scale to out. Returns (x0,y0,cw,ch,s)."""
    target = out_w / out_h
    if w / h > target:                      # too wide -> crop width
        cw, ch = int(round(h * target)), h
    else:                                   # too tall -> crop height
        cw, ch = w, int(round(w / target))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, cw, ch, out_w / cw       # uniform scale (out_h/ch is equal by construction)


def build(out_root: Path, limit=None):
    scenes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    if limit:
        scenes = scenes[:limit]
    store = out_root / "_images"            # shared image store; the two set views symlink into it
    store.mkdir(parents=True, exist_ok=True)

    for si, sc in enumerate(scenes):
        meta = json.load(open(f"{SRC}/{sc}/transforms.json"))
        frames = meta["frames"]
        sdir = store / sc
        img_dir = sdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        new_frames = []
        for fr in frames:
            src_img = os.path.join(SRC, sc, fr["file_path"])
            name = os.path.basename(fr["file_path"])
            dst = img_dir / name
            with Image.open(src_img) as im:
                im = im.convert("RGB")
                w, h = im.size
                x0, y0, cw, ch, s = crop_resize_box(w, h)
                if not dst.exists():
                    im.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, OUT_H), Image.LANCZOS).save(dst)
            # intrinsics: translate for the crop, then uniform scale. Source K is in PIXELS.
            nf = dict(fr)
            nf["fl_x"], nf["fl_y"] = fr["fl_x"] * s, fr["fl_y"] * s
            nf["cx"], nf["cy"] = (fr["cx"] - x0) * s, (fr["cy"] - y0) * s
            nf["w"], nf["h"] = OUT_W, OUT_H
            nf["file_path"] = f"./images/{name}"
            new_frames.append(nf)
        json.dump({"orientation_override": meta.get("orientation_override", "none"),
                   "frames": new_frames}, open(sdir / "transforms.json", "w"))
        if (si + 1) % 32 == 0:
            print(f"  images {si + 1}/{len(scenes)}", flush=True)

    # two split views over the same store
    for name, (nfr, (tr1, te1), (tr2, te2)) in SETS.items():
        root = out_root / name
        root.mkdir(parents=True, exist_ok=True)
        for sc in scenes:
            d = root / sc
            d.mkdir(exist_ok=True)
            link = d / "images"
            if not link.exists():
                os.symlink(os.path.relpath(store / sc / "images", d), link)
            tj = d / "transforms.json"
            if not tj.exists():
                os.symlink(os.path.relpath(store / sc / "transforms.json", d), tj)
            json.dump({"train_ids": tr1, "test_ids": te1}, open(d / "train_test_split_1.json", "w"))
            json.dump({"train_ids": tr2, "test_ids": te2}, open(d / "train_test_split_2.json", "w"))
        json.dump({"n_frames": nfr, "note": "clip = first n_frames of the 71-frame source"},
                  open(root / "_clip.json", "w"))
        print(f"  built {name}: {len(scenes)} scenes, clip {nfr} frames")
    return scenes


def verify(out_root: Path):
    """Hard checks. Any failure raises -- a silently wrong scene set poisons every downstream number."""
    import torch
    ok = True
    for name, (nfr, (tr1, te1), (tr2, te2)) in SETS.items():
        root = out_root / name
        scenes = sorted(d for d in os.listdir(root) if os.path.isdir(root / d))
        assert scenes, f"{name}: no scenes"
        print(f"\n[{name}] {len(scenes)} scenes, clip {nfr}")
        for sc in scenes:
            d = root / sc
            meta = json.load(open(d / "transforms.json"))
            fr = meta["frames"]
            n_img = len(list((d / "images").glob("*.png")))
            assert len(fr) == n_img, f"{sc}: {len(fr)} frames vs {n_img} images"
            assert len(fr) >= nfr, f"{sc}: only {len(fr)} frames < clip {nfr}"
            s1 = json.load(open(d / "train_test_split_1.json"))
            s2 = json.load(open(d / "train_test_split_2.json"))
            assert s1["train_ids"] == tr1 and s1["test_ids"] == te1, f"{sc}: split_1 mismatch"
            assert s2["train_ids"] == tr2 and s2["test_ids"] == te2, f"{sc}: split_2 mismatch"
            # every id must fall inside the clip the model will actually generate
            for tag, ids in (("split_1", te1 + tr1), ("split_2", te2 + tr2)):
                bad = [i for i in ids if i >= nfr]
                assert not bad, f"{sc}: {tag} ids {bad} outside clip 0..{nfr - 1}"
            with Image.open(d / "images" / os.path.basename(fr[0]["file_path"])) as im:
                assert im.size == (OUT_W, OUT_H), f"{sc}: image {im.size} != {(OUT_W, OUT_H)}"
            assert fr[0]["w"] == OUT_W and fr[0]["h"] == OUT_H, f"{sc}: transforms w/h wrong"
        # intrinsics sanity on one scene: principal point should sit near the centre after rebasing
        d = root / scenes[0]
        f0 = json.load(open(d / "transforms.json"))["frames"][0]
        assert 0.3 * OUT_W < f0["cx"] < 0.7 * OUT_W, f"cx {f0['cx']} not near centre"
        assert 0.3 * OUT_H < f0["cy"] < 0.7 * OUT_H, f"cy {f0['cy']} not near centre"
        print(f"  cx,cy = {f0['cx']:.1f},{f0['cy']:.1f}  (centre {OUT_W/2:.0f},{OUT_H/2:.0f})"
              f"   fl_x,fl_y = {f0['fl_x']:.1f},{f0['fl_y']:.1f}")

        # POSE PARITY: relative extrinsics must be unchanged from the source (we only touched
        # pixels and intrinsics; any drift here means the scene set is geometrically different).
        src_fr = json.load(open(f"{SRC}/{scenes[0]}/transforms.json"))["frames"]
        A = np.array([f["transform_matrix"] for f in src_fr[:nfr]], dtype=np.float64)
        B = np.array([f["transform_matrix"] for f in
                      json.load(open(d / "transforms.json"))["frames"][:nfr]], dtype=np.float64)
        rel_a = np.linalg.inv(A[0]) @ A
        rel_b = np.linalg.inv(B[0]) @ B
        err = np.abs(rel_a - rel_b).max()
        assert err < 1e-9, f"pose drift {err}"
        print(f"  relative-extrinsic max err vs source: {err:.2e}  (poses untouched)")
    print("\nALL CHECKS PASSED" if ok else "\nFAILURES")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
    ap.add_argument("--limit", type=int, default=None, help="build only the first N scenes (smoke test)")
    ap.add_argument("--verify_only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_root)
    if not a.verify_only:
        build(out, a.limit)
    verify(out)

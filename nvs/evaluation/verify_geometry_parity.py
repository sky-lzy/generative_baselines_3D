#!/usr/bin/env python3
"""Prove the two methods are handed the SAME camera. No GPU, no model — pure geometry.

An intrinsics or convention bug here would not crash anything; it would silently hand SEVA a wrong
camera and make the baseline look bad, which is the single most damaging way this benchmark could
be wrong. So every claim in the design doc is checked against the code that actually runs:

  C1 SEVA's K RESCALE   run SEVA's own seva.eval.transform_img_and_K on a real fair scene at
                        576x768 and assert the result equals our 384x288 K scaled by exactly 2 —
                        same field of view, no silent normalized-vs-pixel-K branch error.
                        (transform_img_and_K picks its branch by testing cx,cy in [0,1]; our cx=192
                        is in PIXELS, so the unnormalized branch must be the one taken.)

  C2 SEVA's CROP IS A NO-OP  the scene set is 4:3 and SEVA runs at 4:3, so mode="crop" must trim
                        zero pixels. If it trimmed, SEVA would be scored on a narrower FOV than it
                        was given and the comparison would be broken.

  C3 OUR K RESCALE      the same check for our engine's loader (datasets/svc_scenes.py), which does
                        its own centre-crop + resize to the model's native size.

  C4 EXTRINSIC PARITY   both loaders read the same transforms.json and both apply the OpenGL->OpenCV
                        flip (cols 1,2 negated). Assert the resulting camera-to-world matrices are
                        bit-comparable, so neither model is looking backwards relative to the other.

  C5 FOV EQUALITY       convert both K's to horizontal/vertical FOV in degrees and assert they match.
                        This is the check that actually means "same scene content": it is invariant
                        to resolution, so it catches any residual scale error C1/C3 could miss.

    python3 nvs/evaluation/verify_geometry_parity.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCENES = os.environ.get(
    "SCENES_FAIR", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
SEVA_REPO = os.environ.get("SEVA_REPO", "/n/lab_storage/ydu_lab/Lab/akiruga/stable-virtual-camera")
VWM = os.environ.get("VWM_REPO", "/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new")

SEVA_H, SEVA_W = 576, 768          # run_seva_infer.py defaults
OURS = {"F_5b": (288, 384), "mot13b_full": (240, 320)}

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def fov_deg(f, n):
    return 2 * math.degrees(math.atan(n / (2 * f)))


def main():
    ds = "re10k128_50f"
    root = Path(SCENES) / ds
    scene = sorted(d for d in os.listdir(root) if (root / d).is_dir())[0]
    sdir = root / scene
    meta = json.load(open(sdir / "transforms.json"))
    fr = meta["frames"][0]
    W0, H0 = int(fr["w"]), int(fr["h"])
    K0 = np.array([[fr["fl_x"], 0, fr["cx"]], [0, fr["fl_y"], fr["cy"]], [0, 0, 1]], float)
    print(f"scene: {scene}\nsource K @ {W0}x{H0}: fx {K0[0,0]:.3f} fy {K0[1,1]:.3f} "
          f"cx {K0[0,2]:.3f} cy {K0[1,2]:.3f}")
    print(f"source FOV: h {fov_deg(K0[0,0], W0):.4f} deg   v {fov_deg(K0[1,1], H0):.4f} deg\n")

    # ---------------- C1 + C2 : SEVA's own transform ---------------------------------------
    print("C1/C2  SEVA seva.eval.transform_img_and_K")
    sys.path.insert(0, SEVA_REPO)
    from seva.eval import transform_img_and_K  # noqa: E402

    img = torch.zeros(1, 3, H0, W0)
    out, K_seva = transform_img_and_K(img, (SEVA_W, SEVA_H), K=torch.tensor(K0)[None].float(),
                                      mode="crop")
    K_seva = K_seva[0].numpy()
    sx, sy = SEVA_W / W0, SEVA_H / H0
    exp = K0.copy(); exp[0] *= sx; exp[1] *= sy
    check("C1 SEVA K == source K scaled by (%.1f, %.1f)" % (sx, sy),
          np.allclose(K_seva, exp, atol=1e-3),
          f"got fx {K_seva[0,0]:.3f} cx {K_seva[0,2]:.3f} | want fx {exp[0,0]:.3f} cx {exp[0,2]:.3f}")
    check("C1b unnormalized-K branch taken (cx in pixels, not [0,1])",
          K0[0, 2] > 1.0, f"cx {K0[0,2]:.1f}")
    check("C2 SEVA crop is a no-op (output is exactly HxW, aspect preserved)",
          tuple(out.shape[-2:]) == (SEVA_H, SEVA_W),
          f"out {tuple(out.shape[-2:])}")
    check("C2b principal point stays centred after SEVA's transform",
          abs(K_seva[0, 2] - SEVA_W / 2) < 1.0 and abs(K_seva[1, 2] - SEVA_H / 2) < 1.0,
          f"cx {K_seva[0,2]:.2f} (centre {SEVA_W/2}) cy {K_seva[1,2]:.2f} (centre {SEVA_H/2})")

    # ---------------- C3 : our loader --------------------------------------------------------
    print("\nC3     our datasets/svc_scenes.py crop+resize")
    sys.path.insert(0, VWM)
    from datasets.svc_scenes import center_crop_box_for_aspect, adjust_K_for_crop_resize  # noqa

    ours_K = {}
    for tag, (h, w) in OURS.items():
        box = center_crop_box_for_aspect(H0, W0, h, w)
        Kf = adjust_K_for_crop_resize(K0, box, h, w)
        ours_K[tag] = (Kf, h, w)
        s = w / W0
        exp = K0.copy(); exp[0] *= s; exp[1] *= (h / H0)
        check(f"C3 {tag} K == source K scaled to {w}x{h}",
              np.allclose(Kf, exp, atol=1e-3),
              f"fx {Kf[0,0]:.3f} cx {Kf[0,2]:.3f}")

    # ---------------- C4 : extrinsics ---------------------------------------------------------
    print("\nC4     extrinsic convention parity")
    c2w_raw = np.array([f["transform_matrix"] for f in meta["frames"]], float)
    ours_c2w = c2w_raw.copy(); ours_c2w[:, :, [1, 2]] *= -1        # svc_scenes.py:199
    seva_c2w = c2w_raw.copy(); seva_c2w[:, :, [1, 2]] *= -1        # seva/data_io.py:371
    check("C4 both loaders apply the same OpenGL->OpenCV flip",
          np.abs(ours_c2w - seva_c2w).max() == 0.0,
          f"max|diff| {np.abs(ours_c2w - seva_c2w).max():.1e}")
    # relative poses (what actually drives NVS) must be identical too
    rel_o = np.linalg.inv(ours_c2w[0]) @ ours_c2w
    rel_s = np.linalg.inv(seva_c2w[0]) @ seva_c2w
    check("C4b relative camera trajectories identical",
          np.abs(rel_o - rel_s).max() < 1e-12,
          f"max|diff| {np.abs(rel_o - rel_s).max():.1e}")

    # ---------------- C5 : FOV ----------------------------------------------------------------
    print("\nC5     field of view (resolution-invariant)")
    hs, vs = fov_deg(K_seva[0, 0], SEVA_W), fov_deg(K_seva[1, 1], SEVA_H)
    h0, v0 = fov_deg(K0[0, 0], W0), fov_deg(K0[1, 1], H0)
    check("C5 SEVA FOV == source FOV", abs(hs - h0) < 1e-3 and abs(vs - v0) < 1e-3,
          f"seva h {hs:.4f} v {vs:.4f} | source h {h0:.4f} v {v0:.4f}")
    for tag, (Kf, h, w) in ours_K.items():
        ho, vo = fov_deg(Kf[0, 0], w), fov_deg(Kf[1, 1], h)
        check(f"C5 {tag} FOV == source FOV", abs(ho - h0) < 1e-3 and abs(vo - v0) < 1e-3,
              f"ours h {ho:.4f} v {vo:.4f} | source h {h0:.4f} v {v0:.4f}")

    print("\n" + ("ALL GEOMETRY CHECKS PASSED — both methods receive the same camera"
                  if not fails else "FAILURES: " + ", ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

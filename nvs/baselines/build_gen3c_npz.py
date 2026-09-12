#!/usr/bin/env python
"""Build GEN3C multiview .npz inputs from a fair NVS scene, using BOTH ncf2 views.

Why: our existing Gen3C integration used cosmos_predict1...gen3c_single_image and seeded
from ONE frame, while ours/SEVA get the two ncf2 conditioning views. GEN3C itself supports
multiview (cosmos_predict1.diffusion.inference.gen3c_multiview), so the single-view
restriction was OUR wrapper's, not the model's. This makes the like-for-like run possible.

gen3c_multiview reads (see its __main__):
    images_key_frames (N,C,H,W) float [-1,1]     depth_key_frames (N,1,H,W)
    mask_key_frames   (N,1,H,W)                  K_key_frames     (N,3,3) OpenCV, pixels
    w2cs_key_frames   (N,4,4)                    w2cs_all (T,4,4)   Ks_all (T,3,3) optional

GEOMETRY. transforms.json holds nerfstudio/OpenGL c2w:
    c2w_nerf = inv(w2c_opencv) @ FLIP,  FLIP = diag(1,-1,-1,1)
so, since FLIP is its own inverse,
    w2c_opencv = inv(c2w_nerf @ FLIP)
This is the same convention verified bit-exactly in build_test_scenes_raw.py.

DEPTH is NOT in the scene dir and must be supplied (--depth_npz), from a multi-view
estimator that makes the two views SHARE A SCALE (VGGT), then rescaled so its camera
baseline matches the GT baseline. Two independent monocular depths would not share scale --
that part of the old note was right; it just isn't a property of GEN3C.
"""
import argparse, json, os
import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def load_scene(scene_dir, split, n_frames):
    t = json.load(open(os.path.join(scene_dir, f"transforms.json")))
    frames = t["frames"][:n_frames]
    sp = json.load(open(os.path.join(scene_dir, f"train_test_split_{split}.json")))
    w2cs, Ks = [], []
    for f in frames:
        c2w_nerf = np.array(f["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w_nerf @ FLIP)
        w2cs.append(w2c)
        Ks.append(np.array([[f["fl_x"], 0, f["cx"]], [0, f["fl_y"], f["cy"]], [0, 0, 1]], dtype=np.float64))
    return np.stack(w2cs), np.stack(Ks), sp["train_ids"], sp["test_ids"], frames


def check_geometry(w2cs):
    """Fail loudly on a bad convention rather than produce a plausible-but-wrong npz."""
    errs = []
    for i, w in enumerate(w2cs):
        R = w[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
            errs.append(f"frame {i}: R not orthonormal (max dev {np.abs(R@R.T-np.eye(3)).max():.2e})")
        d = np.linalg.det(R)
        if not np.isclose(d, 1.0, atol=1e-4):
            errs.append(f"frame {i}: det(R)={d:.6f}, want +1 (a -1 means a handedness flip)")
        if not np.allclose(w[3], [0, 0, 0, 1], atol=1e-8):
            errs.append(f"frame {i}: bottom row {w[3]} != [0,0,0,1]")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--depth_npz", default=None,
                    help="(N,H,W) scale-consistent depth for the key frames, VGGT-derived and "
                         "baseline-rescaled. Omit for --check_only.")
    ap.add_argument("--split", type=int, default=2, choices=[1, 2])
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--check_only", action="store_true", help="validate geometry, write nothing")
    a = ap.parse_args()

    w2cs, Ks, train_ids, test_ids, frames = load_scene(a.scene_dir, a.split, a.n_frames)
    errs = check_geometry(w2cs)
    print(f"scene   : {os.path.basename(a.scene_dir.rstrip('/'))}")
    print(f"frames  : {len(w2cs)}  key(cond)={train_ids}  scored={test_ids[0]}..{test_ids[-1]} (n={len(test_ids)})")
    print(f"intrinsics f=({Ks[0][0,0]:.2f},{Ks[0][1,1]:.2f}) c=({Ks[0][0,2]:.2f},{Ks[0][1,2]:.2f})")
    C = np.stack([-w[:3, :3].T @ w[:3, 3] for w in w2cs])       # camera centres in world
    base = np.linalg.norm(C[train_ids[1]] - C[train_ids[0]])
    print(f"GT baseline between the two key views: {base:.6f} world units")
    print(f"trajectory extent: {np.linalg.norm(C.max(0)-C.min(0)):.6f}")
    if errs:
        print("GEOMETRY ERRORS:"); [print("  " + e) for e in errs[:5]]
        return 1
    print("geometry OK: all R orthonormal, det=+1, bottom row canonical")
    if a.check_only:
        print("check_only -> nothing written"); return 0
    if not a.depth_npz:
        print("ERROR: --depth_npz required to write the npz (GEN3C needs key-frame depth)"); return 1
    # (writer lands with the VGGT stage; geometry is the part that must be right first)
    print("ERROR: depth stage not yet wired -- refusing to emit an npz with placeholder depth")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

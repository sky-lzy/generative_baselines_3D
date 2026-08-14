#!/usr/bin/env python3
"""SEVA (Stable Virtual Camera) baseline bridge for the fair NVS benchmark.

Mirrors inference/run_nvs_infer.py: this file adds NO modelling logic, it only builds the exact
`demo.py` command SEVA ships with and runs it once per scene (per-scene so a single bad scene cannot
take out a whole cell, and so re-running skips finished scenes).

RESOLUTION — the fairness-critical part
---------------------------------------
Run at **H=576 W=768**. Two constraints force this pair:

  * SEVA's UNet halves resolution per level and its VAE downsamples by F, so `demo.py` requires both
    sides divisible by F*2**3 = 64. 384x288 (our native size) is NOT usable: 288/64 = 4.5.
  * 576x768 is 4:3, matching the scene set's aspect exactly, and is exactly 2x our 384x288 images.

Because the scene set is already written at 384x288 (see nvs/data/build_fair_scenes.py), SEVA's 2x
is a pure upsample of the SAME pixels -- it operates at its trained scale but receives no information
we do not have. This is Strategy B from SEVA's own benchmark README (aspect-preserving input, square
crop moved to the output/scoring side).

INTRINSICS — why we do not pre-resize the images ourselves
----------------------------------------------------------
`demo.py` -> `seva/eval.py::transform_img_and_K` resizes the image AND rescales K in the same call:

    K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]
    K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

Handing SEVA pre-upsampled pixels with the scene's original K would leave focal/principal point in
384x288 units while the pixels are 576x768 -- a silent 2x camera error that would wreck its geometry
and make the baseline look far worse than it is. So we pass the scene dir untouched and let SEVA's
own code do the resize+K rescale together. That is the "no inference bug" guarantee.

The 4:3 aspect also means the internal crop is a no-op: the target aspect already equals the source
aspect, so `mode="crop"` trims nothing and no field of view is lost on either side.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# SEVA writes cwd-relative work_dirs/, and the /net/holy-isilon checkout is a read-only mount,
# so inference MUST run from the writable copy (this already caused one silent total failure).
SEVA_REPO_DEFAULT = "/n/lab_storage/ydu_lab/Lab/akiruga/stable-virtual-camera"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="fair scene-set dir <scenes_fair>/<set>")
    ap.add_argument("--output_dir", required=True,
                    help="preds cell dir; a symlink is pointed at SEVA's work_dirs output")
    ap.add_argument("--num_cond_frames", type=int, choices=[1, 2], required=True,
                    help="-> demo.py --num_inputs; selects train_test_split_<k>.json")
    ap.add_argument("--camera_scale", type=float, default=2.0,
                    help="SEVA's unit-length dial (its default is 2.0); the analogue of our "
                         "moment_scale_mult, swept identically on both sides")
    ap.add_argument("--H", type=int, default=576)
    ap.add_argument("--W", type=int, default=768)
    ap.add_argument("--seva_repo", default=os.environ.get("SEVA_REPO", SEVA_REPO_DEFAULT))
    ap.add_argument("--save_subdir", required=True, help="work_dirs/demo/img2img/<save_subdir>")
    ap.add_argument("--scenes", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--timeout", type=int, default=2400, help="per-scene seconds")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    for d in (a.H, a.W):
        if d % 64:
            sys.exit(f"ERROR: SEVA needs H,W divisible by 64 (F*2**3); got {a.H}x{a.W}")

    scenes = ([s for s in a.scenes.split(",") if s] if a.scenes
              else sorted(d for d in os.listdir(a.data_root)
                          if os.path.isdir(os.path.join(a.data_root, d))))
    split_name = f"train_test_split_{a.num_cond_frames}.json"
    n_expected = len(json.load(open(Path(a.data_root) / scenes[0] / split_name))["test_ids"])

    out_root = Path(a.seva_repo) / "work_dirs" / "demo" / "img2img" / a.save_subdir
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/n/netscratch/ydu_lab/Lab/akiruga")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[seva] {len(scenes)} scenes | {a.H}x{a.W} | num_inputs={a.num_cond_frames} "
          f"| camera_scale={a.camera_scale} | expect {n_expected} pngs/scene", flush=True)

    failures = []
    for i, sc in enumerate(scenes):
        done = out_root / sc / "samples-rgb"
        if done.is_dir() and len(list(done.glob("*.png"))) >= n_expected:
            print(f"  [{i+1}/{len(scenes)}] SKIP {sc} (complete)", flush=True)
            continue
        cmd = [sys.executable, "-u", "demo.py",
               f"--data_path={a.data_root}", f"--data_items=['{sc}']",
               "--task=img2img", f"--num_inputs={a.num_cond_frames}",
               f"--H={a.H}", f"--W={a.W}", f"--camera_scale={a.camera_scale}",
               f"--save_subdir={a.save_subdir}"]
        if a.dry_run:
            print("  " + " ".join(cmd)); continue
        r = subprocess.run(cmd, cwd=a.seva_repo, env=env, timeout=a.timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        n = len(list(done.glob("*.png"))) if done.is_dir() else 0
        ok = (r.returncode == 0 and n >= n_expected)
        print(f"  [{i+1}/{len(scenes)}] {sc} rc={r.returncode} pngs={n}/{n_expected} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(sc)
            print("   ...tail:", "\n   ".join(r.stdout.strip().splitlines()[-6:]), flush=True)

    if a.dry_run:
        return
    # expose SEVA's output under the standard preds path so the scorer takes one --run_dir form
    outp = Path(a.output_dir)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.is_symlink() or outp.exists():
        if outp.is_symlink():
            outp.unlink()
    if not outp.exists():
        os.symlink(out_root, outp)
    print(f"[seva] {a.output_dir} -> {out_root}")
    if failures:
        sys.exit(f"[seva] {len(failures)} scene(s) FAILED: {', '.join(failures[:8])}")


if __name__ == "__main__":
    main()

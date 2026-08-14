#!/usr/bin/env python3
"""Build GT | ours | SEVA comparison videos for the fair NVS report.

Every frame in the strip goes through the SAME to_metric_domain() the scorer uses (centre square
crop -> resize), so what you watch is literally what was measured. A qualitative panel built from
raw model output at each method's native resolution would show SEVA at 2x our size and invite the
eye to credit it for detail the metrics never saw.

Two video kinds:

  compare   GT | <ours> | SEVA for one scene, side by side, target frames only. Used for the
            best / median / worst picks per benchmark.
  scales    one row per swept scale for a single method+scene, so the scale search is visible
            rather than asserted. The picked scale is labelled.

Frames come from the same lossless sources the scorer reads (raw_arrays.npz for ours, PNGs for
SEVA), never from pred_rgb.mp4 — re-encoding an H.264 file into another H.264 file would show
compression artefacts that played no part in any number in the table.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS / "evaluation"))
from score_nvs_fair import read_ours, read_seva, read_gt, to_metric_domain  # noqa: E402

LABEL_H = 22


def ffmpeg_exe():
    """ffmpeg is NOT on PATH on these login/compute nodes. Prefer the conda env's binary, fall back
    to the one imageio-ffmpeg ships, and only then to PATH — so video building never silently
    depends on a module being loaded."""
    cand = Path(sys.executable).parent / "ffmpeg"
    if cand.exists():
        return str(cand)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def label(img_np, text):
    """Stamp a caption bar above a HxWx3 uint8 frame."""
    h, w = img_np.shape[:2]
    out = Image.new("RGB", (w, h + LABEL_H), (18, 18, 20))
    out.paste(Image.fromarray(img_np), (0, LABEL_H))
    d = ImageDraw.Draw(out)
    d.text((5, 5), text[:40], fill=(235, 235, 235))
    return np.array(out)


def to_uint8(x):
    return (x.permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def write_mp4(frames, path, fps=10):
    """H.264 via ffmpeg pipe. yuv420p + even dims so browsers will actually play it."""
    h, w = frames[0].shape[:2]
    w -= w % 2; h -= h % 2
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{frames[0].shape[1]}x{frames[0].shape[0]}", "-r", str(fps), "-i", "-",
           "-vf", f"crop={w}:{h}:0:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    return p.wait() == 0


SEVA_WORK = Path("/n/lab_storage/ydu_lab/Lab/akiruga/stable-virtual-camera"
                 "/work_dirs/demo/img2img")


def load_pred(preds, cell, method, scenes_root, scene, idx, ids, size):
    d = Path(preds) / cell
    if method == "seva":
        # SEVA's preds_fair entry is a symlink created only when a shard job EXITS, so fall back to
        # its real output root -- otherwise a cell that is finished but un-symlinked renders nothing.
        if not (d / scene).exists() and (SEVA_WORK / f"fair_{cell}" / scene).exists():
            d = SEVA_WORK / f"fair_{cell}"
        x, err = read_seva(str(d / scene), ids)
    else:
        x, err = read_ours(str(d / f"sample_{idx:05d}"), ids)
    if x is None:
        return None, err
    return to_metric_domain(x, size), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(NVS / "preds_fair"))
    ap.add_argument("--scenes_fair",
                    default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
    ap.add_argument("--out", default=str(NVS / "report" / "videos"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ncf", type=int, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cells", required=True,
                    help="comma list of label=cell entries, left to right after GT")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    sroot = Path(a.scenes_fair) / a.dataset
    scenes = sorted(d for d in os.listdir(sroot) if (sroot / d).is_dir())
    idx = scenes.index(a.scene)
    ids = sorted(json.load(open(sroot / a.scene / f"train_test_split_{a.ncf}.json"))["test_ids"])

    cols = [("GT", to_uint8(to_metric_domain(read_gt(str(sroot / a.scene), ids), a.size)))]
    for entry in a.cells.split(","):
        lab, cell = entry.split("=", 1)
        method = "seva" if "seva" in cell.lower() else "ours"
        x, err = load_pred(a.preds, cell, method, sroot, a.scene, idx, ids, a.size)
        if x is None:
            print(f"[warn] {lab} ({cell}): {err}")
            continue
        cols.append((lab, to_uint8(x)))

    if len(cols) < 2:
        sys.exit("nothing to render")
    n = min(len(c[1]) for c in cols)
    frames = [np.concatenate([label(c[1][t], c[0]) for c in cols], axis=1) for t in range(n)]
    out = Path(a.out) / f"{a.name}.mp4"
    ok = write_mp4(frames, out, a.fps)
    print(("wrote " if ok else "FAILED ") + str(out) + f"  ({n} frames, {len(cols)} panels)")


if __name__ == "__main__":
    main()

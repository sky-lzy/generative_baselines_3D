#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _list_tum_images(root, seq):
    return [str(p) for p in sorted((root / seq / "rgb_90").glob("*.png"))]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pi3 relpose-distance on a TUM frame window.")
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--pi3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.pi3_root))
    from pi3.models.pi3 import Pi3
    from relpose.evo_utils import eval_metrics, get_tum_poses, load_traj, plot_trajectory, save_tum_poses
    from utils.interfaces import infer_cameras_c2w

    filelist = _list_tum_images(args.tum_root, args.sequence)[args.start_index : args.start_index + args.num_frames]
    if len(filelist) != args.num_frames:
        raise ValueError(f"Expected {args.num_frames} images, got {len(filelist)}")

    model = Pi3.from_pretrained(args.checkpoint).to(args.device).eval()
    hydra_cfg = argparse.Namespace(device=args.device, no_crop=False, load_img_size=512, verbose=False)

    print(
        f"Pi3 TUM window sequence={args.sequence} start={args.start_index} frames={len(filelist)}",
        flush=True,
    )
    pred_c2w, pred_intrs = infer_cameras_c2w(filelist, model, hydra_cfg)
    pred_c2w = np.asarray(pred_c2w.detach().cpu().numpy() if hasattr(pred_c2w, "detach") else pred_c2w)
    pred_traj = get_tum_poses(pred_c2w)
    gt_traj = load_traj(
        str(args.tum_root / args.sequence / "groundtruth_90.txt"),
        traj_format="tum",
        skip=args.start_index,
        num_frames=args.num_frames,
    )

    seq_dir = args.output_dir / "tum" / args.sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / "pred_poses.npy", pred_c2w)
    if pred_intrs is not None:
        np.save(seq_dir / "pred_intrinsics.npy", np.asarray(pred_intrs))
    save_tum_poses(pred_traj, str(seq_dir / "pred_traj.txt"))
    ate, rpe_trans, rpe_rot = eval_metrics(pred_traj, gt_traj, seq=args.sequence, filename=str(seq_dir / "eval_metric.txt"))
    plot_trajectory(pred_traj, gt_traj, title=args.sequence, filename=str(seq_dir / "vis.png"))

    row = {
        "dataset": "tum",
        "seq": args.sequence,
        "start_index": args.start_index,
        "frames": args.num_frames,
        "ATE": float(ate),
        "RPE trans": float(rpe_trans),
        "RPE rot": float(rpe_rot),
    }
    _write_csv(args.output_dir / "tum" / "results.csv", [row])
    summary = {"dataset": "tum", "metrics": {"ATE": float(ate), "RPE trans": float(rpe_trans), "RPE rot": float(rpe_rot)}, "num_sequences": 1, "num_failures": 0, "failures": []}
    (args.output_dir / "tum" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

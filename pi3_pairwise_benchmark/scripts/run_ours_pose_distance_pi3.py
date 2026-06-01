#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np


SINTEL_SEQS = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]


def _load_adapter(adapter: str):
    module_name, sep, func_name = adapter.partition(":")
    if not sep:
        raise ValueError("--adapter must be in module:function form")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


class _Row:
    def __init__(self, sequence_name: str):
        self.sequence_name = sequence_name


def _list_images(dataset: str, root: Path, seq: str) -> list[str]:
    if dataset == "sintel":
        img_dir = root / "final" / seq
        ext = "png"
    elif dataset == "tum":
        img_dir = root / seq / "rgb_90"
        ext = "png"
    else:
        raise ValueError(dataset)
    return [str(p) for p in sorted(img_dir.glob(f"*.{ext}"))]


def _anno_path(dataset: str, root: Path, seq: str) -> Path:
    if dataset == "sintel":
        return root / "camdata_left" / seq
    if dataset == "tum":
        return root / seq / "groundtruth_90.txt"
    raise ValueError(dataset)


def _seqs(dataset: str, root: Path, requested: str | None) -> list[str]:
    if requested:
        return [s for s in requested.split(",") if s]
    if dataset == "sintel":
        return SINTEL_SEQS
    if dataset == "tum":
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    raise ValueError(dataset)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VWM on Pi3 relpose-distance datasets and evaluate with Pi3/evo metrics.")
    parser.add_argument("--dataset", choices=["sintel", "tum"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pi3-root", type=Path, required=True)
    parser.add_argument("--video-world-model-root", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--pose-eval-stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequences", default=None, help="Comma-separated sequence names.")
    parser.add_argument("--override-n-frames", default="auto", help="'auto', 'none', or an integer.")
    parser.add_argument("--override-height", type=int, default=None)
    parser.add_argument("--override-width", type=int, default=None)
    parser.add_argument(
        "--image-preprocess-mode",
        choices=["direct_resize", "center_crop_resize"],
        default="direct_resize",
    )
    parser.add_argument(
        "--adapter",
        default="pi3_pairwise_benchmark.video_world_model_adapter:predict_sequence_c2w",
        help="Callable module:function returning pred_c2w (N,4,4) for one full sequence.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.pi3_root))
    from relpose.evo_utils import eval_metrics, get_tum_poses, load_traj, plot_trajectory, save_tum_poses

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seqs = _seqs(args.dataset, args.data_root, args.sequences)
    if args.max_sequences is not None:
        seqs = seqs[: args.max_sequences]

    predict_sequence_c2w = _load_adapter(args.adapter)
    context = {
        "video_world_model_root": args.video_world_model_root,
        "ckpt_path": args.ckpt_path,
        "device": args.device,
        "sample_steps": args.sample_steps,
        "override_height": args.override_height,
        "override_width": args.override_width,
        "image_preprocess_mode": args.image_preprocess_mode,
    }

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    print(f"Loaded {len(seqs)} {args.dataset} sequences; output_dir={args.output_dir}", flush=True)
    for idx, seq in enumerate(seqs, start=1):
        try:
            filelist = _list_images(args.dataset, args.data_root, seq)[:: args.pose_eval_stride]
            if args.start_index:
                filelist = filelist[args.start_index :]
            if args.num_frames is not None:
                filelist = filelist[: args.num_frames]
            if not filelist:
                raise FileNotFoundError(f"No images found for {seq}")
            local_context = dict(context)
            if args.override_n_frames == "auto":
                local_context["override_n_frames"] = len(filelist)
            elif args.override_n_frames != "none":
                local_context["override_n_frames"] = int(args.override_n_frames)

            print(f"[{idx}/{len(seqs)}] {seq}: frames={len(filelist)} inference begin", flush=True)
            pred_c2w = np.asarray(
                predict_sequence_c2w(filelist, row=_Row(seq), context=local_context),
                dtype=np.float64,
            )
            if pred_c2w.shape != (len(filelist), 4, 4):
                raise ValueError(f"Adapter returned {pred_c2w.shape}, expected {(len(filelist), 4, 4)}")

            seq_dir = args.output_dir / args.dataset / seq
            seq_dir.mkdir(parents=True, exist_ok=True)
            np.save(seq_dir / "pred_poses.npy", pred_c2w)
            pred_traj = get_tum_poses(pred_c2w)
            save_tum_poses(pred_traj, str(seq_dir / "pred_traj.txt"))
            gt_traj = load_traj(
                gt_traj_file=str(_anno_path(args.dataset, args.data_root, seq)),
                traj_format=args.dataset,
                stride=args.pose_eval_stride,
                skip=args.start_index,
                num_frames=len(filelist),
            )
            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj,
                gt_traj,
                seq=seq,
                filename=str(seq_dir / "eval_metric.txt"),
            )
            plot_trajectory(pred_traj, gt_traj, title=seq, filename=str(seq_dir / "vis.png"))
            row = {
                "dataset": args.dataset,
                "seq": seq,
                "frames": len(filelist),
                "ATE": float(ate),
                "RPE trans": float(rpe_trans),
                "RPE rot": float(rpe_rot),
            }
            rows.append(row)
            print(f"[{idx}/{len(seqs)}] {seq}: ok {row}", flush=True)
        except Exception as exc:
            failures.append({"dataset": args.dataset, "seq": seq, "error": str(exc)})
            print(f"[{idx}/{len(seqs)}] {seq}: failed {exc}", flush=True)

    _write_csv(args.output_dir / args.dataset / "results.csv", rows)
    if rows:
        avg = {
            "ATE": float(np.mean([r["ATE"] for r in rows])),
            "RPE trans": float(np.mean([r["RPE trans"] for r in rows])),
            "RPE rot": float(np.mean([r["RPE rot"] for r in rows])),
        }
    else:
        avg = {}
    summary = {
        "dataset": args.dataset,
        "metrics": avg,
        "num_sequences": len(rows),
        "num_failures": len(failures),
        "failures": failures,
    }
    (args.output_dir / args.dataset / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

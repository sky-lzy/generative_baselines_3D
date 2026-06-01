#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import load_manifest, write_json, write_prediction_pair
from pi3_pairwise_benchmark.metrics import compute_pairwise_errors, summarize_pi3_metrics
from pi3_pairwise_benchmark.sampling import select_pairs


def _load_adapter(adapter: str):
    module_name, sep, func_name = adapter.partition(":")
    if not sep:
        raise ValueError("--adapter must be in module:function form")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run video_world_model 2-view pair inference on Pi3 Re10K samples.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-world-model-root", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--diffusion-mode", default="ray_depth_prediction_firstlast")
    parser.add_argument("--lang-guidance", type=float, default=0.0)
    parser.add_argument("--hist-guidance", type=float, default=1.0)
    parser.add_argument(
        "--adapter",
        default="pi3_pairwise_benchmark.video_world_model_adapter:predict_pair_c2w",
        help="Callable module:function returning pred_c2w (2,4,4) for one image pair.",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-pairs-per-sequence", type=int, default=None)
    parser.add_argument("--pair-policy", default="all", choices=["all", "short", "medium", "long", "mixed"])
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--short-gap-min", type=int, default=1)
    parser.add_argument("--short-gap-max", type=int, default=5)
    parser.add_argument("--medium-gap-min", type=int, default=10)
    parser.add_argument("--medium-gap-max", type=int, default=20)
    parser.add_argument("--long-gap-min", type=int, default=30)
    parser.add_argument("--long-gap-max", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest, require_images=True)
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]
    predict_pair_c2w = _load_adapter(args.adapter)

    all_r: list[float] = []
    all_t: list[float] = []
    failures: list[dict[str, str]] = []
    context = {
        "video_world_model_root": args.video_world_model_root,
        "ckpt_path": args.ckpt_path,
        "device": args.device,
        "sample_steps": args.sample_steps,
        "diffusion_mode": args.diffusion_mode,
        "lang_guidance": args.lang_guidance,
        "hist_guidance": args.hist_guidance,
    }
    print(f"Loaded {len(rows)} manifest rows; output_dir={args.output_dir}", flush=True)
    for row_idx, row in enumerate(rows):
        pairs = select_pairs(
            num_frames=len(row.image_paths),
            policy=args.pair_policy,
            max_pairs=args.max_pairs_per_sequence,
            seed=args.pair_seed,
            sequence_key=row.sequence_name,
            short_gap_min=args.short_gap_min,
            short_gap_max=args.short_gap_max,
            medium_gap_min=args.medium_gap_min,
            medium_gap_max=args.medium_gap_max,
            long_gap_min=args.long_gap_min,
            long_gap_max=args.long_gap_max,
        )
        print(f"[sequence {row_idx + 1}/{len(rows)}] {row.sequence_name}: {len(pairs)} pairs", flush=True)
        for pair_idx, pair in enumerate(pairs):
            image_paths = [row.image_paths[pair[0]], row.image_paths[pair[1]]]
            try:
                print(f"  pair {pair_idx + 1}/{len(pairs)} {pair}: inference begin", flush=True)
                pred_c2w = np.asarray(
                    predict_pair_c2w(image_paths=image_paths, pair=pair, row=row, context=context),
                    dtype=np.float64,
                )
                gt_w2c = row.gt_w2c[list(pair)]
                errs = compute_pairwise_errors(np.linalg.inv(pred_c2w), gt_w2c)
                metrics = summarize_pi3_metrics(errs.rotation_deg, errs.translation_deg)
                write_prediction_pair(args.output_dir, row.sequence_name, pair, pred_c2w, gt_w2c, metrics)
                all_r.extend(errs.rotation_deg.tolist())
                all_t.extend(errs.translation_deg.tolist())
                print(f"  pair {pair_idx + 1}/{len(pairs)} {pair}: ok {metrics}", flush=True)
            except Exception as exc:
                failures.append({"sequence": row.sequence_name, "pair": f"{pair[0]},{pair[1]}", "error": str(exc)})
                print(f"  pair {pair_idx + 1}/{len(pairs)} {pair}: failed {exc}", flush=True)

    report = {
        "metrics": summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t)) if all_r else {},
        "num_failures": len(failures),
        "failures": failures,
        "adapter": args.adapter,
    }
    write_json(args.output_dir / "summary.json", report)
    print(report, flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

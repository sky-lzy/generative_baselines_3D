#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import traceback
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import load_manifest, write_json, write_prediction_sequence
from pi3_pairwise_benchmark.metrics import compute_pairwise_errors, invert_se3_stack, summarize_pi3_metrics


def _load_adapter(adapter: str):
    module_name, sep, func_name = adapter.partition(":")
    if not sep:
        raise ValueError("--adapter must be in module:function form")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Geo4D 50-view inference with Pi3 relative-pose metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geo4d-root", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-no", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument(
        "--adapter",
        default="pi3_pairwise_benchmark.geo4d_adapter:predict_sequence_c2w",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest, require_images=True)
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]
    predict_sequence_c2w = _load_adapter(args.adapter)

    all_r: list[float] = []
    all_t: list[float] = []
    failures: list[dict[str, str]] = []
    context = {
        "geo4d_root": args.geo4d_root,
        "ckpt_path": args.ckpt_path,
        "config_path": args.config_path,
        "height": args.height,
        "width": args.width,
        "ddim_steps": args.ddim_steps,
        "stride": args.stride,
        "seed": args.seed,
        "gpu_no": args.gpu_no,
        "scratch_dir": args.output_dir,
    }
    print(f"Loaded {len(rows)} manifest rows; output_dir={args.output_dir}", flush=True)
    for row_idx, row in enumerate(rows):
        try:
            print(f"[sequence {row_idx + 1}/{len(rows)}] {row.sequence_name}: Geo4D inference begin", flush=True)
            pred_c2w = np.asarray(
                predict_sequence_c2w(image_paths=row.image_paths, row=row, context=context),
                dtype=np.float64,
            )
            if pred_c2w.shape != row.gt_w2c.shape:
                raise ValueError(f"Adapter returned {pred_c2w.shape}, expected {row.gt_w2c.shape}")
            pred_w2c = invert_se3_stack(pred_c2w)
            errs = compute_pairwise_errors(pred_w2c, row.gt_w2c)
            metrics = summarize_pi3_metrics(errs.rotation_deg, errs.translation_deg)
            write_prediction_sequence(args.output_dir, row.sequence_name, pred_c2w, row.gt_w2c, metrics)
            all_r.extend(errs.rotation_deg.tolist())
            all_t.extend(errs.translation_deg.tolist())
            print(f"[sequence {row_idx + 1}/{len(rows)}] {row.sequence_name}: ok {metrics}", flush=True)
        except Exception as exc:
            failures.append({"sequence": row.sequence_name, "error": str(exc)})
            print(f"[sequence {row_idx + 1}/{len(rows)}] {row.sequence_name}: failed {exc}", flush=True)
            traceback.print_exc()

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

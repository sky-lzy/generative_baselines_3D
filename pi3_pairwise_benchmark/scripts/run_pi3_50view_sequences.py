#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import load_manifest, write_json, write_prediction_sequence
from pi3_pairwise_benchmark.metrics import compute_pairwise_errors, invert_se3_stack, summarize_pi3_metrics


def _load_pi3(pi3_root: Path, pretrained_model_name_or_path: str, device: str):
    import sys

    if str(pi3_root) not in sys.path:
        sys.path.insert(0, str(pi3_root))
    model_mod = importlib.import_module("pi3.models.pi3")
    return model_mod.Pi3.from_pretrained(pretrained_model_name_or_path).to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pi3 once per sequence and save predicted cameras for Pi3 metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-name-or-path", default="yyfz233/Pi3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    if not args.pi3_root.is_dir():
        raise FileNotFoundError(f"Missing Pi3 checkout: {args.pi3_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest, require_images=True)
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]

    model = _load_pi3(args.pi3_root, args.pretrained_model_name_or_path, args.device)
    interfaces = importlib.import_module("utils.interfaces")
    namespace = argparse.Namespace(device=args.device, no_crop=False, load_img_size=512, verbose=False)

    all_r: list[float] = []
    all_t: list[float] = []
    failures: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            print(f"[sequence {idx}/{len(rows)}] {row.sequence_name}: Pi3 inference begin", flush=True)
            pred_c2w, pred_intrinsics = interfaces.infer_cameras_c2w(row.image_paths, model, namespace)
            pred_c2w = np.asarray(
                pred_c2w.detach().cpu().numpy() if hasattr(pred_c2w, "detach") else pred_c2w,
                dtype=np.float64,
            )
            if pred_c2w.shape != row.gt_w2c.shape:
                raise ValueError(f"Pi3 returned {pred_c2w.shape}, expected {row.gt_w2c.shape}")
            pred_intrinsics_np = None
            if pred_intrinsics is not None:
                pred_intrinsics_np = np.asarray(
                    pred_intrinsics.detach().cpu().numpy() if hasattr(pred_intrinsics, "detach") else pred_intrinsics
                )
            errors = compute_pairwise_errors(invert_se3_stack(pred_c2w), row.gt_w2c)
            metrics = summarize_pi3_metrics(errors.rotation_deg, errors.translation_deg)
            write_prediction_sequence(args.output_dir, row.sequence_name, pred_c2w, row.gt_w2c, metrics, pred_intrinsics_np)
            all_r.extend(errors.rotation_deg.tolist())
            all_t.extend(errors.translation_deg.tolist())
            print(f"[sequence {idx}/{len(rows)}] {row.sequence_name}: ok {metrics}", flush=True)
        except Exception as exc:
            failures.append({"sequence": row.sequence_name, "error": str(exc)})
            print(f"[sequence {idx}/{len(rows)}] {row.sequence_name}: failed {exc}", flush=True)

    report = {
        "metrics": summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t)) if all_r else {},
        "num_failures": len(failures),
        "failures": failures,
    }
    write_json(args.output_dir / "summary.json", report)
    print(report, flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

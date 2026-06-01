#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import load_manifest, write_json, write_prediction_pair
from pi3_pairwise_benchmark.metrics import compute_pairwise_errors, summarize_pi3_metrics
from pi3_pairwise_benchmark.sampling import select_pairs


def _load_pi3(pi3_root: Path, pretrained_model_name_or_path: str, device: str):
    import sys

    if str(pi3_root) not in sys.path:
        sys.path.insert(0, str(pi3_root))
    model_mod = importlib.import_module("pi3.models.pi3")
    return model_mod.Pi3.from_pretrained(pretrained_model_name_or_path).to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pi3 as a fair 2-view baseline over every pair in each 10-view sample.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-name-or-path", default="yyfz233/Pi3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--pair-policy", default="all", choices=["all", "short", "medium", "long", "mixed"])
    parser.add_argument("--pairs-per-sequence", type=int, default=None)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--short-gap-min", type=int, default=1)
    parser.add_argument("--short-gap-max", type=int, default=5)
    parser.add_argument("--medium-gap-min", type=int, default=10)
    parser.add_argument("--medium-gap-max", type=int, default=20)
    parser.add_argument("--long-gap-min", type=int, default=30)
    parser.add_argument("--long-gap-max", type=int, default=None)
    args = parser.parse_args()

    if not args.pi3_root.is_dir():
        raise FileNotFoundError(f"Missing Pi3 checkout: {args.pi3_root}")
    rows = load_manifest(args.manifest, require_images=True)
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]

    model = _load_pi3(args.pi3_root, args.pretrained_model_name_or_path, args.device)
    interfaces = importlib.import_module("utils.interfaces")
    all_r: list[float] = []
    all_t: list[float] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        pairs = select_pairs(
            num_frames=len(row.image_paths),
            policy=args.pair_policy,
            max_pairs=args.pairs_per_sequence,
            seed=args.pair_seed,
            sequence_key=row.sequence_name,
            short_gap_min=args.short_gap_min,
            short_gap_max=args.short_gap_max,
            medium_gap_min=args.medium_gap_min,
            medium_gap_max=args.medium_gap_max,
            long_gap_min=args.long_gap_min,
            long_gap_max=args.long_gap_max,
        )
        for pair in pairs:
            image_paths = [row.image_paths[pair[0]], row.image_paths[pair[1]]]
            try:
                pred_c2w, _ = interfaces.infer_cameras_c2w(
                    image_paths,
                    model,
                    argparse.Namespace(device=args.device, no_crop=False, load_img_size=512, verbose=False),
                )
                pred_c2w = np.asarray(pred_c2w.detach().cpu().numpy() if hasattr(pred_c2w, "detach") else pred_c2w, dtype=np.float64)
                gt_w2c = row.gt_w2c[list(pair)]
                errs = compute_pairwise_errors(np.linalg.inv(pred_c2w), gt_w2c)
                metrics = summarize_pi3_metrics(errs.rotation_deg, errs.translation_deg)
                write_prediction_pair(args.output_dir, row.sequence_name, pair, pred_c2w, gt_w2c, metrics)
                all_r.extend(errs.rotation_deg.tolist())
                all_t.extend(errs.translation_deg.tolist())
            except Exception as exc:
                failures.append({"sequence": row.sequence_name, "pair": f"{pair[0]},{pair[1]}", "error": str(exc)})

    report = {
        "metrics": summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t)) if all_r else {},
        "num_failures": len(failures),
        "failures": failures,
    }
    write_json(args.output_dir / "summary.json", report)
    print(report)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

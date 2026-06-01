from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_manifest, write_json
from .metrics import compute_pairwise_errors, pairwise_indices, summarize_pi3_metrics


def evaluate_pair_prediction_root(manifest_path: str | Path, prediction_root: str | Path) -> dict:
    rows = load_manifest(manifest_path)
    prediction_root = Path(prediction_root)
    all_r: list[float] = []
    all_t: list[float] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        seq_dir = prediction_root / row.sequence_name.replace("/", "_")
        for i, j in pairwise_indices(10):
            pair_dir = seq_dir / f"pair_{i:02d}_{j:02d}"
            pred_path = pair_dir / "pred_w2c.npy"
            if not pred_path.is_file():
                failures.append({"sequence": row.sequence_name, "pair": f"{i},{j}", "error": "missing pred_w2c.npy"})
                continue
            pred_w2c = np.load(pred_path)
            gt_w2c = row.gt_w2c[[i, j]]
            errors = compute_pairwise_errors(pred_w2c=pred_w2c, gt_w2c=gt_w2c)
            all_r.extend(errors.rotation_deg.tolist())
            all_t.extend(errors.translation_deg.tolist())

    metrics = summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t)) if all_r else {}
    return {"metrics": metrics, "num_failures": len(failures), "failures": failures}


def write_pair_prediction_report(manifest_path: str | Path, prediction_root: str | Path, output_path: str | Path) -> dict:
    report = evaluate_pair_prediction_root(manifest_path, prediction_root)
    write_json(output_path, report)
    return report


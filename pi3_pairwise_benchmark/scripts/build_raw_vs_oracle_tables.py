#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pi3_pairwise_benchmark.metrics import (
    compute_pairwise_errors,
    invert_se3_stack,
    pairwise_indices,
    summarize_pi3_metrics,
)


METRIC_KEYS = [
    "Racc_5",
    "Tacc_5",
    "Auc_5",
    "Racc_15",
    "Tacc_15",
    "Auc_15",
    "Racc_30",
    "Tacc_30",
    "Auc_30",
]


def _relative_translation_vectors(w2c: np.ndarray) -> list[np.ndarray | None]:
    vectors = []
    for i, j in pairwise_indices(len(w2c)):
        rel = w2c[j] @ np.linalg.inv(w2c[i])
        t = rel[:3, 3]
        norm = np.linalg.norm(t)
        if norm > 1e-12:
            vectors.append(t / norm)
        else:
            vectors.append(None)
    return vectors


def _apply_camera_basis(c2w: np.ndarray, basis: np.ndarray) -> np.ndarray:
    out = np.array(c2w, copy=True)
    out[:, :3, :3] = c2w[:, :3, :3] @ basis[None]
    return out


def _score(items: list[dict[str, object]]) -> dict[str, float]:
    all_r: list[float] = []
    all_t: list[float] = []
    for item in items:
        pred_c2w = item["pred_c2w"]  # type: ignore[assignment]
        gt_w2c = item["gt_w2c"]  # type: ignore[assignment]
        errors = compute_pairwise_errors(invert_se3_stack(pred_c2w), gt_w2c)
        all_r.extend(errors.rotation_deg.tolist())
        all_t.extend(errors.translation_deg.tolist())
    metrics = summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t))
    metrics["num_pairs"] = float(len(all_r))
    return metrics


def _align_sequence_items(items: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    pred_vectors = []
    gt_vectors = []
    for item in items:
        pred_c2w = item["pred_c2w"]  # type: ignore[assignment]
        gt_w2c = item["gt_w2c"]  # type: ignore[assignment]
        pred = _relative_translation_vectors(invert_se3_stack(pred_c2w))
        gt = _relative_translation_vectors(gt_w2c)
        if len(pred) != len(gt):
            raise ValueError(f"Vector count mismatch for {item['sequence']}: {len(pred)} vs {len(gt)}")
        for pred_vec, gt_vec in zip(pred, gt):
            if pred_vec is not None and gt_vec is not None:
                pred_vectors.append(pred_vec)
                gt_vectors.append(gt_vec)
    pred_all = np.asarray(pred_vectors, dtype=np.float64)
    gt_all = np.asarray(gt_vectors, dtype=np.float64)
    rotation, rmsd = Rotation.align_vectors(gt_all, pred_all)
    basis = rotation.as_matrix().T
    aligned = []
    for item in items:
        aligned.append(
            {
                "sequence": item["sequence"],
                "pred_c2w": _apply_camera_basis(item["pred_c2w"], basis),  # type: ignore[arg-type]
                "gt_w2c": item["gt_w2c"],
            }
        )
    info = {"rmsd": float(rmsd), "num_vectors": float(len(pred_all))}
    return aligned, info


def _oracle_per_sequence(items: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        grouped[str(item["sequence"])].append(item)
    aligned_all = []
    info = []
    for sequence in sorted(grouped):
        aligned, seq_info = _align_sequence_items(grouped[sequence])
        aligned_all.extend(aligned)
        info.append({"sequence": sequence, **seq_info})
    return aligned_all, info


def _load_stack_items(output_dir: Path) -> list[dict[str, object]]:
    items = []
    for pred_path in sorted(output_dir.glob("*/pred_c2w.npy")):
        sequence = pred_path.parent.name
        gt_path = pred_path.parent / "gt_w2c.npy"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        items.append(
            {
                "sequence": sequence,
                "pred_c2w": np.load(pred_path),
                "gt_w2c": np.load(gt_path),
            }
        )
    if not items:
        raise FileNotFoundError(f"No stack predictions found under {output_dir}")
    return items


def _load_pair_items(output_dir: Path) -> list[dict[str, object]]:
    items = []
    for pred_path in sorted(output_dir.glob("*/pair_*/pred_c2w.npy")):
        sequence = pred_path.parent.parent.name
        gt_path = pred_path.parent / "gt_w2c.npy"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        items.append(
            {
                "sequence": sequence,
                "pred_c2w": np.load(pred_path),
                "gt_w2c": np.load(gt_path),
            }
        )
    if not items:
        raise FileNotFoundError(f"No pair predictions found under {output_dir}")
    return items


def _summarize_method(name: str, setting: str, output_dir: Path, mode: str) -> dict[str, object]:
    items = _load_stack_items(output_dir) if mode == "stack" else _load_pair_items(output_dir)
    raw = _score(items)
    aligned_items, alignment_info = _oracle_per_sequence(items)
    aligned = _score(aligned_items)
    return {
        "method": name,
        "setting": setting,
        "mode": mode,
        "output_dir": str(output_dir),
        "num_sequences": len({str(item["sequence"]) for item in items}),
        "raw": raw,
        "oracle_per_sequence_camera_basis_so3": aligned,
        "alignment_info": alignment_info,
    }


def _format_table(rows: list[dict[str, object]], metric_key: str) -> str:
    header = "| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        metrics = row[metric_key]  # type: ignore[index]
        values = [
            str(row["setting"]),
            str(row["method"]),
            *[f"{metrics[key]:.2f}" for key in METRIC_KEYS],  # type: ignore[index]
            f"{metrics['num_pairs']:.0f}",  # type: ignore[index]
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build raw and per-sequence oracle Pi3 metric tables.")
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    specs = [
        ("Pi3", "50-view", args.base_output_dir / "pi3_50view_save", "stack"),
        ("Ours/VWM", "50-view", args.base_output_dir / "ours_50view_s40", "stack"),
        ("Geo4D", "50-view", args.base_output_dir / "geo4d_50view_s5", "stack"),
        ("Pi3", "2-view long", args.base_output_dir / "pi3_2view_long2_perseq", "pair"),
        ("Ours/VWM", "2-view long", args.base_output_dir / "ours_2view_long2_perseq_s40", "pair"),
        ("Geo4D", "2-view long", args.base_output_dir / "geo4d_2view_long2_perseq_s5", "pair"),
    ]
    rows = [_summarize_method(*spec) for spec in specs]
    report = {
        "base_output_dir": str(args.base_output_dir),
        "note": "Oracle rows apply a GT-assisted per-sequence SO3 camera-basis rotation fit from predicted relative translation directions to GT directions. This is an upper-bound diagnostic, not a fair benchmark.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = "\n".join(
        [
            "# Re10K Raw vs Oracle Tables",
            "",
            str(report["note"]),
            "",
            "## Raw Pi3 Metrics",
            "",
            _format_table(rows, "raw"),
            "",
            "## Oracle Per-Sequence Camera-Basis SO3",
            "",
            _format_table(rows, "oracle_per_sequence_camera_basis_so3"),
            "",
        ]
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

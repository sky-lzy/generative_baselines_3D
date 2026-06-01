#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "Geo4D"))

from dust3r.utils.vo_eval import eval_metrics  # noqa: E402
from pi3_pairwise_benchmark.io import load_manifest, safe_sequence_name  # noqa: E402


def _c2w_to_geo4d_traj(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape (N,4,4), got {c2w.shape}")
    xyz = c2w[:, :3, 3]
    xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
    wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    timestamps = np.arange(len(c2w), dtype=np.float64)
    return np.column_stack([xyz, wxyz]), timestamps


def _load_pred_c2w(output_dir: Path, sequence_name: str) -> np.ndarray:
    seq_dir = output_dir / safe_sequence_name(sequence_name)
    pred_path = seq_dir / "pred_c2w.npy"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing saved prediction: {pred_path}")
    return np.load(pred_path)


def _mean_nonzero(values: list[float]) -> float:
    filtered = [v for v in values if math.isfinite(v) and abs(v) > 0.0]
    return float(np.mean(filtered)) if filtered else float("nan")


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    summary = report["summary"]  # type: ignore[index]
    reference = report["geo4d_paper_reference_tum_dynamics"]  # type: ignore[index]
    lines = [
        "# Geo4D Official-Metric Check on Saved TUM Outputs",
        "",
        "This runs Geo4D's own `dust3r.utils.vo_eval.eval_metrics` on the saved predictions.",
        "That metric uses evo ATE/RPE with `align=True` and `correct_scale=True`, i.e. Umeyama/Sim(3)-aligned trajectory evaluation.",
        "",
        "## Our Saved TUM RGB-D Windows",
        "",
        f"- sequences: {summary['num_sequences']}",
        f"- frames per sequence: {summary['frames_per_sequence']}",
        f"- ATE: {summary['ate_mean']:.5f}",
        f"- RPE trans: {summary['rpe_trans_mean']:.5f}",
        f"- RPE rot: {summary['rpe_rot_mean']:.5f}",
        "",
        "## Geo4D Paper TUM-Dynamics Reference",
        "",
        f"- protocol: {reference['protocol']}",
        f"- ATE: {reference['ate']:.5f}",
        f"- RPE trans: {reference['rpe_trans']:.5f}",
        f"- RPE rot: {reference['rpe_rot']:.5f}",
        "",
        "## Interpretation",
        "",
        "These numbers are useful as an implementation sanity check, but they are not a strict paper reproduction because our saved run uses 50-frame windows sampled from our TUM RGB-D manifest, while Geo4D reports TUM-dynamics using the first 90 frames of each sequence with temporal stride 3.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved Geo4D predictions with Geo4D's official ATE/RPE metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, require_images=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "per_sequence_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_sequence: list[dict[str, object]] = []
    ate_values: list[float] = []
    rpe_trans_values: list[float] = []
    rpe_rot_values: list[float] = []
    frame_counts: list[int] = []

    for row in rows:
        pred_c2w = _load_pred_c2w(args.prediction_dir, row.sequence_name)
        n = len(row.image_paths)
        if pred_c2w.shape[0] < n:
            raise ValueError(f"{row.sequence_name}: prediction has {pred_c2w.shape[0]} frames, expected {n}")
        pred_traj = _c2w_to_geo4d_traj(pred_c2w[:n])
        gt_traj = _c2w_to_geo4d_traj(row.gt_c2w)
        metric_path = metrics_dir / f"{safe_sequence_name(row.sequence_name)}_eval_metric.txt"
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq=row.sequence_name,
            filename=str(metric_path),
            sample_stride=1,
        )
        record = {
            "sequence_name": row.sequence_name,
            "num_frames": n,
            "ate": float(ate),
            "rpe_trans": float(rpe_trans),
            "rpe_rot": float(rpe_rot),
            "metric_file": str(metric_path),
        }
        per_sequence.append(record)
        ate_values.append(float(ate))
        rpe_trans_values.append(float(rpe_trans))
        rpe_rot_values.append(float(rpe_rot))
        frame_counts.append(n)

    report: dict[str, object] = {
        "summary": {
            "num_sequences": len(per_sequence),
            "frames_per_sequence": sorted(set(frame_counts)),
            "ate_mean": _mean_nonzero(ate_values),
            "rpe_trans_mean": _mean_nonzero(rpe_trans_values),
            "rpe_rot_mean": _mean_nonzero(rpe_rot_values),
        },
        "geo4d_paper_reference_tum_dynamics": {
            "protocol": "TUM-dynamics, first 90 frames per sequence, temporal stride 3; Sim(3)/Umeyama aligned ATE/RPE.",
            "ate": 0.073,
            "rpe_trans": 0.020,
            "rpe_rot": 0.635,
            "source": "Geo4D arXiv v2 Table 2",
        },
        "per_sequence": per_sequence,
    }
    json_path = args.output_dir / "geo4d_official_metrics_summary.json"
    md_path = args.output_dir / "geo4d_official_metrics_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

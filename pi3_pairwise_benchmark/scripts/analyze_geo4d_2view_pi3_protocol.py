#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pi3_pairwise_benchmark.io import load_manifest, safe_sequence_name, write_json
from pi3_pairwise_benchmark.metrics import compute_pairwise_errors, invert_se3_stack, summarize_pi3_metrics
from pi3_pairwise_benchmark.sampling import select_pairs


def _read_geo4d_tum_wxyz(path: Path) -> np.ndarray:
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _, tx, ty, tz, qw, qx, qy, qz = [float(v) for v in line.split()]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return np.stack(poses)


def _geo4d_raw_key(sequence_name: str, padded_paths: list[str]) -> str:
    digest = hashlib.sha1("\n".join(padded_paths).encode("utf-8")).hexdigest()[:10]
    return f"{safe_sequence_name(sequence_name)}_{digest}"


def _load_pair_prediction(output_dir: Path, sequence_name: str, pair_paths: tuple[str, str]) -> np.ndarray:
    padded = [pair_paths[0]] * 8 + [pair_paths[1]] * 8
    raw_path = output_dir / "geo4d_raw" / "outputs" / _geo4d_raw_key(sequence_name, padded) / "pred_traj.txt"
    if raw_path.is_file():
        return _read_geo4d_tum_wxyz(raw_path)[[0, -1]]
    raise FileNotFoundError(f"Missing raw Geo4D pair trajectory: {raw_path}")


def _proper_signed_permutation_matrices() -> list[tuple[str, np.ndarray]]:
    out = []
    for perm in itertools.permutations(range(3)):
        permutation = np.zeros((3, 3), dtype=np.float64)
        permutation[np.arange(3), perm] = 1.0
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            basis = permutation @ np.diag(signs)
            if np.linalg.det(basis) > 0.0:
                name = "".join("xyz"[idx] for idx in perm) + "_" + "".join("p" if s > 0 else "n" for s in signs)
                out.append((name, basis))
    return out


def _apply_camera_basis(pair_c2w: np.ndarray, basis: np.ndarray) -> np.ndarray:
    out = np.array(pair_c2w, copy=True)
    out[:, :3, :3] = pair_c2w[:, :3, :3] @ basis[None]
    return out


def _relative_translation_vector(w2c_pair: np.ndarray) -> np.ndarray | None:
    rel = w2c_pair[1] @ np.linalg.inv(w2c_pair[0])
    t = rel[:3, 3]
    norm = np.linalg.norm(t)
    if norm < 1e-12:
        return None
    return t / norm


def _align_vectors(pred_pairs: list[np.ndarray], gt_pairs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    pred_vectors = []
    gt_vectors = []
    for pred_c2w, gt_w2c in zip(pred_pairs, gt_pairs):
        pred_v = _relative_translation_vector(invert_se3_stack(pred_c2w))
        gt_v = _relative_translation_vector(gt_w2c)
        if pred_v is None or gt_v is None:
            continue
        pred_vectors.append(pred_v)
        gt_vectors.append(gt_v)
    pred = np.asarray(pred_vectors, dtype=np.float64)
    gt = np.asarray(gt_vectors, dtype=np.float64)
    rotation, rmsd = Rotation.align_vectors(gt, pred)
    return rotation.as_matrix(), {"rmsd": float(rmsd), "num_vectors": float(len(pred))}


def _score(pred_pairs: list[np.ndarray], gt_pairs: list[np.ndarray]) -> dict[str, float]:
    all_r = []
    all_t = []
    for pred_c2w, gt_w2c in zip(pred_pairs, gt_pairs):
        errors = compute_pairwise_errors(invert_se3_stack(pred_c2w), gt_w2c)
        all_r.extend(errors.rotation_deg.tolist())
        all_t.extend(errors.translation_deg.tolist())
    return summarize_pi3_metrics(np.asarray(all_r), np.asarray(all_t))


def _format_metrics(metrics: dict[str, float]) -> str:
    keys = ["Racc_5", "Tacc_5", "Auc_5", "Racc_15", "Tacc_15", "Auc_15", "Racc_30", "Tacc_30", "Auc_30"]
    return " | ".join(f"{key}={metrics[key]:.2f}" for key in keys)


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    strict = report["strict_raw_pi3"]  # type: ignore[index]
    best_fixed = report["best_fixed_camera_basis"]  # type: ignore[index]
    oracle = report["gt_assisted"]  # type: ignore[index]
    lines = [
        "# Geo4D 2-View Pi3-Protocol Diagnostic",
        "",
        "This diagnostic uses saved Geo4D two-unique-image padded pair predictions and Pi3 relative-pose metrics.",
        "GT-assisted rows are explicitly unfair and should not be reported as the main benchmark.",
        "",
        "## Strict Pi3 Protocol",
        "",
        f"- raw corrected Geo4D: {_format_metrics(strict)}",  # type: ignore[arg-type]
        "",
        "## Fixed Convention Sweep",
        "",
        f"- best proper signed-permutation basis: `{best_fixed['name']}`",  # type: ignore[index]
        f"- metrics: {_format_metrics(best_fixed['metrics'])}",  # type: ignore[index,arg-type]
        f"- basis:\n\n```text\n{np.asarray(best_fixed['basis'])}\n```",  # type: ignore[index]
        "",
        "## GT-Assisted Diagnostics",
        "",
        f"- oracle global camera-basis SO3: {_format_metrics(oracle['oracle_global_camera_basis_so3']['metrics'])}",  # type: ignore[index,arg-type]
        f"- oracle per-sequence camera-basis SO3: {_format_metrics(oracle['oracle_per_sequence_camera_basis_so3']['metrics'])}",  # type: ignore[index,arg-type]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Geo4D 2-view outputs under Pi3 relative-pose metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geo4d-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-policy", default="long", choices=["all", "short", "medium", "long", "mixed"])
    parser.add_argument("--max-pairs-per-sequence", type=int, default=20)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--long-gap-min", type=int, default=30)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, require_images=False)
    pred_pairs_by_sequence: list[list[np.ndarray]] = []
    gt_pairs_by_sequence: list[list[np.ndarray]] = []
    pair_index: list[dict[str, object]] = []
    for row in rows:
        pairs = select_pairs(
            len(row.image_paths),
            policy=args.pair_policy,
            max_pairs=args.max_pairs_per_sequence,
            seed=args.pair_seed,
            sequence_key=row.sequence_name,
            long_gap_min=args.long_gap_min,
        )
        pred_seq = []
        gt_seq = []
        for pair in pairs:
            pred_seq.append(
                _load_pair_prediction(
                    args.geo4d_output_dir,
                    row.sequence_name,
                    (row.image_paths[pair[0]], row.image_paths[pair[1]]),
                )
            )
            gt_seq.append(row.gt_w2c[list(pair)])
            pair_index.append({"sequence": row.sequence_name, "pair": list(pair)})
        pred_pairs_by_sequence.append(pred_seq)
        gt_pairs_by_sequence.append(gt_seq)

    pred_pairs = [pair for seq in pred_pairs_by_sequence for pair in seq]
    gt_pairs = [pair for seq in gt_pairs_by_sequence for pair in seq]
    strict = _score(pred_pairs, gt_pairs)

    basis_reports = []
    for name, basis in _proper_signed_permutation_matrices():
        transformed = [_apply_camera_basis(pair, basis) for pair in pred_pairs]
        basis_reports.append({"name": name, "basis": basis.tolist(), "metrics": _score(transformed, gt_pairs)})
    basis_reports.sort(key=lambda item: item["metrics"]["Auc_30"], reverse=True)  # type: ignore[index]

    oracle_basis, oracle_info = _align_vectors(pred_pairs, gt_pairs)
    oracle_global = [_apply_camera_basis(pair, oracle_basis.T) for pair in pred_pairs]

    oracle_per_sequence = []
    per_sequence_info = []
    for pred_seq, gt_seq in zip(pred_pairs_by_sequence, gt_pairs_by_sequence):
        seq_basis, seq_info = _align_vectors(pred_seq, gt_seq)
        oracle_per_sequence.extend([_apply_camera_basis(pair, seq_basis.T) for pair in pred_seq])
        per_sequence_info.append({"basis": seq_basis.T.tolist(), "align_vectors": seq_info})

    report: dict[str, object] = {
        "manifest": str(args.manifest),
        "geo4d_output_dir": str(args.geo4d_output_dir),
        "pair_policy": args.pair_policy,
        "max_pairs_per_sequence": args.max_pairs_per_sequence,
        "num_pairs": len(pred_pairs),
        "pairs": pair_index,
        "strict_raw_pi3": strict,
        "best_fixed_camera_basis": basis_reports[0],
        "fixed_camera_basis_top5": basis_reports[:5],
        "gt_assisted": {
            "oracle_global_camera_basis_so3": {
                "basis": oracle_basis.T.tolist(),
                "align_vectors": oracle_info,
                "metrics": _score(oracle_global, gt_pairs),
            },
            "oracle_per_sequence_camera_basis_so3": {
                "per_sequence": per_sequence_info,
                "metrics": _score(oracle_per_sequence, gt_pairs),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "geo4d_2view_pi3_protocol_diagnostic.json", report)
    _write_markdown(args.output_dir / "geo4d_2view_pi3_protocol_diagnostic.md", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

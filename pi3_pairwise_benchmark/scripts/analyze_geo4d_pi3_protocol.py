#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pi3_pairwise_benchmark.io import load_manifest, write_json
from pi3_pairwise_benchmark.metrics import (
    compute_pairwise_errors,
    invert_se3_stack,
    pairwise_indices,
    summarize_pi3_metrics,
)


def _read_geo4d_tum_wxyz(path: Path) -> np.ndarray:
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"Expected TUM row with 8 values in {path}, got {len(parts)}")
        _, tx, ty, tz, qw, qx, qy, qz = [float(v) for v in parts]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return np.stack(poses)


def _load_geo4d_sequence(output_dir: Path, sequence_name: str, num_frames: int) -> np.ndarray:
    raw_root = output_dir / "geo4d_raw" / "outputs"
    raw_paths = sorted(raw_root.glob(f"{sequence_name}_*/pred_traj.txt"))
    if len(raw_paths) == 1:
        pred = _read_geo4d_tum_wxyz(raw_paths[0])
    elif len(raw_paths) > 1:
        raise ValueError(f"Multiple raw Geo4D trajectories found for {sequence_name}: {raw_paths}")
    else:
        pred = np.load(output_dir / sequence_name / "pred_c2w.npy")
    if pred.shape[0] < num_frames:
        raise ValueError(f"{sequence_name}: Geo4D has {pred.shape[0]} poses, expected {num_frames}")
    return pred[:num_frames]


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


def _apply_camera_basis(c2w: np.ndarray, basis: np.ndarray) -> np.ndarray:
    out = np.array(c2w, copy=True)
    out[:, :3, :3] = c2w[:, :3, :3] @ basis[None]
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching (N,3) points, got {src.shape} and {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / float(len(src))
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u @ vh))
    correction = np.diag([1.0, 1.0, sign])
    rotation = u @ correction @ vh
    if with_scale:
        variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
        scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    else:
        scale = 1.0
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _apply_world_similarity(c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.array(c2w, copy=True)
    out[:, :3, :3] = rotation[None] @ c2w[:, :3, :3]
    out[:, :3, 3] = (scale * (rotation @ c2w[:, :3, 3].T)).T + translation
    return out


def _scale_only_to_gt_centers(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    pred_centered = pred_centers - pred_centers.mean(axis=0)
    gt_centered = gt_centers - gt_centers.mean(axis=0)
    pred_rms = np.sqrt(np.mean(np.sum(pred_centered * pred_centered, axis=1)))
    gt_rms = np.sqrt(np.mean(np.sum(gt_centered * gt_centered, axis=1)))
    scale = 1.0 if pred_rms < 1e-12 else float(gt_rms / pred_rms)
    out = np.array(pred_c2w, copy=True)
    out[:, :3, 3] = (pred_centers - pred_centers.mean(axis=0)) * scale + gt_centers.mean(axis=0)
    return out


def _relative_translation_vectors(w2c: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    vectors = []
    for i, j in pairs:
        rel = w2c[j] @ np.linalg.inv(w2c[i])
        t = rel[:3, 3]
        norm = np.linalg.norm(t)
        if norm > 1e-12:
            vectors.append(t / norm)
    return np.asarray(vectors, dtype=np.float64)


def _align_pred_translation_vectors_to_gt(
    pred_c2w_list: list[np.ndarray],
    gt_w2c_list: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    pred_vectors = []
    gt_vectors = []
    for pred_c2w, gt_w2c in zip(pred_c2w_list, gt_w2c_list):
        pairs = pairwise_indices(len(pred_c2w))
        pred_w2c = invert_se3_stack(pred_c2w)
        pred_seq = _relative_translation_vectors(pred_w2c, pairs)
        gt_seq = _relative_translation_vectors(gt_w2c, pairs)
        if len(pred_seq) != len(gt_seq):
            raise ValueError("Internal relative vector mismatch")
        pred_vectors.append(pred_seq)
        gt_vectors.append(gt_seq)

    pred = np.concatenate(pred_vectors, axis=0)
    gt = np.concatenate(gt_vectors, axis=0)
    rotation, rmsd = Rotation.align_vectors(gt, pred)
    return rotation.as_matrix(), {"rmsd": float(rmsd), "num_vectors": float(len(pred))}


def _align_one_sequence_translation_vectors_to_gt(
    pred_c2w: np.ndarray,
    gt_w2c: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    pairs = pairwise_indices(len(pred_c2w))
    pred = _relative_translation_vectors(invert_se3_stack(pred_c2w), pairs)
    gt = _relative_translation_vectors(gt_w2c, pairs)
    rotation, rmsd = Rotation.align_vectors(gt, pred)
    return rotation.as_matrix(), {"rmsd": float(rmsd), "num_vectors": float(len(pred))}


def _score_sequences(pred_c2w_list: list[np.ndarray], gt_w2c_list: list[np.ndarray]) -> dict[str, float]:
    all_r = []
    all_t = []
    for pred_c2w, gt_w2c in zip(pred_c2w_list, gt_w2c_list):
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
    unfair = report["gt_assisted"]  # type: ignore[index]
    lines = [
        "# Geo4D Pi3-Protocol Diagnostic",
        "",
        "This diagnostic uses saved Geo4D predictions and Pi3 relative-pose metrics.",
        "The fixed camera-basis sweep is a convention check. GT-assisted rows are explicitly unfair and should not be reported as the main benchmark.",
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
        f"- scale-only center matching: {_format_metrics(unfair['scale_only'])}",  # type: ignore[index,arg-type]
        f"- SE3 center alignment: {_format_metrics(unfair['se3_center_align'])}",  # type: ignore[index,arg-type]
        f"- Sim3 center alignment: {_format_metrics(unfair['sim3_center_align'])}",  # type: ignore[index,arg-type]
        f"- oracle global camera-basis SO3: {_format_metrics(unfair['oracle_global_camera_basis_so3']['metrics'])}",  # type: ignore[index,arg-type]
        f"- oracle per-sequence camera-basis SO3: {_format_metrics(unfair['oracle_per_sequence_camera_basis_so3']['metrics'])}",  # type: ignore[index,arg-type]
        "",
        "Scale/SE3/Sim3 trajectory alignment is expected to leave Pi3 relative translation-angle scores effectively unchanged, because the metric is invariant to global world-frame similarity transforms.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Geo4D outputs under strict and diagnostic Pi3 protocols.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geo4d-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, require_images=False)
    pred_c2w_list = [
        _load_geo4d_sequence(args.geo4d_output_dir, row.sequence_name, len(row.image_paths))
        for row in rows
    ]
    gt_w2c_list = [row.gt_w2c for row in rows]
    gt_c2w_list = [row.gt_c2w for row in rows]

    strict = _score_sequences(pred_c2w_list, gt_w2c_list)

    basis_reports = []
    for name, basis in _proper_signed_permutation_matrices():
        transformed = [_apply_camera_basis(pred, basis) for pred in pred_c2w_list]
        basis_reports.append(
            {
                "name": name,
                "basis": basis.tolist(),
                "metrics": _score_sequences(transformed, gt_w2c_list),
            }
        )
    basis_reports.sort(key=lambda item: item["metrics"]["Auc_30"], reverse=True)  # type: ignore[index]

    scale_only = [
        _scale_only_to_gt_centers(pred, gt)
        for pred, gt in zip(pred_c2w_list, gt_c2w_list)
    ]
    se3_center_aligned = []
    sim3_center_aligned = []
    for pred, gt in zip(pred_c2w_list, gt_c2w_list):
        _, rotation, translation = _umeyama(pred[:, :3, 3], gt[:, :3, 3], with_scale=False)
        se3_center_aligned.append(_apply_world_similarity(pred, 1.0, rotation, translation))
        scale, rotation, translation = _umeyama(pred[:, :3, 3], gt[:, :3, 3], with_scale=True)
        sim3_center_aligned.append(_apply_world_similarity(pred, scale, rotation, translation))

    oracle_basis, oracle_info = _align_pred_translation_vectors_to_gt(pred_c2w_list, gt_w2c_list)
    oracle_transformed = [_apply_camera_basis(pred, oracle_basis.T) for pred in pred_c2w_list]
    per_sequence_oracle = []
    per_sequence_oracle_info = []
    for pred, gt_w2c in zip(pred_c2w_list, gt_w2c_list):
        seq_basis, seq_info = _align_one_sequence_translation_vectors_to_gt(pred, gt_w2c)
        per_sequence_oracle.append(_apply_camera_basis(pred, seq_basis.T))
        per_sequence_oracle_info.append({"basis": seq_basis.T.tolist(), "align_vectors": seq_info})

    report: dict[str, object] = {
        "manifest": str(args.manifest),
        "geo4d_output_dir": str(args.geo4d_output_dir),
        "num_sequences": len(rows),
        "strict_raw_pi3": strict,
        "best_fixed_camera_basis": basis_reports[0],
        "fixed_camera_basis_top5": basis_reports[:5],
        "gt_assisted": {
            "scale_only": _score_sequences(scale_only, gt_w2c_list),
            "se3_center_align": _score_sequences(se3_center_aligned, gt_w2c_list),
            "sim3_center_align": _score_sequences(sim3_center_aligned, gt_w2c_list),
            "oracle_global_camera_basis_so3": {
                "basis": oracle_basis.T.tolist(),
                "align_vectors": oracle_info,
                "metrics": _score_sequences(oracle_transformed, gt_w2c_list),
            },
            "oracle_per_sequence_camera_basis_so3": {
                "per_sequence": per_sequence_oracle_info,
                "metrics": _score_sequences(per_sequence_oracle, gt_w2c_list),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "geo4d_pi3_protocol_diagnostic.json", report)
    _write_markdown(args.output_dir / "geo4d_pi3_protocol_diagnostic.md", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

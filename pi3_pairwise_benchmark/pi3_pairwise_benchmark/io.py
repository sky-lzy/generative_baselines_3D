from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import invert_se3_stack


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    sequence_name: str
    image_ids: list[int]
    image_paths: list[str]
    gt_w2c: np.ndarray
    gt_c2w: np.ndarray

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ManifestRow":
        image_ids = list(payload["image_ids"])
        image_paths = list(payload["image_paths"])
        gt_w2c = np.asarray(payload["gt_w2c"], dtype=np.float64)
        gt_c2w = np.asarray(payload["gt_c2w"], dtype=np.float64)
        if len(image_ids) != len(image_paths):
            raise ValueError("Manifest rows must contain the same number of image ids and image paths")
        if len(image_ids) < 2:
            raise ValueError("Manifest rows must contain at least 2 image ids and image paths")
        expected_pose_shape = (len(image_ids), 4, 4)
        if gt_w2c.shape != expected_pose_shape or gt_c2w.shape != expected_pose_shape:
            raise ValueError(
                f"Manifest poses must be {expected_pose_shape}, got {gt_w2c.shape} and {gt_c2w.shape}"
            )
        return cls(
            dataset=str(payload.get("dataset", "Re10K")),
            sequence_name=str(payload["sequence_name"]),
            image_ids=[int(x) for x in image_ids],
            image_paths=[str(x) for x in image_paths],
            gt_w2c=gt_w2c,
            gt_c2w=gt_c2w,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sequence_name": self.sequence_name,
            "image_ids": self.image_ids,
            "image_paths": self.image_paths,
            "gt_w2c": self.gt_w2c.tolist(),
            "gt_c2w": self.gt_c2w.tolist(),
        }


def safe_sequence_name(sequence_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_name).strip("_") or "sequence"


def load_manifest(path: str | Path, require_images: bool = False) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    path = Path(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = ManifestRow.from_json(json.loads(line))
        except Exception as exc:
            raise ValueError(f"Invalid manifest row {line_no} in {path}: {exc}") from exc
        if require_images:
            missing = [p for p in row.image_paths if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"{row.sequence_name} has missing images: {missing[:3]}")
        rows.append(row)
    return rows


def write_manifest(path: str | Path, rows: list[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_prediction_pair(
    output_dir: str | Path,
    sequence_name: str,
    pair: tuple[int, int],
    pred_c2w: np.ndarray,
    gt_w2c: np.ndarray,
    metrics: dict[str, float],
    pred_intrinsics: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir)
    pair_dir = output_dir / safe_sequence_name(sequence_name) / f"pair_{pair[0]:02d}_{pair[1]:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_w2c = np.asarray(gt_w2c, dtype=np.float64)
    if pred_c2w.shape != (2, 4, 4):
        raise ValueError(f"pred_c2w must have shape (2,4,4), got {pred_c2w.shape}")
    if gt_w2c.shape != (2, 4, 4):
        raise ValueError(f"gt_w2c must have shape (2,4,4), got {gt_w2c.shape}")

    np.save(pair_dir / "pred_c2w.npy", pred_c2w)
    np.save(pair_dir / "pred_w2c.npy", invert_se3_stack(pred_c2w))
    np.save(pair_dir / "gt_w2c.npy", gt_w2c)
    if pred_intrinsics is not None:
        np.save(pair_dir / "pred_intrinsics.npy", np.asarray(pred_intrinsics))
    write_json(pair_dir / "metrics.json", metrics)
    return pair_dir


def write_prediction_sequence(
    output_dir: str | Path,
    sequence_name: str,
    pred_c2w: np.ndarray,
    gt_w2c: np.ndarray,
    metrics: dict[str, float],
    pred_intrinsics: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir)
    seq_dir = output_dir / safe_sequence_name(sequence_name)
    seq_dir.mkdir(parents=True, exist_ok=True)

    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_w2c = np.asarray(gt_w2c, dtype=np.float64)
    if pred_c2w.ndim != 3 or pred_c2w.shape[1:] != (4, 4):
        raise ValueError(f"pred_c2w must have shape (N,4,4), got {pred_c2w.shape}")
    if gt_w2c.shape != pred_c2w.shape:
        raise ValueError(f"gt_w2c must match pred_c2w shape {pred_c2w.shape}, got {gt_w2c.shape}")

    np.save(seq_dir / "pred_c2w.npy", pred_c2w)
    np.save(seq_dir / "pred_w2c.npy", invert_se3_stack(pred_c2w))
    np.save(seq_dir / "gt_w2c.npy", gt_w2c)
    if pred_intrinsics is not None:
        np.save(seq_dir / "pred_intrinsics.npy", np.asarray(pred_intrinsics))
    write_json(seq_dir / "metrics.json", metrics)
    return seq_dir

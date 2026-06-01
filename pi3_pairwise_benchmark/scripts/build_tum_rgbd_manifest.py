#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import ManifestRow, write_manifest


def _read_timestamp_file(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), parts[1]))
    return rows


def _read_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    poses: list[tuple[float, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        timestamp = float(parts[0])
        tx, ty, tz = (float(x) for x in parts[1:4])
        qx, qy, qz, qw = (float(x) for x in parts[4:8])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _quat_xyzw_to_rot(qx, qy, qz, qw)
        pose[:3, 3] = [tx, ty, tz]
        poses.append((timestamp, pose))
    return poses


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _associate_rgb_to_pose(
    scene_dir: Path,
    max_time_delta: float,
) -> list[tuple[Path, np.ndarray]]:
    rgb_rows = _read_timestamp_file(scene_dir / "rgb.txt")
    gt_rows = _read_groundtruth(scene_dir / "groundtruth.txt")
    if not rgb_rows or not gt_rows:
        return []
    gt_times = [row[0] for row in gt_rows]
    associated: list[tuple[Path, np.ndarray]] = []
    for rgb_time, rel_image_path in rgb_rows:
        idx = bisect.bisect_left(gt_times, rgb_time)
        candidates = []
        if idx < len(gt_rows):
            candidates.append(gt_rows[idx])
        if idx > 0:
            candidates.append(gt_rows[idx - 1])
        if not candidates:
            continue
        gt_time, c2w = min(candidates, key=lambda row: abs(row[0] - rgb_time))
        if abs(gt_time - rgb_time) <= max_time_delta:
            image_path = scene_dir / rel_image_path
            if image_path.is_file():
                associated.append((image_path.resolve(), c2w))
    return associated


def _candidate_windows(
    scene_dir: Path,
    associated: list[tuple[Path, np.ndarray]],
    num_frames: int,
    frame_step: int,
    window_stride: int,
) -> list[ManifestRow]:
    span = (num_frames - 1) * frame_step + 1
    rows: list[ManifestRow] = []
    for start in range(0, len(associated) - span + 1, window_stride):
        indices = [start + i * frame_step for i in range(num_frames)]
        image_paths = [str(associated[idx][0]) for idx in indices]
        gt_c2w = np.stack([associated[idx][1] for idx in indices], axis=0)
        rows.append(
            ManifestRow(
                dataset="TUM-RGBD",
                sequence_name=f"{scene_dir.name}_start{start:05d}_step{frame_step}",
                image_ids=list(range(num_frames)),
                image_paths=image_paths,
                gt_w2c=np.linalg.inv(gt_c2w),
                gt_c2w=gt_c2w,
            )
        )
    return rows


def _sample_evenly(rows: list[ManifestRow], max_sequences: int | None) -> list[ManifestRow]:
    if max_sequences is None or len(rows) <= max_sequences:
        return rows
    selected = np.linspace(0, len(rows) - 1, max_sequences, dtype=int)
    return [rows[int(idx)] for idx in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Pi3-style manifest from raw TUM RGB-D trajectories.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/n/netscratch/kempner_rcai_lab/Everyone/datasets/tum"),
    )
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--max-sequences", type=int, default=50)
    parser.add_argument("--max-time-delta", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[ManifestRow] = []
    for scene_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
        if not (scene_dir / "rgb.txt").is_file() or not (scene_dir / "groundtruth.txt").is_file():
            continue
        associated = _associate_rgb_to_pose(scene_dir, args.max_time_delta)
        rows.extend(
            _candidate_windows(
                scene_dir=scene_dir,
                associated=associated,
                num_frames=args.num_frames,
                frame_step=args.frame_step,
                window_stride=args.window_stride,
            )
        )
    rows = _sample_evenly(rows, args.max_sequences)
    if not rows:
        raise RuntimeError(f"No TUM RGB-D windows found under {args.data_root}")
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} TUM RGB-D rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

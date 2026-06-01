#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from pi3_pairwise_benchmark.io import ManifestRow, write_manifest


_CAM_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
}


def _read_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with path.open("rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack("<iI", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            if model_id not in _CAM_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = _CAM_MODELS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_colmap_images(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (image_id,) = struct.unpack("<I", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            (camera_id,) = struct.unpack("<I", f.read(4))
            name = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            f.seek(num_points2d * 24, 1)
            entries.append(
                {
                    "image_id": int(image_id),
                    "qvec": np.asarray(qvec, dtype=np.float64),
                    "tvec": np.asarray(tvec, dtype=np.float64),
                    "camera_id": int(camera_id),
                    "name": name.decode("utf-8"),
                }
            )
    return entries


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _frame_number(name: str) -> int:
    match = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", name)
    return int(match.group(1)) if match else -1


def _select_window(entries: list[dict[str, Any]], num_frames: int, window: str) -> list[dict[str, Any]]:
    if len(entries) < num_frames:
        raise ValueError(f"Need at least {num_frames} posed frames, got {len(entries)}")
    if window == "first":
        start = 0
    elif window == "center":
        start = (len(entries) - num_frames) // 2
    else:
        raise ValueError(f"Unsupported window policy: {window}")
    return entries[start : start + num_frames]


def _scene_rows(
    data_root: Path,
    image_subdir: str,
    num_frames: int,
    window: str,
    max_sequences: int | None,
) -> list[ManifestRow]:
    images_root = data_root / "images"
    rows: list[ManifestRow] = []
    for scene_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        gs_dir = scene_dir / scene / "gaussian_splat"
        sparse_dir = gs_dir / "sparse" / "0"
        image_dir = gs_dir / image_subdir
        if not image_dir.is_dir():
            continue
        if not (sparse_dir / "cameras.bin").is_file() or not (sparse_dir / "images.bin").is_file():
            continue

        _read_colmap_cameras(sparse_dir / "cameras.bin")
        entries = _read_colmap_images(sparse_dir / "images.bin")
        entries = [e for e in entries if (image_dir / e["name"]).is_file()]
        entries.sort(key=lambda e: (_frame_number(e["name"]), e["name"]))
        if len(entries) < num_frames:
            continue

        selected = _select_window(entries, num_frames=num_frames, window=window)
        gt_w2c = []
        for entry in selected:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = _qvec_to_rotmat(entry["qvec"])
            pose[:3, 3] = entry["tvec"]
            gt_w2c.append(pose)
        gt_w2c_np = np.stack(gt_w2c, axis=0)
        gt_c2w_np = np.linalg.inv(gt_w2c_np)
        rows.append(
            ManifestRow(
                dataset="DL3DV-Eval",
                sequence_name=scene,
                image_ids=[int(_frame_number(e["name"])) for e in selected],
                image_paths=[str(image_dir / e["name"]) for e in selected],
                gt_w2c=gt_w2c_np,
                gt_c2w=gt_c2w_np,
            )
        )
        if max_sequences is not None and len(rows) >= max_sequences:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Pi3-style manifest from DL3DV-Evaluation COLMAP poses.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/n/holylfs05/LABS/rcai_lab/Lab/dataset/data_downloads/dl3dv/DL3DV-Evaluation"),
    )
    parser.add_argument("--image-subdir", default="images_4")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--window", choices=("first", "center"), default="first")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _scene_rows(
        data_root=args.data_root,
        image_subdir=args.image_subdir,
        num_frames=args.num_frames,
        window=args.window,
        max_sequences=args.max_sequences,
    )
    if not rows:
        raise RuntimeError(f"No valid DL3DV-Eval rows found under {args.data_root}")
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} DL3DV-Eval rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

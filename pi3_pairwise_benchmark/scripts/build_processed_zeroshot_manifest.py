#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pi3_pairwise_benchmark.io import ManifestRow, write_manifest


def _extract_frames(video_path: Path, output_dir: Path, num_frames: int, overwrite: bool) -> list[str]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{idx:05d}.png" for idx in range(num_frames)]
    if not overwrite and all(path.is_file() for path in paths):
        return [str(path) for path in paths]

    count = 0
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx >= num_frames:
            break
        iio.imwrite(paths[idx], frame)
        count += 1
    if count < num_frames:
        raise ValueError(f"{video_path} has only {count} frames, need {num_frames}")
    return [str(path) for path in paths]


def _build_rows(
    data_root: Path,
    dataset: str,
    frames_root: Path,
    num_frames: int,
    max_sequences: int | None,
    overwrite_frames: bool,
) -> list[ManifestRow]:
    dataset_root = data_root / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing processed zero-shot dataset dir: {dataset_root}")

    rows: list[ManifestRow] = []
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        cameras_path = sample_dir / "gt_cameras.npz"
        rgb_path = sample_dir / "gt_rgb.mp4"
        if not cameras_path.is_file() or not rgb_path.is_file():
            continue
        cameras = np.load(cameras_path)
        gt_c2w = cameras["extrinsics"].astype(np.float64)
        if gt_c2w.shape[0] < num_frames:
            continue
        gt_c2w = gt_c2w[:num_frames]
        gt_w2c = np.linalg.inv(gt_c2w)
        image_paths = _extract_frames(
            video_path=rgb_path,
            output_dir=frames_root / dataset / sample_dir.name,
            num_frames=num_frames,
            overwrite=overwrite_frames,
        )
        rows.append(
            ManifestRow(
                dataset=dataset,
                sequence_name=f"{dataset}_{sample_dir.name}",
                image_ids=list(range(num_frames)),
                image_paths=image_paths,
                gt_w2c=gt_w2c,
                gt_c2w=gt_c2w,
            )
        )
        if max_sequences is not None and len(rows) >= max_sequences:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Pi3-style manifest from processed zero-shot datasets with gt_cameras.npz."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/n/netscratch/kempner_rcai_lab/Everyone/datasets/processed_zeroshot"),
    )
    parser.add_argument("--dataset", choices=("7scenes", "tum"), required=True)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite-frames", action="store_true")
    args = parser.parse_args()

    rows = _build_rows(
        data_root=args.data_root,
        dataset=args.dataset,
        frames_root=args.frames_root.resolve(),
        num_frames=args.num_frames,
        max_sequences=args.max_sequences,
        overwrite_frames=args.overwrite_frames,
    )
    if not rows:
        raise RuntimeError(f"No valid {args.dataset} rows found under {args.data_root}")
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} {args.dataset} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

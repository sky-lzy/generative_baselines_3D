#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

from pi3_pairwise_benchmark.io import write_json


def _youtube_id(url: str) -> str:
    marker = "/watch?v="
    if marker not in url:
        return url.strip().rsplit("/", 1)[-1]
    return url[url.find(marker) + len(marker) :].strip()


def _load_re10k_txt(path: Path) -> tuple[str, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    youtube_id = _youtube_id(lines[0])
    annos: list[dict] = []
    for idx, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) < 19:
            continue
        timestamp = int(parts[0])
        intrinsics = [float(x) for x in parts[1:7]]
        pose_3x4 = [float(x) for x in parts[7:19]]
        annos.append({"idx": idx, "timestamp": timestamp, "intrinsics": intrinsics, "pose": pose_3x4})
    return youtube_id, annos


def _load_video(video_path: Path) -> tuple[np.ndarray, float, int, int]:
    frames = iio.imread(str(video_path), index=None)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise OSError(f"Unexpected video shape for {video_path}: {frames.shape}")
    try:
        meta = iio.immeta(str(video_path))
        fps = float(meta.get("fps", 30.0) or 30.0)
    except Exception:
        # Some valid Re10K mp4s fail imageio's metadata reader even though
        # frame decoding succeeds. Re10K videos are 30 fps in this tree, and
        # this fallback lets us materialize the official sampled frames instead
        # of dropping a sequence from the reproduction.
        fps = 30.0
    height, width = frames.shape[1:3]
    return frames[..., :3], fps, width, height


def _frame_index(timestamp_us: int, fps: float) -> int:
    return int(round((timestamp_us / 1e6) * fps))


def _read_frame(frames: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx < 0 or frame_idx >= len(frames):
        raise OSError(f"Frame index {frame_idx} out of bounds for {len(frames)} frames")
    return np.asarray(frames[frame_idx], dtype=np.uint8)


def _pi3_annotation(seq: str, anno: dict, width: int, height: int) -> dict:
    fx, fy, cx, cy = anno["intrinsics"][:4]
    pose = list(anno["pose"])
    extrinsics = [pose[i : i + 4] for i in range(0, len(pose), 4)]
    extrinsics.append([0, 0, 0, 1])
    return {
        "idx": int(anno["idx"]),
        "filepath": str(Path(seq) / "images" / f"{int(anno['idx']):04d}.png"),
        "intrinsics": [
            [width * fx, 0, width * cx],
            [0, height * fy, height * cy],
            [0, 0, 1],
        ],
        "extrinsics": extrinsics,
    }


def prepare_sequence(
    source_root: Path,
    output_root: Path,
    split: str,
    seq: str,
    ids: list[int],
    overwrite_images: bool,
) -> dict:
    txt_path = source_root / "RealEstate10K" / split / f"{seq}.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing annotation txt: {txt_path}")
    youtube_id, annos = _load_re10k_txt(txt_path)
    if max(ids) >= len(annos):
        raise IndexError(f"{seq}: sampled id {max(ids)} >= num annos {len(annos)}")
    video_path = source_root / "videos" / f"{youtube_id}.mp4"
    frames, fps, width, height = _load_video(video_path)

    seq_dir = output_root / seq
    image_dir = seq_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations = [_pi3_annotation(seq, anno, width, height) for anno in annos]
    (seq_dir / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")

    written = 0
    for sample_id in ids:
        out_path = image_dir / f"{sample_id:04d}.png"
        if out_path.exists() and not overwrite_images:
            continue
        frame = _read_frame(frames, _frame_index(annos[sample_id]["timestamp"], fps))
        Image.fromarray(frame).save(out_path)
        written += 1
    return {"sequence": seq, "num_annotations": len(annos), "num_sampled": len(ids), "num_images_written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Pi3-format Re10K from video_world_model's mp4/txt Re10K tree.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seq-id-map", type=Path, required=True)
    parser.add_argument("--seq-file", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--allow-failures", action="store_true", help="Write a summary and return success even if some candidate sequences fail.")
    args = parser.parse_args()

    seq_id_map = json.loads(args.seq_id_map.read_text(encoding="utf-8"))
    if args.seq_file is not None:
        seqs = [line.strip() for line in args.seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        seqs = [seq for seq in seqs if seq in seq_id_map]
    else:
        seqs = list(seq_id_map.keys())
    if args.max_sequences is not None:
        seqs = seqs[: args.max_sequences]

    args.output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    failures: list[dict[str, str]] = []
    for seq in tqdm(seqs, desc="prepare Re10K Pi3 format"):
        try:
            reports.append(
                prepare_sequence(
                    source_root=args.source_root,
                    output_root=args.output_root,
                    split=args.split,
                    seq=seq,
                    ids=[int(x) for x in seq_id_map[seq]],
                    overwrite_images=args.overwrite_images,
                )
            )
        except Exception as exc:
            failures.append({"sequence": seq, "error": str(exc)})

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "seq_id_map": str(args.seq_id_map),
        "seq_file": str(args.seq_file) if args.seq_file else None,
        "num_sequences_requested": len(seqs),
        "num_sequences_prepared": len(reports),
        "num_failures": len(failures),
        "reports": reports,
        "failures": failures,
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.allow_failures or not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

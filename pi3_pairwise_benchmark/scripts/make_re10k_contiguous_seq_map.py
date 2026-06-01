#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_re10k_frames(txt_path: Path) -> int:
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing Re10K txt: {txt_path}")
    return max(0, len(txt_path.read_text(encoding="utf-8").splitlines()) - 1)


def _window_start(num_available: int, num_frames: int, window: str) -> int:
    if num_available < num_frames:
        raise ValueError(f"Need {num_frames} frames, got {num_available}")
    if window == "start":
        return 0
    if window == "center":
        return (num_available - num_frames) // 2
    raise ValueError(f"Unknown window policy: {window}")


def build_contiguous_seq_map(
    source_root: Path,
    sequence_names: list[str],
    split: str = "test",
    num_frames: int = 50,
    max_sequences: int | None = None,
    window: str = "center",
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for seq in sequence_names:
        txt_path = source_root / "RealEstate10K" / split / f"{seq}.txt"
        try:
            num_available = _count_re10k_frames(txt_path)
            start = _window_start(num_available=num_available, num_frames=num_frames, window=window)
        except Exception:
            continue
        out[seq] = list(range(start, start + num_frames))
        if max_sequences is not None and len(out) >= max_sequences:
            break
    return out


def _load_sequence_names(seq_file: Path | None, seq_id_map: Path | None) -> list[str]:
    if seq_file is not None:
        return [line.strip() for line in seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if seq_id_map is not None:
        payload = json.loads(seq_id_map.read_text(encoding="utf-8"))
        return list(payload.keys())
    raise ValueError("Either --seq-file or --seq-id-map must be provided")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Pi3-compatible contiguous-frame Re10K seq-id-map.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seq-file", type=Path, default=None)
    parser.add_argument("--seq-id-map", type=Path, default=None)
    parser.add_argument("--output-seq-id-map", type=Path, required=True)
    parser.add_argument("--output-seq-file", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--window", choices=["start", "center"], default="center")
    args = parser.parse_args()

    sequence_names = _load_sequence_names(args.seq_file, args.seq_id_map)
    seq_map = build_contiguous_seq_map(
        source_root=args.source_root,
        sequence_names=sequence_names,
        split=args.split,
        num_frames=args.num_frames,
        max_sequences=args.max_sequences,
        window=args.window,
    )
    args.output_seq_id_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_id_map.write_text(json.dumps(seq_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_seq_file.write_text("\n".join(seq_map.keys()) + ("\n" if seq_map else ""), encoding="utf-8")
    print(f"Wrote {len(seq_map)} sequences with {args.num_frames} contiguous frames to {args.output_seq_id_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

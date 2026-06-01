#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import ManifestRow, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Pi3 Re10K 10-frame samples into benchmark manifest JSONL.")
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--re10k-root", type=Path, required=True)
    parser.add_argument("--seq-id-map", type=Path, required=True)
    parser.add_argument("--seq-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seq_map = json.loads(args.seq_id_map.read_text(encoding="utf-8"))
    if args.seq_file is not None:
        seqs = [line.strip() for line in args.seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        items = [(seq, seq_map[seq]) for seq in seqs if seq in seq_map]
    else:
        items = list(seq_map.items())
    if args.max_sequences is not None:
        items = items[: args.max_sequences]

    rows: list[ManifestRow] = []
    for seq_name, ids in items:
        ann_path = args.re10k_root / seq_name / "annotations.json"
        annotations = json.loads(ann_path.read_text(encoding="utf-8"))
        selected = [annotations[int(idx)] for idx in ids]
        image_paths = [str(args.re10k_root / item["filepath"]) for item in selected]
        gt_w2c = np.asarray([item["extrinsics"] for item in selected], dtype=np.float64)
        gt_c2w = np.linalg.inv(gt_w2c)
        rows.append(
            ManifestRow(
                dataset="Re10K",
                sequence_name=seq_name,
                image_ids=[int(x) for x in ids],
                image_paths=image_paths,
                gt_w2c=gt_w2c,
                gt_c2w=gt_c2w,
            )
        )
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

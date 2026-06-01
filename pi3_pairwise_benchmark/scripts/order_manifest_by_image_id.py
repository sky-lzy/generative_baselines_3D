#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pi3_pairwise_benchmark.io import ManifestRow, load_manifest, write_manifest


def _sort_row(row: ManifestRow) -> ManifestRow:
    order = sorted(range(len(row.image_ids)), key=lambda idx: row.image_ids[idx])
    return ManifestRow(
        dataset=row.dataset,
        sequence_name=row.sequence_name,
        image_ids=[row.image_ids[idx] for idx in order],
        image_paths=[row.image_paths[idx] for idx in order],
        gt_w2c=row.gt_w2c[order],
        gt_c2w=row.gt_c2w[order],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sort each manifest row by image/frame id.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    rows = [_sort_row(row) for row in load_manifest(args.input, require_images=True)]
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} ordered rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

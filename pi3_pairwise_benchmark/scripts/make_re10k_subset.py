#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create matching subset seq-id map and seq-file for Pi3 smoke runs.")
    parser.add_argument("--seq-id-map", type=Path, required=True)
    parser.add_argument("--seq-file", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--output-seq-id-map", type=Path, required=True)
    parser.add_argument("--output-seq-file", type=Path, required=True)
    args = parser.parse_args()

    seq_id_map = json.loads(args.seq_id_map.read_text(encoding="utf-8"))
    seqs = [line.strip() for line in args.seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    subset = [seq for seq in seqs if seq in seq_id_map][: args.num_sequences]
    subset_map = {seq: seq_id_map[seq] for seq in subset}
    args.output_seq_id_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_id_map.write_text(json.dumps(subset_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_seq_file.write_text("\n".join(subset) + "\n", encoding="utf-8")
    print(f"Wrote {len(subset)} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


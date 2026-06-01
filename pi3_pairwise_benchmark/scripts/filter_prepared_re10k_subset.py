#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_prepared(root: Path, seq: str, ids: list[int]) -> bool:
    seq_dir = root / seq
    if not (seq_dir / "annotations.json").is_file():
        return False
    image_dir = seq_dir / "images"
    return all((image_dir / f"{int(sample_id):04d}.png").is_file() for sample_id in ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a fixed-size subset from successfully prepared Re10K sequences.")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--candidate-seq-id-map", type=Path, required=True)
    parser.add_argument("--candidate-seq-file", type=Path, required=True)
    parser.add_argument("--prepare-summary", type=Path, default=None)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--output-seq-id-map", type=Path, required=True)
    parser.add_argument("--output-seq-file", type=Path, required=True)
    args = parser.parse_args()

    seq_id_map = json.loads(args.candidate_seq_id_map.read_text(encoding="utf-8"))
    candidates = [line.strip() for line in args.candidate_seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful_sequences = None
    if args.prepare_summary is not None:
        summary = json.loads(args.prepare_summary.read_text(encoding="utf-8"))
        successful_sequences = {str(report["sequence"]) for report in summary.get("reports", [])}
    selected = []
    skipped = []
    for seq in candidates:
        ids = [int(x) for x in seq_id_map.get(seq, [])]
        if successful_sequences is not None and seq not in successful_sequences:
            skipped.append(seq)
        elif ids and _is_prepared(args.prepared_root, seq, ids):
            selected.append(seq)
        else:
            skipped.append(seq)
        if len(selected) >= args.num_sequences:
            break

    if len(selected) < args.num_sequences:
        raise RuntimeError(f"Only found {len(selected)} prepared sequences, requested {args.num_sequences}; skipped={skipped}")

    subset_map = {seq: seq_id_map[seq] for seq in selected}
    args.output_seq_id_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq_id_map.write_text(json.dumps(subset_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_seq_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(json.dumps({"num_selected": len(selected), "selected": selected, "skipped_before_selection": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pi3_pairwise_benchmark.io import write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Pi3 Re10K relpose inputs.")
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--re10k-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seq_map = args.pi3_root / "datasets" / "seq-id-maps" / f"Re10K_relpose_seq-id-map_seed{args.seed}.json"
    seq_file = args.pi3_root / "datasets" / "sequences" / "re10k_test_1719.txt"
    if not args.pi3_root.is_dir():
        raise FileNotFoundError(f"Missing Pi3 checkout: {args.pi3_root}")
    if not args.re10k_root.exists():
        raise FileNotFoundError(f"Missing Re10K root: {args.re10k_root}")
    if not seq_map.is_file():
        raise FileNotFoundError(f"Missing Pi3 seq-id map: {seq_map}")

    payload = {
        "pi3_root": str(args.pi3_root.resolve()),
        "re10k_root": str(args.re10k_root.resolve()),
        "seed": args.seed,
        "seq_id_map": str(seq_map.resolve()),
        "seq_id_map_sha256": sha256(seq_map),
        "num_sequences": len(json.loads(seq_map.read_text(encoding="utf-8"))),
        "seq_file": str(seq_file.resolve()) if seq_file.is_file() else None,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


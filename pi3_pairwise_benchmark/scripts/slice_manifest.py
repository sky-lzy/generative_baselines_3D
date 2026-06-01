#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a contiguous slice of a JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="0-based inclusive row start")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    rows = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    selected = rows[args.start :] if args.count is None else rows[args.start : args.start + args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"Wrote {len(selected)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

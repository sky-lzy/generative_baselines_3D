#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict:
    return payload.get("metrics", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pi3-10view, Pi3-2view, and ours-2view reports.")
    parser.add_argument("--pi3-10view", type=Path, required=True)
    parser.add_argument("--pi3-2view", type=Path, required=True)
    parser.add_argument("--ours-2view", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for name, path in (
        ("pi3_10view", args.pi3_10view),
        ("pi3_2view", args.pi3_2view),
        ("ours_2view", args.ours_2view),
    ):
        payload = _metrics(_load(path))
        rows.append({"method": name, **{k: payload.get(k, "") for k in ("Racc_5", "Tacc_5", "Auc_5", "Racc_15", "Tacc_15", "Auc_15", "Racc_30", "Tacc_30", "Auc_30", "num_pairs")}})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone Pi3 evaluation branch into external/Pi3.")
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    args = parser.parse_args()
    if args.target.exists():
        print(f"{args.target} already exists")
        return 0
    args.target.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call([
        "git",
        "clone",
        "--branch",
        "evaluation",
        "https://github.com/yyfz/Pi3.git",
        str(args.target),
    ])


if __name__ == "__main__":
    raise SystemExit(main())


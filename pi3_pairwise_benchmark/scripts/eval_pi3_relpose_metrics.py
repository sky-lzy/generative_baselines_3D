#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.metrics import evaluate_w2c_stack


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Pi3-style relative pose metrics from W2C pose stacks.")
    parser.add_argument("--pred-w2c", type=Path, required=True)
    parser.add_argument("--gt-w2c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pred = np.load(args.pred_w2c)
    gt = np.load(args.gt_w2c)
    metrics = evaluate_w2c_stack(pred, gt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


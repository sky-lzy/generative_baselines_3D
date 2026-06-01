#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path


def mark(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-by-stage Pi3 load diagnostic for Slurm GPU jobs.")
    parser.add_argument("--pi3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

    mark("start")
    mark(f"python={sys.executable}")
    sys.path.insert(0, str(args.pi3_root))

    mark("import torch begin")
    import torch

    mark(f"import torch done version={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        mark(f"cuda device={torch.cuda.get_device_name(0)}")

    mark("import Pi3 begin")
    from pi3.models.pi3 import Pi3

    mark("import Pi3 done")
    mark(f"from_pretrained begin checkpoint={args.checkpoint}")
    model = Pi3.from_pretrained(str(args.checkpoint))
    mark("from_pretrained done")
    mark(f"to({args.device}) begin")
    model = model.to(args.device).eval()
    mark("to/eval done")
    del model
    torch.cuda.empty_cache()
    mark("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

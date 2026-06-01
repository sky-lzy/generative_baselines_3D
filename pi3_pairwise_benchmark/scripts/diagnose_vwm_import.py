#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import importlib
import os
import sys
import time
from pathlib import Path


def mark(message: str) -> None:
    print(f"[diagnose_vwm_import] {message}", flush=True)


def import_step(name: str) -> None:
    start = time.monotonic()
    mark(f"import {name} begin")
    importlib.import_module(name)
    mark(f"import {name} done elapsed={time.monotonic() - start:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged import diagnostic for video_world_model Wan adapter.")
    parser.add_argument("--video-world-model-root", type=Path, required=True)
    parser.add_argument("--dump-after", type=int, default=60)
    args = parser.parse_args()

    root = args.video_world_model_root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, repeat=True)

    mark(f"root={root}")
    mark(f"python={sys.executable}")
    mark(f"cwd={Path.cwd()}")

    modules = [
        "torch",
        "torchvision.transforms",
        "wandb",
        "zmq",
        "msgpack",
        "transformers",
        "algorithms.common.base_pytorch_algo",
        "algorithms.wan.modules.model_ray_depth_mot",
        "algorithms.wan.modules.model",
        "algorithms.wan.modules.t5",
        "algorithms.wan.modules.tokenizers",
        "algorithms.wan.modules.vae",
        "algorithms.wan.modules.vae_v2",
        "algorithms.wan.utils.fm_solvers",
        "algorithms.wan.utils.fm_solvers_unipc",
        "datasets.re10k",
        "algorithms.wan.wan_t2v_ray_depth_mot",
    ]
    for module in modules:
        import_step(module)

    mark("all imports completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

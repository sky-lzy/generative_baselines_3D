#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official Pi3 10-view Re10K relpose reproduction.")
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--re10k-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-name-or-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args, extra = parser.parse_known_args()

    eval_script = args.pi3_root / "relpose" / "eval_angle.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"Missing Pi3 evaluation checkout: {eval_script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.pi3_root}:{env.get('PYTHONPATH', '')}"
    cmd = [
        "python",
        str(eval_script),
        f"output_dir={args.output_dir}",
        f"data.Re10K.cfg.Re10K_DIR={args.re10k_dir}",
        f"seed={args.seed}",
        f"device={args.device}",
        "eval_datasets=[Re10K]",
        "hydra/hydra_logging=default",
        "hydra/job_logging=default",
        *extra,
    ]
    if args.pretrained_model_name_or_path:
        cmd.append(f"pi3.pretrained_model_name_or_path={args.pretrained_model_name_or_path}")
    print(" ".join(str(x) for x in cmd), flush=True)
    return subprocess.call(cmd, cwd=args.pi3_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fan the scoring stage out to GPUs, one job per cell.

Scoring is not free: LPIPS over 36 cells x 128 scenes x up to 49 frames is ~220k forward passes.
On a login node that is hours and would also be antisocial; as one short GPU job per cell it is
minutes. Uses the same evaluation/score_nvs_fair.py the tuning stage used — one scorer, one code
path for both methods, which is the property the whole benchmark rests on.

COMPLETENESS is enforced, not hoped for. `--require_complete` makes the scorer exit non-zero unless
every scene in the set produced a row, so a partial cell fails loudly instead of publishing an
average over whichever scenes happened to finish. A cell that is still generating is skipped with a
count, and re-running this script tops it up.

    python3 nvs/slurm/submit_score.py                 # score every complete cell
    python3 nvs/slurm/submit_score.py --min_frac 1.0  # only cells that are 100% done
    python3 nvs/slurm/submit_score.py --local         # run inline instead of via sbatch
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS))
import run_nvs_fair as R  # noqa: E402
from submit_fair import ACCOUNT_FOR  # noqa: E402

SCORER = NVS / "evaluation" / "score_nvs_fair.py"
# The trailing (?P<suffix>...) group is REQUIRED for re-run passes: a cell named
# F_5b__re10k128_4dim__ncf1__s0.5__hg0.5 does not match a pattern anchored at the scale, so an
# entire A/B pass would generate thousands of scenes and then score exactly nothing.
CELL = re.compile(r"^(?P<method>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<ncf>\d)"
                  r"__s(?P<scale>[\d.]+)(?P<suffix>__[A-Za-z0-9_.]+)?$")

SCRIPT = """#!/bin/bash
#SBATCH --job-name=fscore_{name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx6000pro|h100|h200|a100"
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output={logdir}/score.{name}.%j.out
set -uo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME={hf_home}
echo "=== $(date '+%F %T') host=$(hostname) job=$SLURM_JOB_ID"
{cmd}
rc=$?
echo "=== $(date '+%F %T') EXIT $rc"
exit $rc
"""


def seva_workdir(cfg, cell):
    """SEVA writes into its own work_dirs; the preds_fair symlink only appears when a shard ends."""
    return Path(cfg.paths.seva_repo) / "work_dirs" / "demo" / "img2img" / f"fair_{cell}"


def done_count(cfg, cell, method):
    if method == "seva":
        d = Path(cfg.preds.fair) / cell
        if not d.exists():
            d = seva_workdir(cfg, cell)
        if not d.exists():
            return 0, d
        n = len({p.parent.parent.name for p in d.glob("*/samples-rgb/*.png")})
        return n, d
    d = Path(cfg.preds.fair) / cell
    return (len(list(d.glob("sample_*/rgb_metrics.json"))) if d.exists() else 0), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue")
    ap.add_argument("--min_frac", type=float, default=0.999,
                    help="score a cell only when this fraction of scenes is generated")
    ap.add_argument("--metric_size", type=int, default=256)
    ap.add_argument("--local", action="store_true", help="run scorers inline (no sbatch)")
    ap.add_argument("--force", action="store_true", help="rescore cells that already have a CSV")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="run_nvs_fair cfg overrides, e.g. preds.fair=<dir> "
                         "results.fair=<dir> run.cell_suffix=__hg0.5")
    a = ap.parse_args()

    sys.argv = [sys.argv[0]] + list(a.overrides)
    cfg = R.load_cfg()
    logs = NVS / "slurm" / "logs"; logs.mkdir(parents=True, exist_ok=True)
    cmds = NVS / "slurm" / "cmds"; cmds.mkdir(parents=True, exist_ok=True)
    res = Path(cfg.results.fair); res.mkdir(parents=True, exist_ok=True)

    # every cell that exists on disk, from either engine's output location
    cells = set()
    p = Path(cfg.preds.fair)
    if p.exists():
        cells |= {d.name for d in p.iterdir() if d.is_dir() and CELL.match(d.name)}
    sw = Path(cfg.paths.seva_repo) / "work_dirs" / "demo" / "img2img"
    if sw.exists():
        cells |= {d.name[5:] for d in sw.iterdir()
                  if d.is_dir() and d.name.startswith("fair_") and CELL.match(d.name[5:])}

    # A scoring job takes longer than the driver's poll interval, so "no CSV yet" is NOT the same
    # as "not submitted". Without this the driver piles up duplicate scorers writing the same CSV.
    inflight = set()
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                       capture_output=True, text=True)
    if q.returncode == 0:
        inflight = {ln[len("fscore_"):] for ln in q.stdout.split() if ln.startswith("fscore_")}

    # Honour the method selection and the cell suffix. submit_score discovers cells from DISK, so
    # with results.fair redirected to a re-run tree it happily re-scored the MAIN pass's SEVA cells
    # into that tree -- 12 wasted GPU jobs and a progress counter that no longer meant anything.
    om = cfg.run.get("only_method")
    want = ([str(om)] if om else [t for t, on in cfg.select.methods.items() if on])
    suffix = str(cfg.run.get("cell_suffix") or "")

    parts = [x.strip() for x in a.partition.split(",") if x.strip()]
    todo, skipped = [], []
    for cell in sorted(cells):
        g = CELL.match(cell).groupdict()
        if g["method"] not in want:
            skipped.append((cell, f"method {g['method']} not selected")); continue
        if suffix and not cell.endswith(suffix):
            skipped.append((cell, f"cell suffix != {suffix}")); continue
        method = "seva" if g["method"] == "seva" else "ours"
        scenes_root = Path(cfg.paths.scenes_fair) / g["ds"]
        total = len([d for d in os.listdir(scenes_root)
                     if (scenes_root / d).is_dir()])
        n, run_dir = done_count(cfg, cell, method)
        csv_p = res / f"{cell}.csv"
        if csv_p.exists() and not a.force:
            skipped.append((cell, "csv exists")); continue
        if cell in inflight and not a.force:
            skipped.append((cell, "scoring job already queued/running")); continue
        if n < a.min_frac * total:
            skipped.append((cell, f"{n}/{total} generated")); continue
        cmd = [sys.executable, str(SCORER), "--run_dir", str(run_dir), "--method", method,
               "--scenes_root", str(scenes_root), "--split", g["ncf"],
               "--metric_size", str(a.metric_size), "--out_csv", str(csv_p),
               "--require_complete"]
        if method == "ours":
            cmd.append("--selftest")
        todo.append((cell, cmd))

    print(f"score: {len(todo)} cell(s) ready, {len(skipped)} skipped")
    for c, why in skipped[:40]:
        print(f"  skip {c:<52} {why}")

    for i, (cell, cmd) in enumerate(todo):
        if a.dry_run:
            print(f"DRY {cell}\n    {' '.join(cmd)}"); continue
        if a.local:
            env = dict(os.environ, HF_HOME=cfg.paths.hf_home)
            r = subprocess.run(cmd, env=env)
            print(f"{'PASS' if r.returncode == 0 else 'FAIL'} {cell}")
            continue
        part = parts[i % len(parts)]
        sh = cmds / f"score.{cell}.sbatch"
        sh.write_text(SCRIPT.format(name=cell, partition=part, account=ACCOUNT_FOR(part),
                                    logdir=logs, hf_home=cfg.paths.hf_home,
                                    cmd=" ".join(shlex.quote(x) for x in cmd)))
        r = subprocess.run(["sbatch", "--parsable", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL submit {cell}: {r.stderr.strip()}")
        else:
            print(f"  {r.stdout.strip().split(';')[0]}  score {cell}")


if __name__ == "__main__":
    main()

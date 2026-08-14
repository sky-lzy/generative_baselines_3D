#!/usr/bin/env python3
"""Self-healing top-up: resubmit shards for any cell that is short of scenes and has no live jobs.

On the requeue partitions jobs get preempted constantly (87 preemptions in the first two hours).
--requeue plus per-scene --resume handles almost all of it automatically, but three things still
leave a cell permanently short:

  * a job that exits non-zero for a real reason (a node with a busy GPU, an OOM, a bad node)
  * a job preempted so late that SLURM stops requeueing it
  * a shard that hit its wall clock with scenes still to go

None of those announce themselves in the final table -- the cell just quietly has 119 of 128 scenes,
and --require_complete then blocks its CSV forever. This pass closes that loop: enumerate the cells
the CONFIG says should exist, count what is on disk, and resubmit the shards of any cell that is
short and idle. Resubmission is safe and cheap because both engines skip completed scenes.

    python3 nvs/slurm/topup_fair.py --dry_run
    python3 nvs/slurm/topup_fair.py
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS))
import run_nvs_fair as R  # noqa: E402
import submit_fair as SF  # noqa: E402


def live_cells():
    """cell names that currently have an inference job queued or running"""
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                       capture_output=True, text=True)
    out = set()
    for ln in q.stdout.split():
        if ln.startswith("fnvs_"):
            out.add(ln[len("fnvs_"):].rsplit("__sh", 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_jobs", type=int, default=200, help="cap on shards resubmitted per pass")
    a = ap.parse_args()

    sys.argv = [sys.argv[0]]
    cfg = R.load_cfg()
    live = live_cells()
    parts = [x.strip() for x in a.partition.split(",") if x.strip()]
    cmds = NVS / "slurm" / "cmds"; cmds.mkdir(parents=True, exist_ok=True)
    logs = NVS / "slurm" / "logs"; logs.mkdir(parents=True, exist_ok=True)

    jobs = SF.build_jobs(cfg, [p for p, _, _ in SF.PHASES], None)
    by_cell = {}
    for j in jobs:
        by_cell.setdefault(j["cell"], []).append(j)

    n_sub, short = 0, []
    for cell, js in sorted(by_cell.items()):
        ds = cell.split("__")[1]
        method = "seva" if cell.startswith("seva__") else "ours"
        sroot = Path(cfg.paths.scenes_fair) / ds
        total = len([d for d in os.listdir(sroot) if (sroot / d).is_dir()])
        if method == "seva":
            d = Path(cfg.preds.fair) / cell
            if not d.exists():
                d = Path(cfg.paths.seva_repo) / "work_dirs/demo/img2img" / f"fair_{cell}"
            done = len({p.parent.parent.name for p in d.glob("*/samples-rgb/*.png")}) if d.exists() else 0
        else:
            d = Path(cfg.preds.fair) / cell
            done = len(list(d.glob("sample_*/rgb_metrics.json"))) if d.exists() else 0
        if done >= total:
            continue
        short.append((cell, done, total, cell in live))
        if cell in live:
            continue                      # already being worked on; leave it alone
        for j in js:
            if n_sub >= a.max_jobs:
                break
            res = SF.RES[j["kind"]]
            part = parts[n_sub % len(parts)]
            sh = cmds / f"{j['name']}.sbatch"
            sh.write_text(SF.SCRIPT.format(
                name=j["name"], partition=part, account=SF.ACCOUNT_FOR(part), nice=j["nice"],
                logdir=logs, constraint=res["constraint"], mem=res["mem"], time=res["time"],
                hf_home=cfg.paths.hf_home, vwm_repo=cfg.paths.vwm_repo,
                seva_repo=cfg.paths.seva_repo, bf16="1" if j["bf16"] else "0",
                cmd=" ".join(shlex.quote(x) for x in j["cmd"])))
            if a.dry_run:
                continue
            r = subprocess.run(["sbatch", "--parsable", str(sh)], capture_output=True, text=True)
            if r.returncode == 0:
                n_sub += 1

    print(f"topup: {len(by_cell)} cells expected, {len(short)} short")
    for cell, done, total, is_live in short:
        print(f"  {cell:<50} {done:>3}/{total}" + ("  (jobs live)" if is_live else "  -> RESUBMIT"))
    print(f"topup: {n_sub} shard job(s) {'would be ' if a.dry_run else ''}resubmitted")


if __name__ == "__main__":
    main()

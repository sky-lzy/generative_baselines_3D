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
import json
import os
import shlex
import time
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS))
import run_nvs_fair as R  # noqa: E402
import submit_fair as SF  # noqa: E402


def live_shards():
    """FULL shard job names currently queued or running, e.g. cell__sh03."""
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                       capture_output=True, text=True)
    return {ln[len("fnvs_"):] for ln in q.stdout.split() if ln.startswith("fnvs_")}


# Anti-thrash: a shard that is short and not live is resubmitted at most this often. Without it, a
# cell that is merely EARLY would have its idle shards resubmitted on every 4-minute driver tick,
# and each resubmission costs 5-11 minutes of weight loading.
COOLDOWN_S = 1500
STATE = NVS / "slurm" / "topup_state.json"


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=0))


def shard_missing(cfg, cell, method, ds, ncf, k, n, total_scenes, scene_names):
    """How many of THIS shard's own scenes are still missing.

    Resubmitting a shard that already finished its slice is pure waste -- ~5-11 minutes of weight
    loading for a no-op -- and with a 25-minute cooldown that repeats. So the decision is made per
    shard on its OWN assigned scenes, using the same interleaved rule the submitter used
    (position n in the pool, n % N == k), not on the cell's overall count.
    """
    mine = [i for pos, i in enumerate(range(total_scenes)) if pos % n == k]
    if method == "seva":
        d = Path(cfg.preds.fair) / cell
        if not d.exists():
            d = Path(cfg.paths.seva_repo) / "work_dirs/demo/img2img" / f"fair_{cell}"
        sroot = Path(cfg.paths.scenes_fair) / ds
        miss = 0
        for i in mine:
            sc = scene_names[i]
            need = len(json.load(open(sroot / sc / f"train_test_split_{ncf}.json"))["test_ids"])
            got = len(list((d / sc / "samples-rgb").glob("*.png"))) if d.exists() else 0
            if got < need:
                miss += 1
        return miss
    d = Path(cfg.preds.fair) / cell
    return sum(1 for i in mine
               if not (d / f"sample_{i:05d}" / "rgb_metrics.json").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_jobs", type=int, default=200, help="cap on shards resubmitted per pass")
    a = ap.parse_args()

    sys.argv = [sys.argv[0]]
    cfg = R.load_cfg()
    live = live_shards()
    state = load_state()
    now = time.time()
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
        scene_names = sorted(d for d in os.listdir(sroot) if (sroot / d).is_dir())
        total = len(scene_names)
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
        n_live = sum(1 for j in js if j["name"] in live)
        short.append((cell, done, total, n_live))
        # SHARD-level, not cell-level. A single TIMEOUT used to hide behind nine healthy shards
        # until the whole cell went idle, which on a deadline is an hour lost for nothing.
        for j in js:
            if n_sub >= a.max_jobs:
                break
            if j["name"] in live:
                continue
            if now - float(state.get(j["name"], 0)) < COOLDOWN_S:
                continue
            ncf_ = int(cell.split("__ncf")[1][0])
            if shard_missing(cfg, cell, method, ds, ncf_, j["shard"], j["nshards"],
                             total, scene_names) == 0:
                continue                  # this shard finished its own slice; nothing to redo
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
                state[j["name"]] = now

    if not a.dry_run:
        save_state(state)
    print(f"topup: {len(by_cell)} cells expected, {len(short)} short")
    for cell, done, total, n_live in short:
        print(f"  {cell:<50} {done:>3}/{total}  {n_live} shard(s) live")
    print(f"topup: {n_sub} shard job(s) {'would be ' if a.dry_run else ''}resubmitted")


if __name__ == "__main__":
    main()

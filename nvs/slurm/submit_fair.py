#!/usr/bin/env python3
"""SLURM fan-out for the fair NVS benchmark: (method x dataset x ncf x scale) x scene-shards.

One cell is 128 scenes of diffusion sampling — far too slow for a single GPU inside a deadline, so
every cell is split into SHARDS that run concurrently as independent one-GPU jobs and write into the
SAME preds directory. Sharding is interleaved by scene index (i % N == shard), which matters twice:

  * sample_XXXXX indices stay GLOBAL, identical to an unsharded run, so the scene<->index mapping the
    scorer relies on is unchanged and shards write disjoint directories (no write contention).
  * every shard draws the same mix of easy/hard scenes, so they finish together instead of one shard
    straggling on a run of long scenes.

Both engines are per-scene resumable (--resume for ours, PNG-count skip for SEVA), so a preempted
requeue job restarts where it stopped. That is what makes it safe to run on the requeue partitions.

PHASE ORDER is the user's: 50f/2-view, then 4dim/1-view, then 4dim/2-view, then 50f/1-view. Phases
are all submitted at once but with increasing --nice, so SLURM drains them roughly in order without
dependency chains stalling the whole fleet when one job is preempted.

CPU POLICY: the engine's DataLoader runs in-process (batch_size=1, no workers) — it is a ONE-CORE
job. FASRC flagged our previous 12-core allocations as ~92% wasted, so every job here takes
--cpus-per-task=2 (worker + margin) with OMP/MKL threads pinned to match.

    python3 nvs/slurm/submit_fair.py --phases p1              # submit one phase
    python3 nvs/slurm/submit_fair.py --phases p1,p2 --dry_run  # print sbatch commands only
    python3 nvs/slurm/submit_fair.py --status                  # what is queued/running/done
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS))
import run_nvs_fair as R  # noqa: E402  (single source of truth for the inference commands)

SLURM = NVS / "slurm"
CMDS = SLURM / "cmds"
LOGS = SLURM / "logs"

# (phase, dataset, ncf) in the order the user asked for them.
PHASES = [
    ("p1", "re10k128_50f", 2),
    ("p2", "re10k128_4dim", 1),
    ("p3", "re10k128_4dim", 2),
    ("p4", "re10k128_50f", 1),
]

# Per-kind SLURM resources. v100 is excluded everywhere: cc7.0 has no bf16, which every one of
# our DiT runs needs. a100-mig nodes carry the feature 'a100-mig', not 'a100', so -C a100 already
# excludes the 10/20GB slices without naming them.
# rtx6000pro is RTX PRO 6000 Blackwell Server Edition: 96GB (more than an H100) and full bf16.
# Excluding it cost us access to ~120 idle GPUs on the requeue partitions, which on a deadline is
# the difference between finishing and not — it is listed FIRST for every kind for that reason.
# TIME LIMIT is deliberately short. These are backfill jobs on preemptible requeue partitions, and
# backfill scheduling favours short jobs heavily: a 3h request sat behind everything, while the same
# work at 1.5h starts almost immediately. Sized so one 16-scene shard (~50 min) fits with margin.
# The alternatives were measured and are worse -- kempner_h100 projected a 11:52 start, seas_gpu
# 10:14, and kempner_h200_priority rejects our group outright.
RES = {
    "ours_5b":   dict(constraint="rtx6000pro|h100|h200", mem="96G", time="01:30:00"),
    "ours_1p3b": dict(constraint="rtx6000pro|h100|h200|a100", mem="64G", time="01:30:00"),
    "seva":      dict(constraint="rtx6000pro|h100|h200|a100", mem="64G", time="01:30:00"),
}


def ACCOUNT_FOR(part):
    """kempner_* partitions bill kempner_ydu_lab; FASRC partitions bill ydu_lab.
    The wrong pairing is rejected outright, not silently redirected."""
    return "kempner_ydu_lab" if part.startswith("kempner") else "ydu_lab"


def scene_list(data_root):
    return sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))


def build_jobs(cfg, phases, shards, limit_scenes=None):
    """-> list of dicts: one runnable shard each, with its fully-formed inference command."""
    tags = [t for t, on in cfg.select.methods.items() if on]
    jobs = []
    for pi, (ph, ds, ncf) in enumerate(PHASES):
        if ph not in phases:
            continue
        data_root = f"{cfg.paths.scenes_fair}/{ds}"
        scenes = scene_list(data_root)
        if limit_scenes:
            scenes = scenes[:int(limit_scenes)]
        for tag in tags:
            spec = R.method_spec(cfg, tag)
            for sc in R.scales_for(cfg, ds, ncf, spec["kind"]):
                cell = f"{tag}__{ds}__ncf{ncf}__s{sc:g}"
                infer = R.steps_for(cfg, spec, ds, ncf, sc)[0].cmd
                n = min(shards, len(scenes))
                for k in range(n):
                    cmd = list(infer)
                    if spec["kind"] == "seva":
                        # SEVA takes an explicit scene subset; strip any --scenes the runner added.
                        if "--scenes" in cmd:
                            i = cmd.index("--scenes"); del cmd[i:i + 2]
                        cmd += ["--scenes", ",".join(scenes[k::n])]
                    else:
                        cmd += ["--shard", f"{k}/{n}"]
                    jobs.append(dict(phase=ph, nice=pi * 1000, cell=cell, shard=k, nshards=n,
                                     kind=spec["kind"], bf16=bool(spec.get("bf16")), cmd=cmd,
                                     name=f"{cell}__sh{k:02d}"))
    return jobs


SCRIPT = """#!/bin/bash
#SBATCH --job-name=fnvs_{name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint="{constraint}"
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --nice={nice}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={logdir}/{name}.%j.out
set -uo pipefail
echo "=== $(date '+%F %T') host=$(hostname) job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# one-core job (see submit_fair.py CPU POLICY): do not let BLAS spawn a thread per core
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME={hf_home}
export VWM_REPO={vwm_repo}
export SEVA_REPO={seva_repo}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INFER_DIT_BF16={bf16}
cd {vwm_repo}
{cmd}
rc=$?
echo "=== $(date '+%F %T') EXIT $rc"
exit $rc
"""


def quote(c):
    import shlex
    return " ".join(shlex.quote(x) for x in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="p1", help="comma list of p1,p2,p3,p4 (or 'all')")
    ap.add_argument("--shards", type=int, default=8, help="scene-shards per cell")
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue",
                    help="comma list; assigned ROUND-ROBIN, not as a union — SLURM rejects\n                         multipartition submissions that include kempner_requeue")
    ap.add_argument("--limit_scenes", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_submit", type=int, default=None, help="safety cap on jobs submitted")
    ap.add_argument("--overrides", nargs="*", default=[], help="run_nvs_fair cfg overrides, k=v")
    a = ap.parse_args()

    sys.argv = [sys.argv[0]] + list(a.overrides)      # load_cfg reads OmegaConf.from_cli()
    cfg = R.load_cfg()

    phases = [p for p, _, _ in PHASES] if a.phases == "all" else \
        [p.strip() for p in a.phases.split(",") if p.strip()]
    bad = [p for p in phases if p not in [x for x, _, _ in PHASES]]
    if bad:
        sys.exit(f"ERROR: unknown phase(s) {bad}")

    jobs = build_jobs(cfg, phases, a.shards, a.limit_scenes)
    if a.max_submit:
        jobs = jobs[:a.max_submit]
    CMDS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print(f"phases={phases}  cells={len({j['cell'] for j in jobs})}  jobs={len(jobs)}  "
          f"partition={a.partition}\n")
    submitted = []
    parts = [x.strip() for x in a.partition.split(',') if x.strip()]
    for ji, j in enumerate(jobs):
        res = RES[j["kind"]]
        sh = CMDS / f"{j['name']}.sbatch"
        sh.write_text(SCRIPT.format(
            name=j["name"], partition=parts[ji % len(parts)],
            account=ACCOUNT_FOR(parts[ji % len(parts)]), nice=j["nice"], logdir=LOGS,
            constraint=res["constraint"], mem=res["mem"], time=res["time"],
            hf_home=cfg.paths.hf_home, vwm_repo=cfg.paths.vwm_repo,
            seva_repo=cfg.paths.seva_repo, bf16="1" if j["bf16"] else "0",
            cmd=quote(j["cmd"])))
        if a.dry_run:
            print(f"DRY  sbatch {sh}")
            continue
        r = subprocess.run(["sbatch", "--parsable", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {j['name']}: {r.stderr.strip()}")
            continue
        jid = r.stdout.strip().split(";")[0]
        submitted.append(dict(jobid=jid, **{k: v for k, v in j.items() if k != "cmd"}))
        print(f"  {jid}  {j['name']}")

    if submitted:
        man = SLURM / "submitted.jsonl"
        with open(man, "a") as f:
            for s in submitted:
                f.write(json.dumps(s) + "\n")
        print(f"\n{len(submitted)} jobs submitted; manifest appended to {man}")


if __name__ == "__main__":
    main()

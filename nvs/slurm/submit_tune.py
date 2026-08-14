#!/usr/bin/env python3
"""Stage 0 of the fair NVS benchmark: pick each method's SAMPLER settings on 10 held-out scenes.

Why this runs before anything else
----------------------------------
Both sides have a classifier-free-guidance dial that nobody has ever tuned on this data, and both
defaults are arbitrary rather than optimal:

  OURS   `--hist_guidance` (engine default 1.0) enters as
             flow = flow_cond*(1 + lang + hist) - lang*flow_no_lang - hist*flow_no_hist
         i.e. an effective CFG scale of (1 + hist) on the CONDITIONING axis, where the negative
         branch replaces the conditioning latents with noise. hist=1.0 therefore means every
         campaign number so far was sampled at guidance 2.0 — and at TWICE the model calls, since
         the uncond branch is a second forward pass. hist=0.0 is both a different operating point
         and half the cost, so this grid is a correctness question and a budget question at once.
         `--lang_guidance` (default 0.0) is probed but expected to be inert: svc_scenes emits an
         EMPTY caption, so the "conditional" text embedding carries no scene information.

  SEVA   `--cfg` (default 2.0). Its own docs/CLI_USAGE.md raises this to 6.0 for single-view
         RealEstate10K. Tuning our guidance while pinning SEVA's to a default would be exactly the
         swept-vs-unswept comparison this benchmark exists to avoid, so SEVA gets a grid of the
         same depth around the value its authors recommend.

Both grids run on the SAME 10 scenes, at the two extreme conditioning densities (50f/2-view, the
easiest, and 4dim/1-view, the hardest), because the guidance optimum generally moves with how much
conditioning the model has. The winner per (method, ncf) is what the full 128-scene run uses.

These jobs are also the end-to-end SMOKE TEST — no frame has yet been generated through this
pipeline — and the source of the qualitative GT/ours/SEVA check for intrinsics sanity.

    python3 nvs/slurm/submit_tune.py --dry_run
    python3 nvs/slurm/submit_tune.py
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS))
import run_nvs_fair as R  # noqa: E402
from submit_fair import RES, SCRIPT, scene_list, ACCOUNT_FOR  # noqa: E402

SLURM = NVS / "slurm"
N_SCENES = 10

# (dataset, ncf): the two extremes of conditioning density.
CELLS = [("re10k128_50f", 2), ("re10k128_4dim", 1)]

# Ours: (hist_guidance, lang_guidance). Grid over the conditioning-CFG axis; two lang probes to
# confirm text guidance is inert on caption-free scenes rather than assuming it.
OURS_GRID = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (1.0, 2.0)]
# SEVA: same depth, bracketing BOTH its 2.0 default and the 6.0 its docs prescribe for 1-view RE10K.
SEVA_GRID = {2: [1.5, 2.0, 3.0, 4.0], 1: [2.0, 4.0, 6.0, 8.0]}

# Which of our models to tune. F is the flagship the user asked about; the 1.3B raw-camera models
# share the identical sampler code, so F's optimum is applied to them unless time allows a re-tune.
OURS_METHODS = ["F_5b"]


def tune_tag(method, ds, ncf, **kw):
    bits = "_".join(f"{k}{v:g}" for k, v in sorted(kw.items()))
    return f"TUNE__{method}__{ds}__ncf{ncf}__{bits}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue",
                    help="comma list; assigned ROUND-ROBIN, not as a union — SLURM rejects\n                         multipartition submissions that include kempner_requeue")
    ap.add_argument("--n_scenes", type=int, default=N_SCENES)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overrides", nargs="*", default=[])
    a = ap.parse_args()

    sys.argv = [sys.argv[0]] + list(a.overrides)
    cfg = R.load_cfg()
    cmds = SLURM / "cmds"; cmds.mkdir(parents=True, exist_ok=True)
    logs = SLURM / "logs"; logs.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ds, ncf in CELLS:
        data_root = f"{cfg.paths.scenes_fair}/{ds}"
        scenes = scene_list(data_root)[:a.n_scenes]
        # ---- ours ------------------------------------------------------------------------
        for m in OURS_METHODS:
            spec = R.method_spec(cfg, m)
            base_scale = R.scales_for(cfg, ds, ncf, spec["kind"])
            # tune guidance at ONE scale (the middle of the grid) so the two searches stay
            # separable; the scale search then runs at the chosen guidance.
            sc = base_scale[len(base_scale) // 2]
            for hg, lg in OURS_GRID:
                name = tune_tag(m, ds, ncf, hg=hg, lg=lg)
                cmd = list(R.steps_for(cfg, spec, ds, ncf, sc)[0].cmd)
                # redirect output to the tuning cell dir
                cmd[cmd.index("--output_dir") + 1] = f"{cfg.preds.fair}/{name}"
                cmd += ["--extra", f"--max_samples={a.n_scenes}",
                        "--extra", "--hist_guidance", "--extra", f"{hg:g}",
                        "--extra", "--lang_guidance", "--extra", f"{lg:g}"]
                jobs.append(dict(name=name, kind=spec["kind"], bf16=bool(spec.get("bf16")),
                                 cmd=cmd, meta=dict(method=m, ds=ds, ncf=ncf, scale=float(sc),
                                                    hist_guidance=hg, lang_guidance=lg)))
        # ---- seva ------------------------------------------------------------------------
        spec = R.method_spec(cfg, "seva")
        base_scale = R.scales_for(cfg, ds, ncf, "seva")
        sc = base_scale[len(base_scale) // 2]
        for c in SEVA_GRID[ncf]:
            name = tune_tag("seva", ds, ncf, cfg=c)
            cmd = list(R.steps_for(cfg, spec, ds, ncf, sc)[0].cmd)
            cmd[cmd.index("--output_dir") + 1] = f"{cfg.preds.fair}/{name}"
            cmd[cmd.index("--cfg") + 1] = f"{c:g}"
            cmd[cmd.index("--save_subdir") + 1] = f"fair_{name}"
            cmd += ["--scenes", ",".join(scenes)]
            jobs.append(dict(name=name, kind="seva", bf16=False, cmd=cmd,
                             meta=dict(method="seva", ds=ds, ncf=ncf, scale=float(sc), cfg=c)))

    print(f"tuning: {len(jobs)} jobs x {a.n_scenes} scenes\n")
    sub = []
    parts = [x.strip() for x in a.partition.split(',') if x.strip()]
    for ji, j in enumerate(jobs):
        res = RES[j["kind"]]
        sh = cmds / f"{j['name']}.sbatch"
        sh.write_text(SCRIPT.format(
            name=j["name"], partition=parts[ji % len(parts)],
            account=ACCOUNT_FOR(parts[ji % len(parts)]), nice=0, logdir=logs,
            constraint=res["constraint"], mem=res["mem"], time="01:30:00",
            hf_home=cfg.paths.hf_home, vwm_repo=cfg.paths.vwm_repo,
            seva_repo=cfg.paths.seva_repo, bf16="1" if j["bf16"] else "0",
            cmd=" ".join(shlex.quote(x) for x in j["cmd"])))
        if a.dry_run:
            print(f"DRY {j['name']}\n    {' '.join(j['cmd'])}\n"); continue
        r = subprocess.run(["sbatch", "--parsable", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {j['name']}: {r.stderr.strip()}"); continue
        jid = r.stdout.strip().split(";")[0]
        print(f"  {jid}  {j['name']}")
        sub.append(dict(jobid=jid, name=j["name"], **j["meta"]))
    if sub:
        with open(SLURM / "tune_submitted.jsonl", "a") as f:
            for s in sub:
                f.write(json.dumps(s) + "\n")
        print(f"\n{len(sub)} tuning jobs submitted")


if __name__ == "__main__":
    main()

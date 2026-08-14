#!/usr/bin/env python3
"""Deep scale search on the 10 single-view scenes where we currently beat SEVA.

PURPOSE / what this can and cannot show
---------------------------------------
The scenes are chosen BECAUSE we win on them, which is a favourable subset by construction. So:
  * us winning here is WEAK evidence -- it is the subset selected for it;
  * us LOSING here, after both sides get a full search, is strong evidence the 128-scene single-view
    result was an artefact of shallow search rather than a real advantage.
The point is the second case. Read it as a stress test of our own claim, not as a demonstration.

SEVA gets its paper protocol EXACTLY
------------------------------------
docs/CLI_USAGE.md, single-view regime: sweep `camera_scale` over 0.1..2.0 in steps of 0.1 (20 points)
and set `--cfg 6.0` for RealEstate10K. Both are applied verbatim, no substitutions. (Our own 10-scene
grid actually preferred cfg 2.0 over 6.0 by ~0.32 dB, so the paper protocol may still understate SEVA
-- noted rather than silently "improved", because the instruction was to run it as its paper does.)

OUR grid is 20 points too, placed where the curve says
-----------------------------------------------------
Equal depth (20 vs 20) is the fairness constraint; WHERE the points go is informed by the measured
128-scene curves, so the budget is not wasted in regions the model clearly does not peak:

  4DiM 1-view   0.4->15.313  0.5->15.774  0.7->15.957  1.0->14.935  1.5->13.263
                Clean interior peak at ~0.7, monotone decline above 1.0 -> dense in [0.45,0.95],
                only a few coarse probes beyond 1.0 to confirm the decline continues.

  50-frame 1-view  0.5->17.220  0.75->17.029  1.0->16.279  1.5->14.505  2.0->13.606
                Monotone DECREASING from the grid's own lower edge, i.e. the peak is at or below 0.5
                and the original grid never probed there. This is a real defect in the earlier sweep:
                our 17.220 is a lower bound. Expand DOWNWARD to 0.05.

Ours also runs at hist_guidance=0.5, the value the completed 10-scene guidance grid picked for
single-view (+0.717 dB over the 1.0 the 128-scene pass used). So this pass is our best known config.

Writes only into preds_ss10/ and results_ss10/ with a __ss10 cell suffix -- disjoint from both the
hg1.0 pass and the hg0.5 pass.
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
sys.path.insert(0, str(NVS / "report"))
import run_nvs_fair as R  # noqa: E402
import submit_fair as SF  # noqa: E402
from build_report import collect, best  # noqa: E402

# SEVA's paper sweep, verbatim: 0.1 .. 2.0 step 0.1
SEVA_SCALES = [round(0.1 * i, 1) for i in range(1, 21)]

# Ours: 20 points each, dense where the measured curve peaks (see module docstring).
OURS_SCALES = {
    ("re10k128_4dim", 1): [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                           0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.50],
    ("re10k128_50f", 1): [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.25, 1.50],
}
CELLS = [("re10k128_4dim", 1), ("re10k128_50f", 1)]
OURS_HG = 0.5
SEVA_CFG = 6.0


def pick_scenes(results, ds, ncf, n, primary="F_5b"):
    """The n scenes with our LARGEST per-scene PSNR margin over SEVA, each side at its own best
    scale from the completed 128-scene pass."""
    d = collect(results)
    o = best(d.get((ds, ncf, primary), []))
    s = best(d.get((ds, ncf, "seva"), []))
    if not (o and s):
        sys.exit(f"ERROR: need both {primary} and seva results for {ds} ncf{ncf}")
    common = sorted(set(o["rows"]) & set(s["rows"]))
    margins = sorted(((float(o["rows"][c]["psnr"]) - float(s["rows"][c]["psnr"]), c)
                      for c in common), reverse=True)
    return [c for _, c in margins[:n]], margins[:n], o["cell"], s["cell"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="kempner_requeue,gpu_requeue")
    ap.add_argument("--n_scenes", type=int, default=10)
    ap.add_argument("--results", default=str(NVS / "results_fair"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--ours_hg", type=float, default=OURS_HG)
    ap.add_argument("--seva_cfg", type=float, default=SEVA_CFG)
    ap.add_argument("--seva_scales", default=None,
                    help="comma list overriding SEVA's sweep. Used to extend BELOW its paper's 0.1 "
                         "floor: on these scenes its curve is still rising at 0.1, so the paper "
                         "protocol leaves it pinned at the grid edge and its number is a lower "
                         "bound. Going past the paper in the BASELINE's favour is the only way a win "
                         "here means anything.")
    ap.add_argument("--ours_scales", default=None, help="comma list overriding our sweep")
    ap.add_argument("--suffix", default="__ss10",
                    help="cell suffix. A second (guidance, cfg) pair goes in its OWN suffix so both "
                         "sides end up searched over the SAME number of configurations -- 20 scales "
                         "x 2 guidance values each. Searching two guidance values for us while "
                         "pinning SEVA to one would be a depth asymmetry in our favour.")
    a = ap.parse_args()
    ours_hg, seva_cfg = a.ours_hg, a.seva_cfg

    sys.argv = [sys.argv[0]]
    cfg = R.load_cfg()
    preds = NVS / "preds_ss10"
    res = NVS / "results_ss10"
    preds.mkdir(exist_ok=True)
    res.mkdir(exist_ok=True)
    cmds = NVS / "slurm" / "cmds"; cmds.mkdir(parents=True, exist_ok=True)
    logs = NVS / "slurm" / "logs"; logs.mkdir(parents=True, exist_ok=True)
    parts = [x.strip() for x in a.partition.split(",") if x.strip()]

    manifest = {}
    jobs = []
    for ds, ncf in CELLS:
        scenes, margins, ocell, scell = pick_scenes(a.results, ds, ncf, a.n_scenes)
        all_scenes = sorted(d for d in os.listdir(Path(cfg.paths.scenes_fair) / ds)
                            if (Path(cfg.paths.scenes_fair) / ds / d).is_dir())
        idxs = [all_scenes.index(s) for s in scenes]
        manifest[f"{ds}__ncf{ncf}"] = dict(
            scenes=scenes, indices=idxs, ours_baseline_cell=ocell, seva_baseline_cell=scell,
            baseline_margins_dB={c: round(m, 3) for m, c in margins})
        print(f"\n{ds} ncf{ncf}: 10 scenes where we win (baseline margin, dB)")
        for m, c in margins:
            print(f"    {c}  {m:+.3f}")

        # ---- ours: 20 scales at hg0.5 -----------------------------------------------------
        spec = R.method_spec(cfg, "F_5b")
        for sc in ([float(x) for x in a.ours_scales.split(",")] if a.ours_scales
                   else OURS_SCALES[(ds, ncf)]):
            cell = f"F_5b__{ds}__ncf{ncf}__s{sc:g}{a.suffix}"
            cmd = [sys.executable, str(NVS / "inference/run_nvs_infer.py"),
                   "--ckpt_path", spec["ckpt"], "--algorithm", spec["algorithm"],
                   "--data_root", f"{cfg.paths.scenes_fair}/{ds}",
                   "--output_dir", f"{preds}/{cell}",
                   "--height", str(spec["height"]), "--width", str(spec["width"]),
                   "--n_frames", str(R.clip_frames(cfg, ds)), "--num_cond_frames", str(ncf),
                   "--sample_steps", str(cfg.run.sample_steps), "--seed", str(cfg.run.seed)]
            for ov in str(spec.get("dataset_overrides", "") or "").split():
                cmd += ["--dataset_override", ov]
            if sc != 1.0:
                cmd += ["--dataset_override", f"moment_scale_mult={sc:g}"]
            cmd += ["--resume", "--scene_indices", ",".join(map(str, idxs)),
                    "--extra=--save_raw", f"--extra=--hist_guidance={ours_hg:g}",
                    "--extra=--lang_guidance=0"]
            jobs.append(dict(name=cell, kind=spec["kind"], bf16=True, cmd=cmd))

        # ---- SEVA: its paper's 20-point sweep, cfg 6.0 -------------------------------------
        sspec = R.method_spec(cfg, "seva")
        for sc in ([float(x) for x in a.seva_scales.split(",")] if a.seva_scales
                   else SEVA_SCALES):
            cell = f"seva__{ds}__ncf{ncf}__s{sc:g}{a.suffix}"
            cmd = [sys.executable, str(NVS / "inference/run_seva_infer.py"),
                   "--data_root", f"{cfg.paths.scenes_fair}/{ds}",
                   "--output_dir", f"{preds}/{cell}",
                   "--num_cond_frames", str(ncf), "--camera_scale", f"{sc:g}",
                   "--H", str(sspec["H"]), "--W", str(sspec["W"]), "--cfg", f"{seva_cfg:g}",
                   "--seva_repo", cfg.paths.seva_repo, "--save_subdir", f"fair_{cell}",
                   "--scenes", ",".join(scenes)]
            jobs.append(dict(name=cell, kind="seva", bf16=False, cmd=cmd))

    mp = NVS / "results_ss10" / f"_manifest{a.suffix}.json"
    mp.write_text(json.dumps(manifest, indent=2))
    if a.suffix == "__ss10":
        (NVS / "results_ss10" / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  ours hist_guidance={ours_hg:g} | SEVA --cfg={seva_cfg:g} | suffix {a.suffix}")
    print(f"\n{len(jobs)} cells ({len(SEVA_SCALES)} SEVA scales vs "
          f"{len(OURS_SCALES[CELLS[0]])} ours scales, per benchmark) x {a.n_scenes} scenes")

    n = 0
    for j in jobs:
        r = SF.RES[j["kind"]]
        part = parts[n % len(parts)]
        sh = cmds / f"{j['name']}.sbatch"
        sh.write_text(SF.SCRIPT.format(
            name=j["name"], partition=part, account=SF.ACCOUNT_FOR(part), nice=0, logdir=logs,
            constraint=r["constraint"], mem=r["mem"], time="01:30:00",
            hf_home=cfg.paths.hf_home, vwm_repo=cfg.paths.vwm_repo,
            seva_repo=cfg.paths.seva_repo, bf16="1" if j["bf16"] else "0",
            cmd=" ".join(shlex.quote(x) for x in j["cmd"])))
        if a.dry_run:
            continue
        p = subprocess.run(["sbatch", "--parsable", str(sh)], capture_output=True, text=True)
        if p.returncode == 0:
            n += 1
        else:
            print(f"FAIL {j['name']}: {p.stderr.strip()}")
    print(f"{n} job(s) submitted" + (" (dry run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()

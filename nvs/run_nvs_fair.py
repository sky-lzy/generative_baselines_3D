#!/usr/bin/env python3
"""FAIR NVS benchmark runner — ours vs SEVA on identical input pixels.

Expands (method x dataset x ncf x scale) into ordered inference -> evaluation commands.

    python3 nvs/run_nvs_fair.py run.dry_run=true       # print the exact commands
    python3 nvs/run_nvs_fair.py run.limit_scenes=4     # smoke test
    python3 nvs/run_nvs_fair.py run.only_method=seva run.only_dataset=re10k128_4dim
    python3 nvs/run_nvs_fair.py run.stages=[evaluation]

What makes the comparison fair (all four enforced here, not left to convention):
  1. INPUT PIXELS   every method reads the same 384x288 4:3 scene set built by
                    nvs/data/build_fair_scenes.py, with intrinsics rebased for the crop+resize.
                    SEVA upsamples 2x internally to its 576x768 operating size and rescales K in
                    the same call, so it gets no extra information and no camera mismatch.
  2. NO CODEC GAP   ours runs with --save_raw and is scored from raw_arrays.npz (lossless);
                    SEVA is scored from its PNGs. Neither side goes through H.264.
  3. ONE SCORER     evaluation/score_nvs_fair.py, one code path, centre square crop -> metric_size.
  4. SCALE PARITY   both sides swept to the same depth per cell; best-fixed taken per side. A swept
                    number is never compared against an unswept one.

Separate from run_nvs.py on purpose: that module's value is its verified reproduction of the July
campaign, and this benchmark changes the data and the metric domain. Nothing here imports it.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

NVS = Path(__file__).resolve().parent
DATASETS = ["re10k128_50f", "re10k128_4dim"]


def load_cfg():
    cli = OmegaConf.from_cli()
    parts = [OmegaConf.load(NVS / "configs" / "fair_paths.yaml"),
             OmegaConf.load(NVS / "configs" / "fair_methods.yaml"),
             OmegaConf.load(NVS / "configs" / "fair_config.yaml")]
    # Sampler settings chosen on 10 held-out scenes by evaluation/score_tune.py. Merged AFTER the
    # static config so a completed tuning stage overrides the defaults automatically, and merged
    # BEFORE the CLI so an explicit override still wins. Absent = fall back to engine defaults.
    tuned = NVS / "configs" / "fair_tuned.yaml"
    if tuned.exists():
        parts.append(OmegaConf.load(tuned))
    cfg = OmegaConf.merge(*parts, cli)
    if OmegaConf.select(cli, "paths.standardized_eval") is None:
        cfg.paths.standardized_eval = os.environ.get("BENCHMARK_ROOT", str(NVS.parent))
    OmegaConf.resolve(cfg)
    return cfg


def method_spec(cfg, tag):
    m = dict(cfg.methods[tag])
    for k, v in dict(cfg.kind_defaults.get(m["kind"], {})).items():
        m.setdefault(k, v)
    m["tag"] = tag
    return m


def clip_frames(cfg, ds):
    """n_frames for this scene set, recorded by the builder (50f -> 50, 4dim -> 70)."""
    p = Path(cfg.paths.scenes_fair) / ds / "_clip.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} missing — run nvs/data/build_fair_scenes.py first")
    return int(json.load(open(p))["n_frames"])


def seva_cfg(cfg, ds, ncf):
    """SEVA's guidance scale, in priority order:
         1. tuned.<ds>.ncf<k>.seva.cfg   — measured on the 10-scene tuning grid
         2. seva_cfg_by_ncf.ncf<k>       — its docs' prescription (6.0 for single-view RE10K)
         3. 2.0                          — demo.py's default
    Ours' guidance is tuned on the same 10 scenes, so tuning SEVA's is what keeps the two sides
    symmetric; falling back to its authors' own recommendation is the next-fairest thing."""
    node = OmegaConf.select(cfg, f"tuned.{ds}.ncf{ncf}.seva.cfg")
    if node is None:
        node = OmegaConf.select(cfg, f"seva_cfg_by_ncf.ncf{ncf}")
    return float(node) if node is not None else 2.0


def ours_guidance(cfg, tag, ds, ncf):
    """(hist_guidance, lang_guidance) for one of our models, or None to leave engine defaults.
    hist enters as an effective CFG scale of (1 + hist) on the conditioning axis.

    run.hist_guidance overrides everything when set. That exists so a whole benchmark pass can be
    re-run at one guidance value WITHOUT editing the tuned config, which is what makes an A/B like
    hg1.0-vs-hg0.5 a controlled comparison rather than a config edit nobody can reconstruct later."""
    forced = cfg.run.get("hist_guidance")
    if forced is not None:
        return float(forced), float(cfg.run.get("lang_guidance") or 0.0)
    node = OmegaConf.select(cfg, f"tuned.{ds}.ncf{ncf}.{tag}")
    if node is None:
        return None
    return float(node.get("hist_guidance", 1.0)), float(node.get("lang_guidance", 0.0))


def cell_suffix(cfg):
    """Appended to every cell name. With separate preds/results roots this keeps a re-run pass
    completely disjoint from an earlier one -- distinct preds dirs, CSVs, sbatch files and logs --
    so no previous result, video or raw array can be overwritten."""
    return str(cfg.run.get("cell_suffix") or "")


def scales_for(cfg, ds, ncf, kind):
    side = "seva" if kind == "seva" else "ours"
    node = OmegaConf.select(cfg, f"scales.{ds}.ncf{ncf}.{side}")
    return [float(x) for x in node] if node is not None else [1.0]


class Step:
    def __init__(self, stage, cmd, marker=None):
        self.stage, self.cmd, self.marker = stage, [str(c) for c in cmd], marker


def steps_for(cfg, spec, ds, ncf, scale):
    py = sys.executable
    kind = spec["kind"]
    stag = f"__s{scale:g}"
    cell = f"{spec['tag']}__{ds}__ncf{ncf}{stag}{cell_suffix(cfg)}"
    data_root = f"{cfg.paths.scenes_fair}/{ds}"
    out = f"{cfg.preds.fair}/{cell}"
    csv = f"{cfg.results.fair}/{cell}.csv"
    nfr = clip_frames(cfg, ds)
    limit = cfg.run.get("limit_scenes")

    if kind == "seva":
        infer = [py, str(NVS / "inference/run_seva_infer.py"),
                 "--data_root", data_root, "--output_dir", out,
                 "--num_cond_frames", ncf, "--camera_scale", scale,
                 "--H", spec["H"], "--W", spec["W"], "--cfg", seva_cfg(cfg, ds, ncf),
                 "--seva_repo", cfg.paths.seva_repo, "--save_subdir", f"fair_{cell}"]
        if limit:
            scenes = sorted(d for d in os.listdir(data_root)
                            if os.path.isdir(os.path.join(data_root, d)))[:int(limit)]
            infer += ["--scenes", ",".join(scenes)]
        method_flag = "seva"
    else:
        infer = [py, str(NVS / "inference/run_nvs_infer.py"),
                 "--ckpt_path", spec["ckpt"], "--algorithm", spec["algorithm"],
                 "--data_root", data_root, "--output_dir", out,
                 "--height", spec["height"], "--width", spec["width"],
                 "--n_frames", nfr, "--num_cond_frames", ncf,
                 "--sample_steps", cfg.run.sample_steps, "--seed", cfg.run.seed]
        for ov in str(spec.get("dataset_overrides", "") or "").split():
            infer += ["--dataset_override", ov]
        if scale != 1.0:
            infer += ["--dataset_override", f"moment_scale_mult={scale:g}"]
        if cfg.run.resume:
            infer.append("--resume")
        # NOTE the "=" form: argparse treats `--extra --save_raw` as --extra with a
        # missing value, since the payload itself starts with "-".
        infer += ["--extra=--save_raw"]           # LOSSLESS preds; see run_nvs_infer --extra
        g = ours_guidance(cfg, spec["tag"], ds, ncf)
        if g is not None:
            infer += [f"--extra=--hist_guidance={g[0]:g}", f"--extra=--lang_guidance={g[1]:g}"]
        if limit:
            infer += [f"--extra=--max_samples={int(limit)}"]
        method_flag = "ours"

    ev = [py, str(NVS / "evaluation/score_nvs_fair.py"),
          "--run_dir", out, "--method", method_flag, "--scenes_root", data_root,
          "--split", ncf, "--metric_size", cfg.run.metric_size, "--out_csv", csv]
    if method_flag == "ours" and cfg.run.selftest:
        ev.append("--selftest")
    return [Step("inference", infer), Step("evaluation", ev, marker=csv)]


def build_matrix(cfg):
    om, od, on = cfg.run.get("only_method"), cfg.run.get("only_dataset"), cfg.run.get("only_ncf")
    tags = [str(om)] if om else [t for t, on_ in cfg.select.methods.items() if on_]
    dss = [str(od)] if od else [d for d in DATASETS if cfg.select.datasets.get(d, False)]
    ncfs = [int(on)] if on else [int(k) for k in cfg.select.ncf]
    for t in tags:
        if t not in cfg.methods:
            sys.exit(f"ERROR: unknown method '{t}'")
    for d in dss:
        if d not in DATASETS:
            sys.exit(f"ERROR: unknown dataset '{d}'")
    jobs = []
    for t in tags:
        spec = method_spec(cfg, t)
        for ds in dss:
            for ncf in ncfs:
                for sc in scales_for(cfg, ds, ncf, spec["kind"]):
                    jobs.append((f"{t}__{ds}__ncf{ncf}__s{sc:g}", spec,
                                 steps_for(cfg, spec, ds, ncf, sc)))
    return jobs


def marker_ok(p):
    try:
        rows = [l for l in open(p) if l.strip() and not l.startswith(("scene", "AVERAGE"))]
        return len(rows) > 0
    except Exception:
        return False


def main():
    cfg = load_cfg()
    jobs = build_matrix(cfg)
    logs = Path(cfg.paths.standardized_eval) / "nvs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stages = list(cfg.run.stages)
    print(f"\n== FAIR NVS: {len(jobs)} cells (stages={stages}, gpu={cfg.run.gpu}, "
          f"dry_run={cfg.run.dry_run}) ==\n")
    results = []
    for name, spec, steps in jobs:
        for i, st in enumerate(steps):
            if st.stage not in stages:
                continue
            sid = f"{name}[{st.stage}]"
            if cfg.run.skip_existing and st.marker and marker_ok(st.marker):
                print(f"SKIP {sid} (marker exists)"); results.append((sid, "SKIP")); continue
            if cfg.run.dry_run:
                print(f"DRY  {sid}\n     {' '.join(st.cmd)}"); results.append((sid, "DRY")); continue
            env = dict(os.environ)
            env.update({"HF_HOME": cfg.paths.hf_home, "CUDA_VISIBLE_DEVICES": str(cfg.run.gpu),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "VWM_REPO": cfg.paths.vwm_repo, "SEVA_REPO": cfg.paths.seva_repo,
                        "INFER_DIT_BF16": "1" if spec.get("bf16") else "0"})
            t0 = time.time()
            with open(logs / f"{name}.{st.stage}.log", "w") as lf:
                rc = subprocess.run(st.cmd, env=env, cwd=cfg.paths.vwm_repo,
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            tag = "PASS" if rc == 0 else "FAIL"
            print(f"{tag} {sid}  ({time.time()-t0:.0f}s)  log: nvs/logs/{name}.{st.stage}.log")
            results.append((sid, tag))
            if rc != 0 and cfg.run.stop_on_error:
                sys.exit(f"stopping on error in {sid}")
            if rc != 0:
                break
    bad = [s for s, t in results if t == "FAIL"]
    print(f"\n== {len(results)} steps: "
          + ", ".join(f"{t}={sum(1 for _, x in results if x == t)}"
                      for t in ["PASS", "FAIL", "SKIP", "DRY"]))
    if bad:
        print("FAILED: " + "; ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()

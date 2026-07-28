#!/usr/bin/env python3
"""standardized_eval NVS runner — expands the enabled (method x dataset x conditioning) matrix
into ordered inference -> evaluation commands and runs them sequentially on one GPU.

    python nvs/run_nvs.py                          # run configs/nvs_config.yaml
    python nvs/run_nvs.py run.dry_run=true         # print commands only
    python nvs/run_nvs.py run.stages=[evaluation]  # score existing preds only
    python nvs/run_nvs.py run.only_method=s5bF_20000 run.only_dataset=dl3dv
    python nvs/run_nvs.py select.methods.s5bC_12500=true run.gpu=3
    python nvs/run_nvs.py run.ncf=[2,1]            # add the single-view extension

Any config key is overridable from the CLI (OmegaConf dotlist). Every job's stdout/stderr is
teed to nvs/logs/<job>.log; a PASS/FAIL/SKIP summary prints at the end. Self-contained: does
not import or modify anything from the depth+pose benchmark (../runner.py).
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

NVS = Path(__file__).resolve().parent
DATASETS = ["dl3dv", "spatialvid", "mip", "re10k_c50"]   # dirnames under paths.scenes_root


def load_cfg():
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(
        OmegaConf.load(NVS / "configs" / "nvs_paths.yaml"),
        OmegaConf.load(NVS / "configs" / "nvs_methods.yaml"),
        OmegaConf.load(NVS / "configs" / "nvs_config.yaml"),
        cli,
    )
    # keep explicit CLI overrides; otherwise honor env vars, then the yaml absolute defaults
    if OmegaConf.select(cli, "paths.vwm_repo") is None and os.environ.get("VWM_REPO"):
        cfg.paths.vwm_repo = os.environ["VWM_REPO"]
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


def build_env(cfg, spec):
    env = dict(os.environ)
    env.update({
        "VWM_REPO": cfg.paths.vwm_repo, "HF_HOME": cfg.paths.hf_home,
        "CUDA_VISIBLE_DEVICES": str(cfg.run.gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "INFER_DIT_BF16": "1" if spec.get("bf16") else "0",
    })
    return env


def P(x):  # script path
    return str(NVS / x)


class Step:
    def __init__(self, stage, cmd, marker=None):
        self.stage, self.cmd, self.marker = stage, [str(c) for c in cmd], marker


def marker_ok(path):
    """Result CSV counts as done only if it contains at least one real float value."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        return re.search(r"\d+\.\d", p.read_text()) is not None
    except OSError:
        return False


def jobs_for_method(cfg, spec, ds, ncf):
    """One cell = campaign-verbatim inference (bridge) -> per-scene PSNR CSV (scorer)."""
    py = sys.executable
    t = spec["tag"]
    cell = f"{t}__{ds}__ncf{ncf}"
    data_root = f"{cfg.paths.scenes_root}/{ds}"
    out = f"{cfg.preds.nvs}/{cell}"
    overrides = str(spec.get("dataset_overrides", "") or "").split()
    infer = [py, P("inference/run_nvs_infer.py"), "--ckpt_path", spec["ckpt"],
             "--algorithm", spec["algorithm"], "--data_root", data_root,
             "--output_dir", out, "--height", spec["height"], "--width", spec["width"],
             "--n_frames", 50, "--num_cond_frames", ncf,
             "--sample_steps", cfg.run.sample_steps, "--seed", cfg.run.seed]
    for ov in overrides:
        infer += ["--dataset_override", ov]
    if cfg.run.resume:
        infer.append("--resume")
    return [
        Step("inference", infer),
        Step("evaluation", [py, P("evaluation/score_nvs_psnr.py"), "--run_dir", out,
                            "--data_root", data_root,
                            "--out_csv", f"{cfg.results.nvs}/{cell}.csv"],
             marker=f"{cfg.results.nvs}/{cell}.csv"),
    ]


def build_matrix(cfg):
    only_method = cfg.run.get("only_method")
    only_dataset = cfg.run.get("only_dataset")
    method_tags = (
        [str(only_method)]
        if only_method
        else [tag for tag, enabled in cfg.select.methods.items() if enabled]
    )
    datasets = (
        [str(only_dataset)]
        if only_dataset
        else [ds for ds in DATASETS if cfg.select.datasets.get(ds, False)]
    )
    unknown_methods = [tag for tag in method_tags if tag not in cfg.methods]
    if unknown_methods:
        sys.exit(f"ERROR: unknown method selector(s): {', '.join(unknown_methods)}")
    unknown_datasets = [ds for ds in datasets if ds not in DATASETS]
    if unknown_datasets:
        sys.exit(f"ERROR: unknown dataset selector(s): {', '.join(unknown_datasets)}")

    jobs = []  # (name, spec, steps)
    for tag in method_tags:
        spec = method_spec(cfg, tag)
        for ds in datasets:
            for ncf in [int(k) for k in cfg.run.ncf]:
                jobs.append((f"{tag}__{ds}__ncf{ncf}", spec,
                             jobs_for_method(cfg, spec, ds, ncf)))
    return jobs


def main():
    cfg = load_cfg()
    for d in list(cfg.preds.values()) + list(cfg.results.values()):
        Path(d).mkdir(parents=True, exist_ok=True)
    (NVS / "logs").mkdir(exist_ok=True)

    jobs = build_matrix(cfg)
    stages = list(cfg.run.stages)
    print(f"== standardized_eval/nvs: {len(jobs)} jobs "
          f"(stages={stages}, gpu={cfg.run.gpu}, dry_run={cfg.run.dry_run}) ==\n")

    summary = []
    for name, spec, steps in jobs:
        env = build_env(cfg, spec)
        log = NVS / "logs" / f"{name}.log"
        for i, st in enumerate(steps):
            if st.stage not in stages:
                continue
            sid = f"{name}[{i}:{st.stage}]"
            if cfg.run.skip_existing and st.marker and marker_ok(st.marker):
                summary.append((sid, "SKIP (marker exists)"))
                print(f"-- SKIP {sid}  ({st.marker})")
                continue
            shown = " ".join(st.cmd)
            if cfg.run.dry_run:
                envs = " ".join(f"{k}={env[k]}" for k in
                                ("CUDA_VISIBLE_DEVICES", "INFER_DIT_BF16", "VWM_REPO"))
                print(f"-- DRY {sid}\n   ({envs} \\\n    {shown})")
                summary.append((sid, "DRY"))
                continue
            print(f"-- RUN {sid}\n   {shown}\n   log: {log}")
            t0 = time.time()
            with open(log, "a") as lf:
                lf.write(f"\n===== {sid} @ {time.strftime('%F %T')} =====\n{shown}\n")
                lf.flush()
                rc = subprocess.run(st.cmd, env=env, cwd=cfg.paths.vwm_repo,
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            ok = rc == 0 and (st.marker is None or marker_ok(st.marker))
            status = "PASS" if ok else f"FAIL rc={rc}" + ("" if st.marker is None or rc != 0
                                                          else " (marker missing)")
            summary.append((sid, f"{status}  [{time.time()-t0:.0f}s]"))
            print(f"   -> {status}")
            if not ok and cfg.run.stop_on_error:
                break
        else:
            continue
        break  # only reached via stop_on_error

    print("\n== SUMMARY ==")
    for sid, status in summary:
        print(f"  {status:28s} {sid}")
    n_fail = sum("FAIL" in s for _, s in summary)
    print(f"\n{len(summary)} steps: {n_fail} failed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

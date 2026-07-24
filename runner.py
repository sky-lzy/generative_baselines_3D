#!/usr/bin/env python3
"""standardized_eval runner — expands the enabled (method x dataset x task) matrix into
ordered inference -> evaluation commands and runs them sequentially on one GPU.

    python runner.py                          # run configs/config.yaml
    python runner.py run.dry_run=true         # print commands only
    python runner.py run.stages=[evaluation]  # score existing preds only
    python runner.py run.only_method=s5bF_depth_v2 run.only_dataset=sintel
    python runner.py select.methods.pi3=false run.gpu=3
    python runner.py select.datasets.scannetv2=false

Any config key is overridable from the CLI (OmegaConf dotlist). For parallel runs, launch one
runner per GPU with disjoint selections (or wrap `run.dry_run=true` output in your own sbatch).
Every job's stdout/stderr is teed to logs/<job>.log; a PASS/FAIL/SKIP summary prints at the end.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

SE = Path(__file__).resolve().parent
HOMEFIELD = ["sintel", "tum", "scannetv2", "bonn", "kitti"]
POSE_DS = {"sintel", "tum", "scannetv2"}          # home-field datasets with pose GT
DEPTH_DS = {"sintel", "bonn", "kitti"}            # home-field datasets with depth GT
DSCFG = {"sintel": "pi3seq", "tum": "pi3seq_tum", "scannetv2": "pi3seq_scannetv2",
         "bonn": "pi3seq_bonn", "kitti": "pi3seq_kitti"}


def load_cfg():
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(
        OmegaConf.load(SE / "configs" / "paths.yaml"),
        OmegaConf.load(SE / "configs" / "methods.yaml"),
        OmegaConf.load(SE / "configs" / "config.yaml"),
        cli,
    )
    # This initialized copy lives inside the active VWM checkout.  Keep explicit
    # CLI overrides, but otherwise make code/output roots follow this checkout
    # instead of the archival absolute paths recorded by the source campaign.
    if OmegaConf.select(cli, "paths.vwm_repo") is None:
        cfg.paths.vwm_repo = os.environ.get("VWM_REPO", str(SE.parents[1]))
    if OmegaConf.select(cli, "paths.standardized_eval") is None:
        cfg.paths.standardized_eval = os.environ.get("BENCHMARK_ROOT", str(SE))
    OmegaConf.resolve(cfg)
    return cfg


def method_spec(cfg, tag):
    m = dict(cfg.methods[tag])
    kind = m["kind"]
    for k, v in dict(cfg.kind_defaults.get(kind, {})).items():
        m.setdefault(k, v)
    m["tag"] = tag
    return m


def build_env(cfg, spec):
    env = dict(os.environ)
    env.update({
        "VWM_REPO": cfg.paths.vwm_repo, "PI3_ROOT": cfg.paths.pi3_root,
        "PI3_METRICS": cfg.paths.pi3_metrics, "GEN3D_ROOT": cfg.paths.gen3d_root,
        "GEO4D_DIR": cfg.paths.geo4d_dir, "RE10K_HF50": cfg.paths.re10k_hf50,
        "RE10K_ANN": cfg.paths.re10k_ann, "HF_HOME": cfg.paths.hf_home,
        "CUDA_VISIBLE_DEVICES": str(cfg.run.gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "INFER_DIT_BF16": "1" if spec.get("bf16") else "0",
    })
    return env


def P(x):  # script path
    return str(SE / x)


def ours_component_flags(spec):
    """Translate optional method-registry components into the small inference CLI."""
    flags = []
    if spec.get("depth_vae_ckpt"):
        flags += ["--depth_vae_ckpt", spec["depth_vae_ckpt"]]
    if spec.get("shared_camera_intrinsics", False):
        flags.append("--shared_camera_intrinsics")
    if spec.get("bundle_adjust", False):
        flags.append("--bundle_adjust")
    return flags


class Step:
    def __init__(self, stage, cmd, marker=None):
        self.stage, self.cmd, self.marker = stage, [str(c) for c in cmd], marker


def marker_ok(path):
    """Result CSV counts as done only if it contains at least one real float value —
    a scorer run against missing preds writes headers/zeros and must NOT count as PASS."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        return re.search(r"\d+\.\d", p.read_text()) is not None
    except OSError:
        return False


def jobs_for_ours(cfg, spec, ds, tasks):
    """Ours diffusion models: run_ours_pi3 / run_eval_scannetpp -> pi3-metric scorers."""
    t, py = spec["tag"], sys.executable
    scale = str(spec.get("scale_flags", "") or "").split()
    components = ours_component_flags(spec)
    steps = []
    if ds in HOMEFIELD:
        if not ((ds in POSE_DS and tasks.pose) or (ds in DEPTH_DS and tasks.depth)):
            return None
        tag = f"{t}__contig_first"
        steps.append(Step("inference", [
            py, P("inference/ours/run_ours_pi3.py"), "--ckpt_path", spec["ckpt"],
            "--algorithm", spec["algorithm"], "--model", tag, "--dataset", ds,
            "--dataset_cfg", DSCFG[ds], "--height", spec["height"], "--width", spec["width"],
            "--frame_mode", "contig_first", "--output_dir", cfg.preds.ours,
            "--sample_steps", cfg.run.sample_steps] + scale + components))
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_ours_pi3.py"), "--model", tag, "--dataset", ds,
            "--preds_dir", cfg.preds.ours, "--out_dir", cfg.results.pose],
            marker=f"{cfg.results.pose}/{tag}_{ds}.csv"))
        if ds in DEPTH_DS and tasks.depth:
            steps.append(Step("evaluation", [
                py, P("evaluation/score_depth_hf.py"), "--kind", "ours", "--model", tag,
                "--dataset", ds, "--preds_dir", cfg.preds.ours,
                "--out_dir", cfg.results.depth_ablation],
                marker=f"{cfg.results.depth_ablation}/{tag}_{ds}.csv"))
    elif ds == "re10k50":
        if not tasks.pose:
            return None
        steps.append(Step("inference", [
            py, P("inference/ours/run_ours_pi3.py"), "--ckpt_path", spec["ckpt"],
            "--algorithm", spec["algorithm"], "--model", t, "--dataset", "re10k_hf50",
            "--dataset_cfg", "pi3seq_re10k_hf50", "--height", spec["height"],
            "--width", spec["width"], "--frame_mode", "uniform",
            "--output_dir", cfg.preds.ours, "--sample_steps", cfg.run.sample_steps]
            + scale + components))
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_re10k_dist50.py"), "--preds_dir", cfg.preds.ours,
            "--model", t, "--tag", t, "--out_dir", cfg.results.re10k50],
            marker=f"{cfg.results.re10k50}/{t}_re10k.csv"))
    elif ds == "scannetpp":
        # run_eval_scannetpp is inference AND depth scoring in one pass (saves pred/gt c2w too)
        steps.append(Step("inference", [
            py, P("inference/ours/run_eval_scannetpp.py"), "--ckpt_path", spec["ckpt"],
            "--algorithm", spec["algorithm"], "--model", t, "--dataset_cfg", "scannetpp_full",
            "--height", spec["height"], "--width", spec["width"],
            "--sample_steps", cfg.run.sample_steps,
            "--preds_dir", cfg.preds.ours, "--out_dir", cfg.results.scannetpp_depth]
            + scale + components,
            marker=f"{cfg.results.scannetpp_depth}/{t}_scannetpp.csv" if tasks.depth else None))
        if tasks.pose:
            steps.append(Step("evaluation", [
                py, P("evaluation/eval_scannetpp_pose.py"), "--preds_dir", cfg.preds.ours,
                "--model", t, "--out_dir", cfg.results.scannetpp_pose],
                marker=f"{cfg.results.scannetpp_pose}/{t}_scannetpp_pose.csv"))
    return steps


def jobs_for_pi3(cfg, ds, tasks):
    """Official PI3: run_pi3_c50 scores pose+depth inline; depth ablation scored separately."""
    py = sys.executable
    steps = []
    if ds in HOMEFIELD:
        if not ((ds in POSE_DS and tasks.pose) or (ds in DEPTH_DS and tasks.depth)):
            return None
        steps.append(Step("inference", [
            py, P("inference/pi3/run_pi3_c50.py"), "--dataset", ds,
            "--out_dir", cfg.results.pose, "--preds_dir", cfg.preds.pi3],
            marker=f"{cfg.results.pose}/pi3_{ds}.csv"))
        if ds in DEPTH_DS and tasks.depth:
            steps.append(Step("evaluation", [
                py, P("evaluation/score_depth_hf.py"), "--kind", "baseline", "--model", "pi3",
                "--dataset", ds, "--preds_dir", cfg.preds.pi3,
                "--out_dir", cfg.results.depth_ablation],
                marker=f"{cfg.results.depth_ablation}/pi3_{ds}.csv"))
    elif ds == "re10k50":
        if not tasks.pose:
            return None
        steps.append(Step("inference", [
            py, P("inference/pi3/run_pi3_re10k50.py"), "--out_dir", cfg.preds.baselines]))
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_re10k_dist50.py"), "--preds_dir", cfg.preds.baselines,
            "--model", "pi3", "--tag", "pi3", "--out_dir", cfg.results.re10k50],
            marker=f"{cfg.results.re10k50}/pi3_re10k.csv"))
    elif ds == "scannetpp":
        steps += _scannetpp_baseline(cfg, "pi3", tasks)
    return steps


def jobs_for_da3_geo4d(cfg, b, ds, tasks):
    """DA3 / Geo4D via run_baseline_pi3 (frame-parity with ours) + pi3-metric scorers."""
    py = sys.executable
    steps = []
    if ds in HOMEFIELD:
        if not ((ds in POSE_DS and tasks.pose) or (ds in DEPTH_DS and tasks.depth)):
            return None
        steps.append(Step("inference", [
            py, P("inference/da3_geo4d/run_baseline_pi3.py"), "--baseline", b,
            "--dataset", ds, "--frame_mode", "contig_first", "--out_dir", cfg.preds.baselines]))
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_baseline_pi3.py"), "--baseline", b, "--dataset", ds,
            "--preds_dir", cfg.preds.baselines, "--out_dir", cfg.results.pose],
            marker=f"{cfg.results.pose}/{b}_{ds}.csv"))
        if ds in DEPTH_DS and tasks.depth:
            steps.append(Step("evaluation", [
                py, P("evaluation/score_depth_hf.py"), "--kind", "baseline", "--model", b,
                "--dataset", ds, "--preds_dir", cfg.preds.baselines,
                "--out_dir", cfg.results.depth_ablation],
                marker=f"{cfg.results.depth_ablation}/{b}_{ds}.csv"))
    elif ds == "re10k50":
        if not tasks.pose:
            return None
        steps.append(Step("inference", [
            py, P("inference/da3_geo4d/run_baseline_pi3.py"), "--baseline", b,
            "--dataset", "re10k_hf50", "--out_dir", cfg.preds.baselines]))
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_re10k_dist50.py"), "--preds_dir", cfg.preds.baselines,
            "--model", b, "--tag", b, "--out_dir", cfg.results.re10k50],
            marker=f"{cfg.results.re10k50}/{b}_re10k.csv"))
    elif ds == "scannetpp":
        steps += _scannetpp_baseline(cfg, b, tasks)
    return steps


def _scannetpp_baseline(cfg, b, tasks):
    py = sys.executable
    cmd = [py, P("inference/da3_geo4d/run_baseline_scannetpp.py"), "--baseline", b,
           "--pose_dir", cfg.preds.spp_pose, "--out_dir", cfg.results.scannetpp_depth]
    if not tasks.depth:
        cmd.append("--pose_only")
    steps = [Step("inference", cmd,
                  marker=(f"{cfg.results.scannetpp_depth}/{b}_scannetpp.csv"
                          if tasks.depth else None))]
    if tasks.pose:
        steps.append(Step("evaluation", [
            py, P("evaluation/eval_scannetpp_pose.py"), "--preds_dir", cfg.preds.spp_pose,
            "--model", b, "--out_dir", cfg.results.scannetpp_pose],
            marker=f"{cfg.results.scannetpp_pose}/{b}_scannetpp_pose.csv"))
    return steps


def build_matrix(cfg):
    tasks = cfg.select.tasks
    all_ds = HOMEFIELD + ["re10k50", "scannetpp"]
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
        else [
            ds
            for ds in all_ds
            if cfg.select.datasets.get(
                "re10k50" if ds == "re10k50" else ds, False
            )
        ]
    )
    unknown_methods = [tag for tag in method_tags if tag not in cfg.methods]
    if unknown_methods:
        sys.exit(f"ERROR: unknown method selector(s): {', '.join(unknown_methods)}")
    unknown_datasets = [ds for ds in datasets if ds not in all_ds]
    if unknown_datasets:
        sys.exit(f"ERROR: unknown dataset selector(s): {', '.join(unknown_datasets)}")

    jobs = []  # (name, spec, steps)
    for tag in method_tags:
        spec = method_spec(cfg, tag)
        for ds in datasets:
            if spec["kind"] in ("ours_5b", "ours_1p3b"):
                steps = jobs_for_ours(cfg, spec, ds, tasks)
            elif spec["kind"] == "pi3":
                steps = jobs_for_pi3(cfg, ds, tasks)
            else:  # da3 / geo4d
                steps = jobs_for_da3_geo4d(cfg, tag, ds, tasks)
            if steps:
                jobs.append((f"{tag}__{ds}", spec, steps))
    return jobs


def main():
    cfg = load_cfg()
    for d in list(cfg.preds.values()) + list(cfg.results.values()):
        Path(d).mkdir(parents=True, exist_ok=True)
    (SE / "logs").mkdir(exist_ok=True)

    jobs = build_matrix(cfg)
    stages = list(cfg.run.stages)
    print(f"== standardized_eval: {len(jobs)} jobs "
          f"(stages={stages}, gpu={cfg.run.gpu}, dry_run={cfg.run.dry_run}) ==\n")

    summary = []
    for name, spec, steps in jobs:
        env = build_env(cfg, spec)
        log = SE / "logs" / f"{name}.log"
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
                print(f"-- DRY {sid}\n   (cd {cfg.paths.vwm_repo} && {envs} ... \\\n    {shown})")
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

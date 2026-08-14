#!/usr/bin/env python3
"""NVS inference bridge — one (method, dataset, conditioning) cell.

Builds the EXACT home-field campaign command (field-for-field identical to
vwm_repo/eval_svc/hf_homefield.sbatch) and runs the source repo's inference engine
(scripts/inference_single_gpu_ray_to_rgb_depth_eval.py, dataset=svc_scenes) with cwd=VWM_REPO:

    python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
        --ckpt_path <CKPT> --algorithm <ALGO> --dataset svc_scenes --output_dir <OUT> \
        --dataset_override data_root=<scenes_root>/<ds> [--dataset_override k=v ...] \
        --height <H> --width <W> --n_frames 50 --num_cond_frames <K> \
        --sample_steps 40 --seed 42 --no_augmentations --show_metrics

Two-view protocol: --num_cond_frames 2 conditions on the FIRST and LAST frame (plus the full
raymap sequence) and generates the 48 interior frames; per-scene PSNR over interior frames is
written by the engine itself to <OUT>/sample_XXXXX/rgb_metrics.json (computed on raw tensors,
BEFORE mp4 compression) — evaluation/score_nvs_psnr.py aggregates those.

Count-aware SKIP (the campaign convention): a cell is complete iff final_stats.json exists AND
its count equals the number of scene dirs; a partial cell from an interrupted run is re-run,
never skipped. INFER_DIT_BF16 / HF_HOME / CUDA_VISIBLE_DEVICES come from the caller's env
(run_nvs.py sets them per method).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def expected_scenes(data_root: str, min_frames: int = 26) -> int:
    """Replicates datasets/svc_scenes.py::_load_records — sorted scene dirs holding a
    transforms.json with >= min_frames frames (sample_00000..N follow this exact order)."""
    n = 0
    for sd in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        tj = sd / "transforms.json"
        if not tj.exists():
            continue
        if len(json.load(open(tj))["frames"]) >= min_frames:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="NVS home-field inference bridge (one cell).")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--data_root", required=True, help="<scenes_root>/<ds> scene-set dir")
    ap.add_argument("--output_dir", required=True, help="preds cell dir <tag>__<ds>__ncf<k>")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--num_cond_frames", type=int, choices=[1, 2], default=2)
    ap.add_argument("--sample_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_override", action="append", default=[],
                    help="extra svc_scenes overrides (repeatable), e.g. scale_mode=global_metric")
    ap.add_argument("--extra", action="append", default=[],
                    help="verbatim extra flag forwarded to the engine (repeatable). Used by the "
                         "fair benchmark to pass --save_raw (lossless raw_arrays.npz predictions, "
                         "so scoring never goes through H.264) and --max_samples for smoke tests.")
    ap.add_argument("--resume", action="store_true",
                    help="skip scenes whose pred_rgb.mp4 exists (indices stay scene-aligned)")
    ap.add_argument("--limit_scenes", type=int, default=None,
                    help="restrict to the FIRST N scenes (by the same sorted order the scorer uses). "
                         "Used for subset scale searches, where the grid runs on a subset and only "
                         "the winning scale is then run on all 128. Composes with --shard: the shard "
                         "is taken within the subset, so shards stay balanced.")
    ap.add_argument("--scene_indices", default=None,
                    help="comma-separated GLOBAL scene indices to run (the sorted-scene-dir order "
                         "the scorer uses). For targeted re-runs on a chosen subset; --limit_scenes "
                         "only takes a prefix and --shard only takes a stride.")
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help="SLURM fan-out: run only scenes with (index %% N) == i, by handing the "
                         "engine --skip_samples for every other index. Sample indices stay GLOBAL, "
                         "so shards of one cell write disjoint sample_XXXXX dirs into the SAME "
                         "output_dir and the scene<->index mapping is identical to an unsharded "
                         "run. Interleaved (not contiguous) so every shard sees the same mix of "
                         "easy/hard scenes and they finish at roughly the same time.")
    args = ap.parse_args()

    vwm = os.environ.get("VWM_REPO")
    if not vwm or not (Path(vwm) / "scripts" / "inference_single_gpu_ray_to_rgb_depth_eval.py").exists():
        sys.exit("ERROR: env VWM_REPO must point at the video_world_model checkout")

    exp = expected_scenes(args.data_root)
    if exp == 0:
        sys.exit(f"ERROR: no scenes found under {args.data_root}")

    # --- scene subset + shard resolution ----------------------------------------------------
    pool = list(range(exp))
    if args.scene_indices:
        pool = [int(x) for x in str(args.scene_indices).split(",") if x.strip() != ""]
        bad = [i for i in pool if i >= exp]
        if bad:
            sys.exit(f"ERROR: --scene_indices {bad} outside 0..{exp - 1}")
    if args.limit_scenes:
        pool = pool[:int(args.limit_scenes)]
    # A caller may also cap the run with a passthrough --max_samples=N (the tuning stage does).
    # Honour it here, or the completeness post-check below demands all 128 scenes and fails a job
    # that did exactly what it was told -- which is worse than not checking, because it makes every
    # genuine "incomplete" report untrustworthy.
    for e in args.extra:
        if e.startswith("--max_samples="):
            try:
                pool = pool[:int(e.split("=", 1)[1])]
            except ValueError:
                pass
    mine = list(pool)
    if args.shard:
        try:
            si, sn = (int(x) for x in str(args.shard).split("/"))
        except Exception:
            sys.exit(f"ERROR: --shard must look like i/N, got {args.shard!r}")
        if not (sn >= 1 and 0 <= si < sn):
            sys.exit(f"ERROR: bad shard {args.shard} (need 0 <= i < N, N >= 1)")
        mine = [i for n, i in enumerate(pool) if n % sn == si]
        if not mine:
            print(f"[nvs-infer] shard {args.shard}: no scenes of {len(pool)} — nothing to do")
            return

    # count-aware SKIP: complete iff final_stats count == expected scene count.
    # Under --shard final_stats.json is written per-shard and is meaningless as a cell-level
    # marker, so the shard relies on --resume for per-scene skipping instead.
    fs = Path(args.output_dir) / "final_stats.json"
    if fs.exists() and not args.shard:
        try:
            c = json.load(open(fs)).get("count", 0)
        except Exception:
            c = 0
        if c == exp:
            print(f"SKIP complete cell: {args.output_dir} ({c}/{exp} scenes)")
            return
        print(f"REDO partial cell: {args.output_dir} ({c}/{exp} scenes)")

    cmd = [sys.executable, "scripts/inference_single_gpu_ray_to_rgb_depth_eval.py",
           "--ckpt_path", args.ckpt_path, "--algorithm", args.algorithm,
           "--dataset", "svc_scenes", "--output_dir", args.output_dir,
           "--dataset_override", f"data_root={args.data_root}"]
    for ov in args.dataset_override:
        cmd += ["--dataset_override", ov]
    cmd += ["--height", str(args.height), "--width", str(args.width),
            "--n_frames", str(args.n_frames), "--num_cond_frames", str(args.num_cond_frames),
            "--sample_steps", str(args.sample_steps), "--seed", str(args.seed),
            "--no_augmentations", "--show_metrics"]
    if args.resume:
        cmd.append("--resume")
    if args.shard or args.limit_scenes or args.scene_indices:
        # the engine iterates ALL scenes and skips by index, so the shard is expressed as the
        # complement. --max_samples must cover the full range or the tail shard is truncated
        # (its default is 50, which silently dropped 78 of 128 scenes once already).
        skip = ",".join(str(i) for i in range(exp) if i not in set(mine))
        cmd += ["--skip_samples", skip, "--max_samples", str(exp)]
    cmd += list(args.extra)

    print(f"[nvs-infer] {exp} scenes"
          + (f" | shard {args.shard} -> {len(mine)} scenes" if args.shard else "")
          + f" | cwd={vwm}\n  " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=vwm).returncode
    if rc != 0:
        sys.exit(rc)
    # post-check: every scene THIS invocation owns must have produced metrics
    done = {int(p.parent.name.split("_")[1])
            for p in Path(args.output_dir).glob("sample_*/rgb_metrics.json")}
    missing = [i for i in mine if i not in done]
    if missing:
        sys.exit(f"ERROR: incomplete after run — {len(missing)}/{len(mine)} owned scenes have no "
                 f"rgb_metrics.json (first few: {missing[:8]})")
    print(f"[nvs-infer] DONE {args.output_dir} ({len(mine)} owned scenes, "
          f"{len(done)}/{exp} in cell)")


if __name__ == "__main__":
    main()

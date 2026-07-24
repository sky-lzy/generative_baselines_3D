"""ScanNet++ video-depth eval for OUR models, scored with the π³ depth metric
(utils.depth.depth_evaluation, align_with_scale=True) VERBATIM.

Unlike the PI3 datasets, ScanNet++ uses OUR existing test set: the ScanNetppDataset
(scannetpp_full config) already yields, per scene at the model's native resolution:
  - videos       (T,3,H,W) RGB in [-1,1]   -> fed to the model
  - depths_raw   (T,1,H,W) GT depth in METERS (ViPE/EXR), at model res
  - depths_valid (T,1,H,W) bool validity mask
So NO crop/slice is needed: pred and GT are already the same frames + resolution.

Per scene:
  pred [-1,1] modified-disparity -> decode_pred_depth (eval_ours_pi3, percentile_clip floors
       the 1/disp inversion) -> per-frame metric-proportional depth (T,H,W).
  GT  = depths_raw (meters); invalid pixels (depths_valid==False or <=0) are set to 0 so the
       π³ depth_evaluation gt>0 mask drops them.
  metrics = depth_evaluation(pred, gt, align_with_scale=True, max_depth=None, use_gpu=False).

Aggregates valid-pixel-weighted (matches PI3 videodepth/eval.py) and writes
eval_pi3/results_ours/<model>_scannetpp.csv. Also saves per-scene pred_depth_norm for repro.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir  # [standardized_eval PATCH]
from tqdm import tqdm

REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] repo root via env
sys.path.insert(0, str(REPO))

from datasets.scannetpp import ScanNetppDataset
# Reuse the VALIDATED model loader + sampler + per-seq reconfigure from the bridge runner.
from run_ours_pi3 import load_model, set_model_frames, _ALGO_CLASSES
from benchmark_components import (
    recover_predicted_cameras,
    restore_rng_state,
    run_verified_sparse_bundle_adjustment,
    save_rng_state,
)

# VALIDATED [-1,1]->metric-depth decode (same percentile-clip conversion as eval_ours_pi3).
sys.path.insert(0, os.environ.get("GEN3D_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines_3D"))  # [standardized_eval PATCH]
from eval_common_v3 import disparity_norm_to_metric_depth

# Load π³'s utils.depth DIRECTLY from its file path. We cannot `from utils.depth import ...`
# here because importing the model (algorithms.*) already registered REPO/utils as the
# `utils` package, which has no `depth` submodule -> ModuleNotFoundError. importlib from the
# explicit PI3 file bypasses the package-name collision.
import importlib.util as _ilu
_PI3_DEPTH = os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parents[2] / "evaluation" / "pi3_metrics")) + "/utils/depth.py"  # [standardized_eval PATCH] verbatim local metric copy
_spec = _ilu.spec_from_file_location("pi3_utils_depth", _PI3_DEPTH)
_pi3_depth = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pi3_depth)
depth_evaluation = _pi3_depth.depth_evaluation


def decode_pred_depth(pred_norm: np.ndarray) -> np.ndarray:
    """[-1,1] normalized disparity -> metric-proportional depth (VERBATIM as eval_ours_pi3)."""
    return disparity_norm_to_metric_depth(
        depth_norm=pred_norm.astype(np.float32), scale=1.0,
    ).astype(np.float64)


DEVICE = "cuda"


def build_cfg(args):
    overrides = [
        "experiment=exp_video",
        f"algorithm={args.algorithm}",
        f"dataset={args.dataset_cfg}",
        "experiment.tasks=[test]",
        "algorithm.load_prompt_embed=true",
        "algorithm.diffusion_forcing.enabled=True",
        "algorithm.diffusion_forcing.mode=ray_depth_prediction_firstlast",
        f"algorithm.sample_steps={args.sample_steps}",
        f"algorithm.model.tuned_ckpt_path='{args.ckpt_path}'",
        f"dataset.n_frames={args.n_frames}",
        f"dataset.height={args.height}",
        f"dataset.width={args.width}",
        f"dataset.raymap_height={args.height // 8}",
        f"dataset.raymap_width={args.width // 8}",
        # GT depth eval only needs depths_raw/depths_valid. Skip the sky-mask zip read:
        # several ViPE mask PNGs are corrupt (libpng zlib error) and aren't used for the
        # metric-depth GT (only for raymap/depth NORMALIZATION, which we don't use here).
        "++dataset.use_sky_mask=false",
        # Deterministic frame order (disable the 50% random temporal reverse).
        "++dataset.no_augmentations=true",
        "algorithm.model.compile=false",
        "algorithm.vae.compile=false",
        "algorithm.text_encoder.compile=false",
    ]
    if getattr(args, "scale_mode", None):
        overrides.append(f"++dataset.scale_mode={args.scale_mode}")
    if getattr(args, "global_metric_scale", None) is not None:
        overrides.append(f"++dataset.global_metric_scale={args.global_metric_scale}")
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):  # [standardized_eval PATCH]
        cfg = compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)
    return cfg


@torch.no_grad()
@torch.autocast(DEVICE, dtype=torch.bfloat16)
def predict_depth_norm(
    model,
    batch,
    sample_steps,
    *,
    shared_camera_intrinsics=False,
    bundle_adjust=False,
):
    m = batch["videos"].shape[1]
    lat_t = set_model_frames(model, m)
    T_lat = lat_t // 4
    hist_indexes = torch.arange(0, T_lat)
    batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    video = model.sample_seq(batch, hist_indexes=hist_indexes,
                             pbar=tqdm(range(sample_steps), desc="sampling", leave=False)).squeeze(0)
    T = len(video) // 4
    assert T == m, f"expected {m} pixel frames, got {T}"
    # Raymap channels -> c2w, IDENTICAL decode to run_ours_pi3.run_one (pose eval).
    pred_ray_d = video[T:2 * T]
    pred_ray_m = video[2 * T:3 * T]
    pred_raymaps = torch.cat([pred_ray_d, pred_ray_m], dim=1).float()        # (T, 6, rh, rw)
    cameras = recover_predicted_cameras(
        pred_raymaps,
        shared_intrinsics=shared_camera_intrinsics,
        model_native_units=bundle_adjust,
    )
    pred_depth = video[3 * T:]
    confidence = getattr(model, "last_depth_confidence", None)
    if confidence is not None:
        confidence = confidence.float().mean(dim=1).squeeze(0).cpu().numpy()
    base_depth = getattr(model, "last_old_depth", None)
    if base_depth is not None:
        base_depth = base_depth.float().mean(dim=1).squeeze(0).cpu().numpy()
    return (
        pred_depth.float().mean(dim=1).cpu().numpy().astype(np.float32),
        cameras,
        None if confidence is None else confidence.astype(np.float32),
        None if base_depth is None else base_depth.astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--algorithm", default="wan_t2v_ray_depth_mot_concat", choices=list(_ALGO_CLASSES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset_cfg", default="scannetpp_full")
    ap.add_argument("--sample_steps", type=int, default=40)
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--max_depth", default="none", help="'none' or a float cap for GT depth")
    ap.add_argument("--save_rgb", action="store_true",
                    help="also dump the sampled RGB frames (uint8 npy) per scene — one dump "
                         "serves all baselines (sampling is deterministic w/ no_augmentations)")
    ap.add_argument("--scale_mode", default=None)
    ap.add_argument("--global_metric_scale", type=float, default=None)
    ap.add_argument(
        "--depth_vae_ckpt",
        default=None,
        help="optional fine-tuned Wan2.2/5B depth VAE v2 checkpoint",
    )
    ap.add_argument(
        "--shared_camera_intrinsics",
        action="store_true",
        help="fit one prediction-only intrinsic matrix per scene",
    )
    ap.add_argument(
        "--bundle_adjust",
        action="store_true",
        help=(
            "run the frozen verified sparse BA (camera poses + sparse points; "
            "dense depth and intrinsics fixed)"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(REPO / "eval_pi3" / "results_ours"))
    ap.add_argument("--preds_dir", default=str(REPO / "eval_pi3" / "preds"))
    ap.add_argument("--shard_idx", type=int, default=0, help="this shard's index (round-robin over scenes)")
    ap.add_argument("--n_shards", type=int, default=1, help="total shards; disjoint shards fill the per-scene cache, then run once with n_shards=1 (all cached) for the full CSV")
    args = ap.parse_args()

    if args.depth_vae_ckpt and args.algorithm != "wan_t2v_ray_depth_mot_concat_5b":
        ap.error("--depth_vae_ckpt is supported here only for the 5B/Wan2.2 model")
    if args.bundle_adjust and not args.depth_vae_ckpt:
        ap.error("--bundle_adjust requires --depth_vae_ckpt (verified BA input)")
    if args.bundle_adjust and not args.shared_camera_intrinsics:
        ap.error("--bundle_adjust requires --shared_camera_intrinsics (verified pipeline)")

    max_depth = None if str(args.max_depth).lower() == "none" else float(args.max_depth)

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    cfg = build_cfg(args)
    print(f"INFO: loading {args.algorithm} from {args.ckpt_path}", flush=True)
    model = load_model(
        cfg.algorithm,
        _ALGO_CLASSES[args.algorithm],
        depth_vae_ckpt=args.depth_vae_ckpt,
    )
    model.hist_guidance = 1.0
    model.lang_guidance = 0.0
    model.sample_steps = args.sample_steps

    # Free the CPU-side copies left over from load_model (esp. the 5B fp32 state_dict) before
    # we start streaming scenes — keeps RAM headroom under the job's memory cgroup.
    gc.collect(); torch.cuda.empty_cache()

    dataset = ScanNetppDataset(cfg.dataset, split="test")
    print(f"INFO: {len(dataset)} scannetpp scenes", flush=True)
    # STREAM ONE SCENE AT A TIME. num_workers=0 (no prefetch buffering) so we never hold more
    # than a single scene's high-res depth/RGB in RAM at once — earlier num_workers=4 buffered
    # ~8 scenes per worker and got OOM-killed by the memory cgroup. We also del + gc.collect()
    # after every scene. (Same model RAM footprint as the bonn/kitti runs, which never OOM'd.)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    preds_root = Path(args.preds_dir) / args.model / "scannetpp"
    preds_root.mkdir(parents=True, exist_ok=True)

    rows = []   # (scene, abs_rel, delta1, valid_pixels)
    for i, batch in enumerate(tqdm(loader, desc=f"{args.model}/scannetpp")):
        if args.n_shards > 1 and (i % args.n_shards) != args.shard_idx:
            del batch; continue   # this scene belongs to another shard
        scene = batch.get("scene_id", [f"scene_{i}"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else str(scene)
        metric_path = preds_root / f"{scene}_metric.json"
        component_path = preds_root / f"{scene}_component_summary.json"
        rng_state_path = preds_root / f"{scene}_rng_state_after.pt"

        # Resume: if this scene was already scored in a prior (possibly killed) run, reuse it.
        # Pose artifacts are required too (older runs saved depth only -> must rerun).
        if metric_path.exists() and (preds_root / f"{scene}_pred_c2w.npy").exists() \
                and component_path.exists() and rng_state_path.exists():
            try:
                m = json.load(open(metric_path))
                restore_rng_state(rng_state_path)
                rows.append((scene, m["abs_rel"], m["delta1"], m["valid_pixels"]))
                print(f"{scene:24s}  (cached) AbsRel {m['abs_rel']:7.4f}  d1 {m['delta1']:7.4f}",
                      flush=True)
                del batch; gc.collect()
                continue
            except (ValueError, KeyError):
                pass

        component_path.unlink(missing_ok=True)
        rng_state_path.unlink(missing_ok=True)

        try:
            pred_norm, cameras, confidence, base_depth = predict_depth_norm(
                model,
                batch,
                args.sample_steps,
                shared_camera_intrinsics=args.shared_camera_intrinsics,
                bundle_adjust=args.bundle_adjust,
            )
            pred_c2w = cameras.c2w
            ba_summary = None
            if args.bundle_adjust:
                np.save(preds_root / f"{scene}_pred_c2w_pre_ba.npy", pred_c2w)
                pred_c2w, ba_summary = run_verified_sparse_bundle_adjustment(
                    rgb_tchw=batch["videos"].squeeze(0).cpu().numpy(),
                    depth_norm=pred_norm,
                    intrinsics=cameras.intrinsics,
                    c2w=pred_c2w,
                    output_dir=preds_root / f"{scene}_ba",
                    repo_root=REPO,
                )
            pred_metric = decode_pred_depth(pred_norm)                          # (T,H,W)

            # Pose artifacts: raw-decoded pred c2w + ViPE GT c2w (first-frame-relative, metric)
            # from the SAME batch, so frame correspondence is exact. Scored separately by
            # eval_pi3/hf/eval_scannetpp_pose.py (evo Sim(3), same as the other pose datasets).
            np.save(preds_root / f"{scene}_pred_c2w.npy", pred_c2w)
            np.save(preds_root / f"{scene}_pred_intrinsics.npy", cameras.intrinsics)
            if args.shared_camera_intrinsics or args.bundle_adjust:
                np.save(preds_root / f"{scene}_pred_c2w_legacy.npy", cameras.legacy_c2w)
            if confidence is not None:
                np.save(preds_root / f"{scene}_pred_depth_confidence.npy", confidence)
            if base_depth is not None:
                np.save(preds_root / f"{scene}_pred_depth_base_same_latent.npy", base_depth)
            np.save(preds_root / f"{scene}_gt_c2w.npy",
                    batch["extrinsics_gt"].squeeze(0).cpu().numpy().astype(np.float64))
            if args.save_rgb:
                rgb = batch["videos"].squeeze(0).cpu().numpy()                  # (T,3,H,W) [-1,1]
                rgb = ((rgb.transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                np.save(preds_root / f"{scene}_rgb.npy", rgb)

            depths_raw = batch["depths_raw"].squeeze(0).squeeze(1).cpu().numpy()  # (T,H,W) meters
            valid = batch["depths_valid"].squeeze(0).squeeze(1).cpu().numpy().astype(bool)
            gt = np.where(valid & (depths_raw > 0), depths_raw, 0.0).astype(np.float64)

            # pred and gt already same (T,H,W) at model res — no resize/crop needed.
            results, *_ = depth_evaluation(
                pred_metric, gt, align_with_scale=True, use_gpu=False, max_depth=max_depth,
            )
            ar = float(results["Abs Rel"]); d1 = float(results["δ < 1.25"])
            vp = int(results["valid_pixels"])
            rows.append((scene, ar, d1, vp))
            np.save(preds_root / f"{scene}_pred_depth_norm.npy", pred_norm)
            json.dump({"abs_rel": ar, "delta1": d1, "valid_pixels": vp}, open(metric_path, "w"))
            save_rng_state(rng_state_path)
            component_tmp = component_path.with_name(f".{component_path.name}.tmp")
            with open(component_tmp, "w") as handle:
                json.dump(
                    {
                        "depth_vae": getattr(model, "depth_vae_metadata", None),
                        "confidence_saved": confidence is not None,
                        "base_depth_same_latent_saved": base_depth is not None,
                        "confidence_used_for_metrics": False,
                        "confidence_used_for_ba": False,
                        "camera_recovery": cameras.summary,
                        "bundle_adjustment": ba_summary,
                    },
                    handle,
                    indent=2,
                    allow_nan=True,
                )
            os.replace(component_tmp, component_path)
            print(f"{scene:24s}  AbsRel {ar:7.4f}  d1 {d1:7.4f}  vpix {vp}", flush=True)
        except Exception as e:
            print(f"{scene:24s}  SKIPPED ({type(e).__name__}: {e})", flush=True)
        finally:
            # Drop all references to this scene's arrays before pulling the next one, so peak
            # RAM stays ~one scene + model (not many scenes).
            pred_norm = pred_metric = depths_raw = valid = gt = results = batch = None
            gc.collect(); torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no scannetpp scenes scored")
    abs_arr = np.array([r[1] for r in rows], float)
    d1_arr = np.array([r[2] for r in rows], float)
    w = np.array([r[3] for r in rows], float)
    abs_rel_mean = float(abs_arr.mean()); d1_mean = float(d1_arr.mean())
    abs_rel_w = float(np.average(abs_arr, weights=w)); d1_w = float(np.average(d1_arr, weights=w))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.model}_scannetpp.csv")
    with open(csv_path, "w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for scene, ar, d1, vp in rows:
            f.write(f"{scene},,,,{ar:.6f},{d1:.6f},{vp}\n")
        f.write(f"AVERAGE(meanseq),,,,{abs_rel_mean:.6f},{d1_mean:.6f},\n")
        f.write(f"AVERAGE(vpixwt_depth),,,,{abs_rel_w:.6f},{d1_w:.6f},\n")

    print("\n==================== SUMMARY ====================")
    print(f"model={args.model} dataset=scannetpp ({len(rows)} scenes)")
    print(f"DEPTH (mean over seqs): AbsRel {abs_rel_mean:.4f}  delta<1.25 {d1_mean:.4f}")
    print(f"DEPTH (vpix-weighted) : AbsRel {abs_rel_w:.4f}  delta<1.25 {d1_w:.4f}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()

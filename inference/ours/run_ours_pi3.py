"""Run OUR video-diffusion model over π³ eval sequences and save, per sequence:
    pred_c2w.npy        (T, 4, 4)  predicted camera-to-world (from predicted raymaps)
    pred_depth_norm.npy (T, H, W)  predicted depth in model-normalized disparity [-1, 1]
    frames.json         {seq, frame_indices, n_src, n_used}
into  eval_pi3/preds/<model>/<dataset>/<seq>/ .

Reuses the VALIDATED model + RGB->ray+depth sampler from the project verbatim:
  - load_model / model_cls exactly as in
    scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py
  - the concat model's own `sample_seq(..., hist_indexes=arange(T_lat))`
    (pins all RGB latent frames as history; denoises ray+depth from noise)
  - cameras via datasets.aria.raymap_to_camera (returns c2w extrinsics).

Per-sequence frame counts differ (M%4==2), so the model's n_frames / lat_t /
max_tokens are reconfigured per sequence before sampling (batch_size=1).
"""
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import argparse
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

from datasets.pi3seq import Pi3SeqDataset
from algorithms.wan.wan_t2v_ray_depth_mot import WanTextToVideoRayDepthMoT
from algorithms.wan.wan_t2v_ray_depth_mot_concat import WanTextToVideoRayDepthMoTConcat
from algorithms.wan.wan_t2v_ray_depth_mot_concat_5b import WanTextToVideoRayDepthMoTConcat5B
from algorithms.wan.wan_t2v_ray_depth_mot_concat_rebuttal import WanTextToVideoRayDepthMoTConcatRebuttal
from benchmark_components import (
    attach_finetuned_depth_vae_v2,
    recover_predicted_cameras,
    restore_rng_state,
    run_verified_sparse_bundle_adjustment,
    save_rng_state,
)

DEVICE = "cuda"
NORM_META: dict = {}

_ALGO_CLASSES = {
    "wan_t2v_ray_depth_mot": WanTextToVideoRayDepthMoT,
    "wan_t2v_ray_depth_mot_concat": WanTextToVideoRayDepthMoTConcat,
    "wan_t2v_ray_depth_mot_concat_5b": WanTextToVideoRayDepthMoTConcat5B,
    # the normalization-ablation arms were trained with the rebuttal class
    "wan_t2v_ray_depth_mot_concat_rebuttal": WanTextToVideoRayDepthMoTConcatRebuttal,
}


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
        f"dataset.n_frames={args.frame_cap}",
        f"dataset.frame_cap={args.frame_cap}",
        f"dataset.height={args.height}",
        f"dataset.width={args.width}",
        f"dataset.raymap_height={args.height // 8}",
        f"dataset.raymap_width={args.width // 8}",
        "algorithm.model.compile=false",
        "algorithm.vae.compile=false",
        "algorithm.text_encoder.compile=false",
        f"++dataset.frame_mode={args.frame_mode}",
    ]
    if args.scale_preserving:
        overrides.append("++dataset.scale_preserving=true")
    if getattr(args, "scale_mode", None):
        overrides.append(f"++dataset.scale_mode={args.scale_mode}")
    if getattr(args, "global_metric_scale", None) is not None:
        overrides.append(f"++dataset.global_metric_scale={args.global_metric_scale}")
    for _k in ("parallax_kappa", "parallax_depth_map", "parallax_log_alpha",
               "parallax_ray_encoding", "parallax_mu", "geom_range_alpha"):
        _v = getattr(args, _k, None)
        if _v is not None:
            overrides.append(f"++dataset.{_k}={_v}")
    if getattr(args, "seqs", None):
        # Shard support: restrict Pi3SeqDataset to this seq subset. Multiple instances with
        # disjoint --seqs write to the SAME preds dir; score once after all shards finish.
        seqlist = "[" + ",".join(args.seqs.split(",")) + "]"
        overrides.append(f"++dataset.seqs={seqlist}")
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):  # [standardized_eval PATCH]
        cfg = compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)
    return cfg


def load_model(algo_cfg, model_cls, depth_vae_ckpt=None):
    """Verbatim from the inference script (env-gated bf16 DiT for the 5B)."""
    model = model_cls(algo_cfg)
    model.configure_model()
    model.depth_vae_metadata = None
    if depth_vae_ckpt is not None:
        model.depth_vae_metadata = attach_finetuned_depth_vae_v2(
            model, algo_cfg, depth_vae_ckpt
        )
        print(
            f"INFO: attached Wan2.2 depth VAE v2 from {depth_vae_ckpt}",
            flush=True,
        )
    model = model.eval().to(DEVICE)
    if os.environ.get("INFER_DIT_BF16", "0") == "1":
        model.model = model.model.to(torch.bfloat16)
        print("INFO: cast DiT (model.model) -> bf16", flush=True)
    model.vae_scale = [model.vae_mean, model.vae_inv_std]
    return model


def set_model_frames(model, m: int):
    """Reconfigure model for an m-frame sequence (mirrors the algo __init__)."""
    vs0 = model.vae_stride[0]
    df = model.diffusion_forcing
    if df.enabled and df.mode == "ray_depth_prediction_firstlast":
        if m <= 1:
            per_modality_t = m
        else:
            interior = m - 2
            per_modality_t = 2 + (interior + vs0 - 1) // vs0
        lat_t = 4 * per_modality_t
    else:
        lat_t = 4 * (1 + (m - 1) // vs0)
    model.n_frames = m
    model.lat_t = lat_t
    # max_tokens factor is algo-specific (mirrors each algo's own __init__):
    #   concat models (WanTextToVideoRayDepthMoTConcat[/5B]): (3*lat_t//4) — modalities packed
    #     along channels so only 3/4 of the latent time carries query tokens;
    #   non-concat MoT (WanTextToVideoRayDepthMoT, the "paper" model): full lat_t.
    is_concat = "Concat" in type(model).__name__
    t_factor = (3 * lat_t // 4) if is_concat else lat_t
    model.max_tokens = (
        t_factor
        * (model.lat_h // model.patch_size[1])
        * (model.lat_w // model.patch_size[2])
    )
    return lat_t


@torch.no_grad()
@torch.autocast(DEVICE, dtype=torch.bfloat16)
def run_one(
    model,
    batch,
    sample_steps,
    *,
    shared_camera_intrinsics=False,
    bundle_adjust=False,
    norm_meta=None,
):
    m = batch["videos"].shape[1]
    lat_t = set_model_frames(model, m)
    T_lat = lat_t // 4
    hist_indexes = torch.arange(0, T_lat)
    pbar = tqdm(range(sample_steps), desc="sampling", leave=False)
    batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    video = model.sample_seq(batch, hist_indexes=hist_indexes, pbar=pbar).squeeze(0)  # (4T, C, H, W)
    T = len(video) // 4
    assert T == m, f"expected {m} pixel frames, got {T}"
    pred_ray_d = video[T:2 * T]
    pred_ray_m = video[2 * T:3 * T]
    pred_depth = video[3 * T:]
    pred_raymaps = torch.cat([pred_ray_d, pred_ray_m], dim=1).float()        # (T, 6, rh, rw)
    cameras = recover_predicted_cameras(
        pred_raymaps,
        shared_intrinsics=shared_camera_intrinsics,
        model_native_units=bundle_adjust,
        norm_meta=norm_meta,
    )
    pred_depth_norm = pred_depth.float().mean(dim=1).cpu().numpy()           # (T, H, W) in [-1,1]
    confidence = getattr(model, "last_depth_confidence", None)
    if confidence is not None:
        confidence = confidence.float().mean(dim=1).squeeze(0).cpu().numpy()
    base_depth = getattr(model, "last_old_depth", None)
    if base_depth is not None:
        base_depth = base_depth.float().mean(dim=1).squeeze(0).cpu().numpy()
    return (
        cameras,
        pred_depth_norm.astype(np.float32),
        None if confidence is None else confidence.astype(np.float32),
        None if base_depth is None else base_depth.astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--algorithm", default="wan_t2v_ray_depth_mot_concat",
                    choices=list(_ALGO_CLASSES))
    ap.add_argument("--model", required=True, help="label for output dir, e.g. ours_best_1p3b")
    ap.add_argument("--dataset", default="sintel", help="label for output dir")
    ap.add_argument("--dataset_cfg", default="pi3seq")
    ap.add_argument("--output_dir", default=str(REPO / "eval_pi3" / "preds"))
    ap.add_argument("--sample_steps", type=int, default=40)
    ap.add_argument("--hist_guidance", type=float, default=1.0)
    ap.add_argument("--lang_guidance", type=float, default=0.0)
    ap.add_argument("--frame_cap", type=int, default=50)
    ap.add_argument("--frame_mode", default="uniform", choices=["uniform", "contig_first", "contig_middle", "contig_last"],
                    help="uniform=Protocol A (default, UNCHANGED); contig_first=Protocol B")
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--scale_preserving", action="store_true")
    ap.add_argument("--seqs", default=None, help="comma-separated seq subset (sharding); disjoint shards write same preds dir, score once")
    ap.add_argument("--scale_mode", default=None, help="e.g. global_metric (D); leave unset for per-clip (E)")
    ap.add_argument("--global_metric_scale", type=float, default=None, help="metres divisor for scale_mode=global_metric (D: 10.0)")
    # Normalization-ablation arms. scale_mode also accepts vggt|vggt_omega|pi3|da3|genception
    # (datasets/_geometry_builder.py); parallax_* carry scheme C's settings. These are forwarded
    # to the dataset config AND stamped into norm_meta.json so scoring inverts the right map.
    ap.add_argument("--parallax_kappa", type=float, default=None)
    ap.add_argument("--parallax_depth_map", default=None)
    ap.add_argument("--parallax_log_alpha", type=float, default=None)
    ap.add_argument("--parallax_ray_encoding", default=None)
    ap.add_argument("--parallax_mu", type=float, default=None)
    ap.add_argument("--geom_range_alpha", type=float, default=None)
    ap.add_argument(
        "--depth_vae_ckpt",
        default=None,
        help="optional fine-tuned Wan2.2/5B depth VAE v2 checkpoint",
    )
    ap.add_argument(
        "--shared_camera_intrinsics",
        action="store_true",
        help="fit one prediction-only intrinsic matrix per clip",
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
    args = ap.parse_args()
    # SINGLE SOURCE OF TRUTH for the training normalization. Used for (a) arm-aware camera
    # recovery at inference and (b) the norm_meta.json stamped into every preds dir for the
    # scorer. Sharing one dict is what guarantees the decode at inference and the decode at
    # scoring cannot drift apart.
    global NORM_META
    NORM_META = {k: getattr(args, k, None) for k in (
        "scale_mode", "global_metric_scale", "parallax_kappa", "parallax_depth_map",
        "parallax_log_alpha", "parallax_ray_encoding", "parallax_mu", "geom_range_alpha")}

    if args.depth_vae_ckpt and args.algorithm != "wan_t2v_ray_depth_mot_concat_5b":
        ap.error("--depth_vae_ckpt is supported here only for the 5B/Wan2.2 model")
    if args.bundle_adjust and not args.depth_vae_ckpt:
        ap.error("--bundle_adjust requires --depth_vae_ckpt (verified BA input)")
    if args.bundle_adjust and not args.shared_camera_intrinsics:
        ap.error("--bundle_adjust requires --shared_camera_intrinsics (verified pipeline)")

    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    cfg = build_cfg(args)
    print(f"INFO: loading model {args.algorithm} from {args.ckpt_path}")
    model = load_model(
        cfg.algorithm,
        _ALGO_CLASSES[args.algorithm],
        depth_vae_ckpt=args.depth_vae_ckpt,
    )
    model.hist_guidance = args.hist_guidance
    model.lang_guidance = args.lang_guidance
    model.sample_steps = args.sample_steps

    dataset = Pi3SeqDataset(cfg.dataset, split="test")
    print(f"INFO: {len(dataset)} sequences")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    out_root = Path(args.output_dir) / args.model / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"{args.model}/{args.dataset}"):
        seq = batch["seq_id"][0] if isinstance(batch["seq_id"], (list, tuple)) else batch["seq_id"]
        seq = seq if isinstance(seq, str) else str(seq)
        frame_indices = batch["frame_indices"].squeeze(0).tolist()
        n_src = int(batch["n_src"][0]) if torch.is_tensor(batch["n_src"]) else int(batch["n_src"])

        seq_dir = out_root / seq
        component_path = seq_dir / "component_summary.json"
        rng_state_path = seq_dir / "rng_state_after.pt"
        # Resume/skip: if this seq already has complete preds (from another shard), skip it.
        # Restoring the post-item RNG state keeps later items paired after preemption.
        if (seq_dir / "pred_c2w.npy").exists() and (seq_dir / "pred_depth_norm.npy").exists() \
                and (seq_dir / "frames.json").exists() \
                and component_path.exists() and rng_state_path.exists():
            restore_rng_state(rng_state_path)
            print(f"  {seq}: skip (already done)", flush=True)
            continue

        component_path.unlink(missing_ok=True)
        rng_state_path.unlink(missing_ok=True)

        seq_dir.mkdir(parents=True, exist_ok=True)
        cameras, pred_depth_norm, confidence, base_depth = run_one(
            model,
            batch,
            args.sample_steps,
            shared_camera_intrinsics=args.shared_camera_intrinsics,
            bundle_adjust=args.bundle_adjust,
            norm_meta=NORM_META,
        )
        pred_c2w = cameras.c2w
        ba_summary = None
        if args.bundle_adjust:
            np.save(seq_dir / "pred_c2w_pre_ba.npy", pred_c2w)
            pred_c2w, ba_summary = run_verified_sparse_bundle_adjustment(
                rgb_tchw=batch["videos"].squeeze(0).cpu().numpy(),
                depth_norm=pred_depth_norm,
                intrinsics=cameras.intrinsics,
                c2w=pred_c2w,
                output_dir=seq_dir,
                repo_root=REPO,
            )

        np.save(seq_dir / "pred_c2w.npy", pred_c2w)
        np.save(seq_dir / "pred_intrinsics.npy", cameras.intrinsics)
        np.save(seq_dir / "pred_depth_norm.npy", pred_depth_norm)
        if args.shared_camera_intrinsics or args.bundle_adjust:
            np.save(seq_dir / "pred_c2w_legacy.npy", cameras.legacy_c2w)
        if confidence is not None:
            np.save(seq_dir / "pred_depth_confidence.npy", confidence)
        if base_depth is not None:
            np.save(seq_dir / "pred_depth_base_same_latent.npy", base_depth)
        with open(seq_dir / "frames.json", "w") as f:
            json.dump({
                "seq": seq,
                "frame_indices": frame_indices,
                "n_src": n_src,
                "n_used": len(frame_indices),
            }, f, indent=2)
        # Stamp the TRAINING normalization into the preds dir. `pred_depth_norm.npy` is the raw
        # [-1,1] channel, and inverting it needs the map the model was TRAINED with. The stock
        # scorers assume signed disparity, which is right only for F / global_metric; a log-map
        # arm scored through that inverse yields plausible-but-meaningless numbers with no error.
        # Writing it here (rather than passing a flag to the scorer) makes the preds
        # self-describing, so a scorer can never be run with the wrong inverse by accident.
        with open(seq_dir / "norm_meta.json", "w") as f:
            json.dump(NORM_META, f, indent=2)
        save_rng_state(rng_state_path)
        component_tmp = component_path.with_name(f".{component_path.name}.tmp")
        with open(component_tmp, "w") as f:
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
                f,
                indent=2,
                allow_nan=True,
            )
        os.replace(component_tmp, component_path)
        print(
            f"  {seq}: saved pred_c2w {pred_c2w.shape}, depth {pred_depth_norm.shape}, "
            f"confidence={confidence is not None}, BA={args.bundle_adjust}",
            flush=True,
        )

    print(f"INFO: done -> {out_root}")


if __name__ == "__main__":
    main()

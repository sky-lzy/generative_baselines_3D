"""
V3 — Geo4D depth evaluation on OURS_FINAL eval outputs.

Runs Geo4D on gt_rgb.mp4 from each sample dir, extracts post-optimised depth
maps via scene.get_depthmaps(), and reports Geo4D's *own* depth metrics
(Abs Rel / Sq Rel / RMSE / Log RMSE / δ<1.25 / δ<1.25² / δ<1.25³) using the
vendored geo4d_eval/depth_eval.py with align_with_lad2=True (per-video global
LAD2 scale-shift fit — NOT per-frame).

GT depth is loaded from gt_depth_raw.npz["depth"] * gt_depth_raw.npz["scale"]
(metric meters). Geo4D depth is also metric. depth_evaluation fits one (s,t)
across the whole video to align the two.

(V2 used a [-1,1]→[0,1] GT remap + RANSAC scale-shift in disparity-like space;
the V3 metric is recomputed in metric-depth space to match Geo4D's published
methodology.)
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb

GEO4D_DIR = os.environ.get(
    "GEO4D_DIR",
    "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D",
)

# V3: depth metrics via vendored Geo4D code in generative_baselines/eval_common_v3.py
_GB_DIR = str(Path(__file__).resolve().parents[1])
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from eval_common_v3 import (  # noqa: E402
    DEPTH_KEYS, aggregate, eval_depth_sequence, load_depth_metric_from_npz,
)
from geo4d.depth_utils import match_depth_to_gt  # noqa: E402
from geo4d.path_utils import resolve_geo4d_path  # noqa: E402

METRIC_KEYS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog")


# ---------------------------------------------------------------------------
# GT depth loader  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def read_gt_depth_npz_metric(npz_path: Path) -> np.ndarray:
    """V3: GT depth as metric meters via eval_common_v3.load_depth_metric_from_npz."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing gt_depth_raw.npz: {npz_path}")
    return load_depth_metric_from_npz(npz_path)


# ---------------------------------------------------------------------------
# Depth metrics  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray) -> dict | None:
    """Returns None if fewer than 10 valid pixels (sample is skipped entirely)."""
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"Shape mismatch: pred {pred_depth.shape} vs gt {gt_depth.shape}")

    valid = (gt_depth > 0) & (pred_depth > 0) & np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if valid.sum() < 10:
        return None

    pred = pred_depth[valid].astype(np.float64)
    gt   = gt_depth[valid].astype(np.float64)

    diff     = pred - gt
    diff_log = np.log(pred) - np.log(gt)
    ratio    = np.maximum(pred / gt, gt / pred)

    return {
        "d1":      float((ratio < 1.25     ).mean()),
        "d2":      float((ratio < 1.25 ** 2).mean()),
        "d3":      float((ratio < 1.25 ** 3).mean()),
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "sq_rel":  float(np.mean(diff ** 2 / gt)),
        "rmse":    float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log":float(np.sqrt(np.mean(diff_log ** 2))),
        "log10":   float(np.mean(np.abs(np.log10(pred) - np.log10(gt)))),
        "silog":   float(np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2)),
    }


def fit_scale_shift_ransac(pred: np.ndarray, gt: np.ndarray, max_iters: int = 200, min_inliers: int = 1000) -> tuple[float, float]:
    """Fits scale (s) and shift (t) such that: s * pred + t ~ gt"""
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)

    valid = (gt_flat > 0) & np.isfinite(pred_flat) & np.isfinite(gt_flat)
    pred_valid = pred_flat[valid]
    gt_valid = gt_flat[valid]

    if pred_valid.size < 2:
        return 1.0, 0.0

    rng = np.random.default_rng(0)
    best_inliers = -1
    best_s, best_t = 1.0, 0.0
    best_mse = np.inf
    n_points = pred_valid.size

    for _ in range(max_iters):
        idx = rng.choice(n_points, size=2, replace=False)
        p1, p2 = pred_valid[idx[0]], pred_valid[idx[1]]
        g1, g2 = gt_valid[idx[0]], gt_valid[idx[1]]

        denom = (p1 - p2)
        if denom == 0: continue

        s = (g1 - g2) / denom
        if s <= 0: continue
        t = g1 - s * p1

        estimate = s * pred_valid + t
        residuals = np.abs(estimate - gt_valid)

        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        threshold = mad if mad > 0 else 1e-3

        inliers = residuals <= (3 * threshold)
        num_inliers = np.count_nonzero(inliers)

        if num_inliers < min_inliers:
            continue

        if num_inliers > best_inliers:
            p_in = pred_valid[inliers]
            g_in = gt_valid[inliers]
            A = np.vstack([p_in, np.ones_like(p_in)]).T
            solution, _, _, _ = np.linalg.lstsq(A, g_in, rcond=None)
            s_ls, t_ls = solution[0], solution[1]

            if s_ls > 0:
                mse = np.mean((s_ls * p_in + t_ls - g_in) ** 2)
                if num_inliers > best_inliers or (num_inliers == best_inliers and mse < best_mse):
                    best_inliers = num_inliers
                    best_mse = mse
                    best_s, best_t = float(s_ls), float(t_ls)

    return best_s, best_t


# ---------------------------------------------------------------------------
# Visualisation  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def plot_depth_comparison(
    gt_depth: np.ndarray,
    pred_depth_raw: np.ndarray,
    pred_depth_aligned: np.ndarray,
    metrics_raw: dict,
    metrics_aligned: dict,
    output_path: str,
) -> None:
    """Save GT / raw-pred / aligned-pred side-by-side (first frame only)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = gt_depth[0].min()
    vmax = gt_depth[0].max()

    im0 = axes[0].imshow(gt_depth[0], cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT depth")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_depth_raw[0], cmap="turbo")
    axes[1].set_title(
        f"Pred (raw)\nAbsRel={metrics_raw['abs_rel']:.3f}  d1={metrics_raw['d1']:.3f}"
    )
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred_depth_aligned[0], cmap="turbo", vmin=vmin, vmax=vmax)
    axes[2].set_title(
        f"Pred (aligned)\nAbsRel={metrics_aligned['abs_rel']:.3f}  d1={metrics_aligned['d1']:.3f}"
    )
    plt.colorbar(im2, ax=axes[2])

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accumulate_metrics(stats: dict, metrics: dict) -> dict:
    """Sample-averaged accumulation."""
    n = stats["count"]
    for k in METRIC_KEYS:
        stats[k] = (stats[k] * n + metrics[k]) / (n + 1)
    stats["count"] += 1
    return stats


# ---------------------------------------------------------------------------
# Geo4D model loading  (verbatim from eval_geo4d_pose.py)
# ---------------------------------------------------------------------------

def setup_geo4d_path():
    if GEO4D_DIR not in sys.path:
        sys.path.insert(0, GEO4D_DIR)


def load_geo4d_model(ckpt_path: str, config_path: str, gpu_no: int = 0):
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config
    from scripts.evaluation.test_geo4d import load_model_checkpoint

    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = True
    ckpt_path = str(resolve_geo4d_path(ckpt_path, GEO4D_DIR))
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()

    pointmap_vae = None
    if "vae_path" in config:
        config["vae_path"] = str(resolve_geo4d_path(config["vae_path"], GEO4D_DIR))
        pointmap_vae_config = config.pop("pointmap_vae_config", OmegaConf.create())
        pointmap_vae = instantiate_from_config(pointmap_vae_config).eval().cuda(gpu_no)
        from lvdm.basics import disabled_train
        pointmap_vae.train = disabled_train
        for param in pointmap_vae.parameters():
            param.requires_grad = False
        vae_weights = torch.load(config["vae_path"])["state_dict"]
        new_sd = OrderedDict(
            (k[6:], v) for k, v in vae_weights.items() if k.startswith("model.")
        )
        pointmap_vae.load_state_dict(new_sd, strict=True)

    return model, config, pointmap_vae


# ---------------------------------------------------------------------------
# Geo4D depth inference
# ---------------------------------------------------------------------------

def infer_geo4d_depth(
    model, config, pointmap_vae,
    video_path: str,
    ddim_steps: int = 5,
    height: int = 320,
    width: int = 512,
    gpu_no: int = 0,
) -> np.ndarray:
    """
    Run Geo4D on a single video and return post-optimised depth maps.
    Returns (T, H, W) float32 numpy array in arbitrary metric units.
    """
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from utils.funcs import load_video_batch
    from scripts.evaluation.test_geo4d import (
        image_guided_synthesis, post_optimization,
        get_sky_mask, get_far_away_mask, denormalize_pc_bbox2,
        raymap_to_camera_matrix, decode_pm_confhead,
    )

    seed_everything(123)

    video_frames, fps_list = load_video_batch(
        [video_path],
        frame_stride=1,
        video_size=(height, width),
        video_frames=-1,
    )
    B, C, T, H, W = video_frames.shape

    if T < 16:
        pad = video_frames[:, :, -1:, :, :].expand(-1, -1, 16 - T, -1, -1)
        video_frames = torch.cat([video_frames, pad], dim=2)
        B, C, T, H, W = video_frames.shape

    views = [{"img": video_frames[0, :, i, :, :], "idx": (i,)} for i in range(T)]

    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, 16, H // 8, W // 8]

    videos_all = video_frames.cuda(gpu_no)
    prompts = ["Output a video that assigns each 3D location in the world a consistent color."]

    stride = 4
    slice_list = list(range(0, T - 16 + 1, stride))
    slice_list = [slice(s, s + 16, 1) for s in slice_list]
    if not slice_list or slice_list[-1] != slice(T - 16, T, 1):
        slice_list.append(slice(T - 16, T, 1))

    pred_list = []
    view_list = []
    pnt_valid_mask = torch.ones((T, H, W, 1), device=f"cuda:{gpu_no}") > 0

    for sl in tqdm(slice_list, desc="  Geo4D windows", leave=False):
        videos = videos_all[:, :, sl, :, :].clone()
        view_list.append(views[sl])

        fps = fps_list[0]
        batch_samples = image_guided_synthesis(
            model, prompts, videos, noise_shape,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=0.0,
            unconditional_guidance_scale=1.0,
            cfg_img=None,
            fs=fps,
            text_input=True,
            multiple_cond_cfg=False,
            loop=False,
            interp=False,
            timestep_spacing="uniform_trailing",
            guidance_rescale=0.7,
            pointmap_vae=pointmap_vae,
        )
        assert batch_samples.shape[1] == 1
        batch_samples = batch_samples[:, 0]

        raymap = crossmap = inverse_depthmap = traj = None
        if model.modality == "pc_ray_cross_depth":
            raymap = batch_samples[:, 4:7]
            crossmap = batch_samples[:, 7:10]
            traj = raymap_to_camera_matrix(raymap, crossmap)
            raymap = rearrange(raymap, "b c t h w -> (b t) c h w")
            raymap = rearrange(raymap, "t c h w -> t h w c")
            crossmap = rearrange(crossmap, "b c t h w -> (b t) c h w")
            crossmap = rearrange(crossmap, "t c h w -> t h w c")
            inverse_depthmap = batch_samples[:, 10:11]
            inverse_depthmap = rearrange(inverse_depthmap, "b c t h w -> (b t) c h w")
            inverse_depthmap = rearrange(inverse_depthmap, "t c h w -> t h w c")
            inverse_depthmap = (inverse_depthmap + 1.0) / 2.0

        batch_samples = batch_samples[:, :4]
        x_recon = rearrange(batch_samples, "b c t h w -> (b t) c h w")
        confidence = torch.nn.Softplus()(x_recon[:, [-1], :, :])
        confidence = rearrange(confidence, "t c h w -> t h w c")
        if pointmap_vae is None:
            confidence = torch.ones_like(confidence)

        x_recon = x_recon[:, :-1, :, :]
        x_recon_reshape = rearrange(x_recon, "t c h w -> t h w c")
        invalid_pts = get_sky_mask(x_recon_reshape, sky_value=1.05, eps=0.35)
        far_away_mask = get_far_away_mask(x_recon_reshape, far_away_value=1.99)
        invalid_pts = invalid_pts | far_away_mask
        confidence[invalid_pts] = 999.0
        pnt_valid_mask[sl] = pnt_valid_mask[sl] * (~invalid_pts)
        inverse_confidence = 1 / confidence
        inverse_confidence[invalid_pts] = 0.0

        x_recon = rearrange(x_recon, "t c h w -> t h w c")
        x_recon = denormalize_pc_bbox2(x_recon, alpha=2.0, beta=2.0)

        pred_pts = {"pts3d": x_recon, "conf": inverse_confidence}
        if inverse_depthmap is not None:
            pred_pts["inverse_depthmap"] = inverse_depthmap
        if traj is not None:
            pred_pts["traj"] = traj
        pred_list.append(pred_pts)

    scene = post_optimization(
        view_list, pred_list, config.postprocess,
        conf_optimize=True, init_method="group", lr=0.03,
        opt_raydir=False,
    )

    # Extract post-optimised depth: list of T tensors (H, W) in metric units
    depthmaps = scene.get_depthmaps()
    depth = torch.stack(depthmaps, dim=0).detach().cpu().float().numpy()  # (T, H, W)
    return depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Output dir from inference_single_gpu_rgb_to_ray_depth_aria_eval.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(GEO4D_DIR, "checkpoints/geo4d/model.ckpt"))
    parser.add_argument("--config", type=str,
                        default=os.path.join(GEO4D_DIR, "configs/inference_geo4d.yaml"))
    parser.add_argument("--ddim_steps", type=int, default=5)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun Geo4D even if cached depth exists")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset name (controls Geo4D max_depth clip)")
    parser.add_argument("--custom_run_name", type=str, default="geo4d_depth_eval_v3")
    args = parser.parse_args()

    setup_geo4d_path()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — will be very slow or broken")

    print(f"Loading Geo4D model from {args.ckpt_path} ...")
    model, config, pointmap_vae = load_geo4d_model(args.ckpt_path, args.config, gpu_no=0)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name.startswith("sample_")
    ])

    per_seq: list[dict] = []
    count = 0
    for sample_dir in tqdm(sample_dirs, desc=f"Geo4D depth v3 ({args.dataset})"):
        if args.max_samples is not None and count >= args.max_samples:
            break

        gt_rgb_path       = sample_dir / "gt_rgb.mp4"
        gt_depth_npz_path = sample_dir / "gt_depth_raw.npz"
        if not gt_rgb_path.exists():
            raise FileNotFoundError(f"{gt_rgb_path} not found — incomplete inference dir")

        gt_depth = read_gt_depth_npz_metric(gt_depth_npz_path)  # (T,H,W) metric meters

        sample_out   = output_dir / sample_dir.name
        cached_depth = sample_out / "pred_depth_geo4d.npz"
        if cached_depth.exists() and not args.rerun:
            pred_depth = np.load(cached_depth)["depth"]
            matched_depth = match_depth_to_gt(pred_depth, gt_depth)
            if matched_depth.shape != pred_depth.shape:
                pred_depth = matched_depth
                np.savez_compressed(cached_depth, depth=pred_depth)
        else:
            try:
                pred_depth = infer_geo4d_depth(
                    model, config, pointmap_vae,
                    video_path=str(gt_rgb_path),
                    ddim_steps=args.ddim_steps,
                    height=args.height,
                    width=args.width,
                    gpu_no=0,
                )
            except Exception as e:
                print(f"Error running Geo4D on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue
            pred_depth = match_depth_to_gt(pred_depth, gt_depth)
            sample_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached_depth, depth=pred_depth)

        pred_depth = match_depth_to_gt(pred_depth, gt_depth)
        gt_depth_eval = gt_depth[: pred_depth.shape[0]]

        try:
            metrics = eval_depth_sequence(
                pred_depth_THW=pred_depth,
                gt_depth_THW=gt_depth_eval,
                dataset=args.dataset,
                normalize_unit_per_video=True,
            )
        except Exception as e:
            print(f"Error evaluating {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        wandb.log({f"sample/{k}": metrics[k] for k in DEPTH_KEYS})
        per_seq.append(metrics)
        count += 1

    final = aggregate(per_seq, DEPTH_KEYS)
    print(f"\nFinal (n={final['n_samples']})  "
          + "  ".join(f"{k}={final[k]:.4f}" for k in DEPTH_KEYS))
    wandb.log({f"final/{k}": final[k] for k in DEPTH_KEYS})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()

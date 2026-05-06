"""
V3 — RayDiffusion pose eval against OURS_FINAL eval outputs.

Same pytorch3d-camera→w2c conversion plumbing as V2 (eval_raydiffusion_pose.py),
but pose metrics are computed via Geo4D's vendored vo_eval (ATE / RPE_trans /
RPE_rot, evo Sim(3)-Umeyama). Aria correction is dropped — fixed at inference
time inside aria.py for V3.

pytorch3d camera convention reminder:
    PerspectiveCameras stores (R, T) with p_cam = p_world @ R + T
    → standard column-vector form: w2c rotation = R.T, translation = T.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

RAYDIFFUSION_DIR = os.environ.get(
    "RAYDIFFUSION_DIR",
    str(Path(__file__).resolve().parent),
)
sys.path.insert(0, RAYDIFFUSION_DIR)

# V3: pose metrics via vendored Geo4D code in generative_baselines/eval_common_v3.py
_GB_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines"
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from eval_common_v3 import POSE_KEYS, aggregate, eval_pose_sequence  # noqa: E402

MAX_FRAMES_RAYDIFF = 8   # RayDiffusion processes at most 8 images simultaneously


# ---------------------------------------------------------------------------
# Pose evaluation (verbatim from eval_ray_mot_pose.py)
# ---------------------------------------------------------------------------

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret = torch.where(positive_mask, torch.sqrt(x.clamp(min=0)), ret)
    return ret


def mat_to_quat(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
    ], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    import torch.nn.functional as F
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    return torch.where(out[..., 3:4] < 0, -out, out)


def build_pair_index(N):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    return i1_, i2_


def closed_form_inverse_se3(se3):
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_T = R.transpose(1, 2)
    inv = torch.eye(4)[None].repeat(len(R), 1, 1).to(R.dtype).to(R.device)
    inv[:, :3, :3] = R_T
    inv[:, :3, 3:] = -torch.bmm(R_T, T)
    return inv


def align_to_first_camera(poses):
    first_inv = closed_form_inverse_se3(poses[0][None])
    return torch.matmul(poses, first_inv)


def compare_translation_by_angle(t_gt, t, eps=1e-15):
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)
    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6
    return err_t


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt   = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, ambiguity=True):
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def se3_to_relative_pose_error(pred_se3, gt_se3, N):
    i1, i2 = build_pair_index(N)
    rel_gt   = closed_form_inverse_se3(gt_se3[i1]).bmm(gt_se3[i2])
    rel_pred = closed_form_inverse_se3(pred_se3[i1]).bmm(pred_se3[i2])
    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_err = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return r_err, t_err


def calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized = histogram.astype(float) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized))), normalized


def compute_pose(pred_w2c, gt_w2c):
    pred_w2c = align_to_first_camera(pred_w2c)
    gt_w2c   = align_to_first_camera(gt_w2c)
    r_err, t_err = se3_to_relative_pose_error(pred_w2c, gt_w2c, len(pred_w2c))
    r_err = r_err.cpu().numpy()
    t_err = t_err.cpu().numpy()
    out = {}
    out["auc30"], _ = calculate_auc_np(r_err, t_err, 30)
    out["auc15"], _ = calculate_auc_np(r_err, t_err, 15)
    out["auc05"], _ = calculate_auc_np(r_err, t_err, 5)
    out["auc03"], _ = calculate_auc_np(r_err, t_err, 3)
    return out


def _procrustes_align(source, target):
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    U, _, Vt = np.linalg.svd(t_c.T @ s_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    ss = (s_c ** 2).sum()
    scale = (t_c * (s_c @ R.T)).sum() / ss if ss > 0 else 1.0
    return scale * (source - mu_s) @ R.T + mu_t


def plot_camera_trajectory(pred_c2w, gt_c2w, output_path):
    pred_c = pred_c2w[:, :3, 3]
    gt_c   = gt_c2w[:, :3, 3]
    pred_aligned = _procrustes_align(pred_c, gt_c)
    T = len(pred_c)
    colors = [plt.get_cmap("hsv")(i / max(T - 1, 1)) for i in range(T)]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(T - 1):
        ax.plot(pred_aligned[i:i+2, 0], pred_aligned[i:i+2, 1], pred_aligned[i:i+2, 2],
                color=colors[i], linewidth=2)
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
               c=colors, s=30, zorder=5, label="RayDiff pred (aligned)")
    ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], color="black", linewidth=1.5, linestyle="--")
    ax.scatter(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], color="black", s=20, zorder=4, label="GT")
    ax.legend()
    ax.set_title("Camera Trajectories — RayDiffusion pred aligned to GT (Procrustes)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RayDiffusion inference
# ---------------------------------------------------------------------------

def load_raydiffusion_model(model_dir: str, device: torch.device):
    from ray_diffusion.inference.load_model import load_model
    model, cfg = load_model(model_dir, device=device)
    return model, cfg


def pytorch3d_cameras_to_c2w(R_p3d: np.ndarray, T_p3d: np.ndarray) -> np.ndarray:
    """
    Convert pytorch3d PerspectiveCameras (R, T) to c2w (4, 4) matrices.

    Pytorch3d convention: p_cam = p_world @ R + T  (row vectors)
    Standard w2c:         p_cam = R_w2c @ p_world + t_w2c
    Therefore:            R_w2c = R_p3d.T, t_w2c = T_p3d
    """
    N = R_p3d.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        R_w2c = R_p3d[i].T   # (3, 3)
        t_w2c = T_p3d[i]     # (3,)
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3]  = t_w2c
        c2w[i] = np.linalg.inv(w2c)
    return c2w


def run_raydiffusion_on_frames(model, cfg, frames_hwc: np.ndarray,
                                device: torch.device,
                                max_frames: int = MAX_FRAMES_RAYDIFF) -> np.ndarray:
    """
    Run RayDiffusion on a set of RGB frames.
    Returns (pred_c2w, indices) where pred_c2w is (K, 4, 4) for the K subsampled
    frames and indices is the array of original frame indices used.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF
    from ray_diffusion.dataset import CustomDataset
    from ray_diffusion.inference.predict import predict_cameras

    T = len(frames_hwc)
    frames_to_use = min(T, max_frames)

    # Sample evenly-spaced frames if we have more than max_frames
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        frames_hwc = frames_hwc[indices]
        frames_to_use = max_frames
    else:
        indices = np.arange(T)

    # Write frames to a temp directory for CustomDataset
    h, w = frames_hwc[0].shape[:2]
    full_bboxes = [[0, 0, w, h]] * frames_to_use

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames_hwc):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"{i:05d}.png"))

        dataset = CustomDataset(
            image_dir=tmpdir,
            mask_dir=None,
            bboxes=full_bboxes,
            mask_images=False,
        )
        batch = dataset.get_data(ids=np.arange(frames_to_use))
        images     = batch["image"].to(device)            # (N, C, H, W)
        crop_params = batch["crop_params"].to(device)     # (N, 4)

    is_regression = cfg.training.regression
    if is_regression:
        pred = predict_cameras(
            model=model, images=images, device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            use_regression=True,
        )
        predicted_cameras = pred[0]
    else:
        pred = predict_cameras(
            model=model, images=images, device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            additional_timesteps=(70,),
            rescale_noise="zero",
            use_regression=False,
            max_num_images=None if frames_to_use <= 8 else 8,
            pbar=False,
        )
        predicted_cameras = pred[1][0]

    R_p3d = predicted_cameras.R.cpu().numpy()  # (N, 3, 3)
    T_p3d = predicted_cameras.T.cpu().numpy()  # (N, 3)

    pred_c2w_subsampled = pytorch3d_cameras_to_c2w(R_p3d, T_p3d)  # (N, 4, 4)

    # Return only the subsampled predictions + which original indices were used.
    # Callers should evaluate only on these frames so that NN-filled frames do
    # not inflate AUC metrics.
    return pred_c2w_subsampled, indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AUC_KEYS = ("auc03", "auc05", "auc15", "auc30")  # legacy V2 keys, unused in V3

# ---------------------------------------------------------------------------
# Aria coordinate-frame correction
# ---------------------------------------------------------------------------
# RayDiffusion was trained on CO3D and predicts cameras in OpenCV convention
# for the frames it receives.  Aria GT cameras (gt_cameras.npz) are stored in
# the Aria *device* frame.  Two rotations bridge the gap:
#   R_hardware  – Aria hardware calibration (device → camera, from Aria API)
#   R_image_roll – undo the 90° CW image rotation baked into Aria gt_rgb.mp4
#                  by the dataset pipeline (rot90(k=3) + resize)
# Applying T_aria_fix = R_hardware @ R_image_roll as a left-multiply on every
# predicted w2c converts from "OpenCV pred on rotated image" to "Aria device
# frame", making it directly comparable to GT.
# (Identical to eval_da3_pose_debug_alt.py with --procrustes_align_pred)

R_ARIA_HARDWARE = np.array([
    [ 0.99606003, -0.04388682,  0.07706079],
    [ 0.08210934,  0.78468796, -0.61442889],
    [-0.03350334,  0.61833547,  0.78519983],
], dtype=np.float64)

R_ARIA_IMAGE_ROLL = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)

def _build_aria_fix_matrix() -> np.ndarray:
    R_total = R_ARIA_HARDWARE @ R_ARIA_IMAGE_ROLL
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_total
    return T

def apply_aria_correction(pred_w2c: np.ndarray) -> np.ndarray:
    """Left-multiply each predicted w2c by the Aria coordinate-frame fix."""
    T_fix = _build_aria_fix_matrix()
    return np.array([T_fix @ w2c for w2c in pred_w2c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Sample dir with gt_rgb.mp4 + gt_cameras.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir",
                        default=os.path.join(RAYDIFFUSION_DIR, "models/co3d_diffusion"),
                        help="Path to RayDiffusion model directory")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_RAYDIFF,
                        help="Max frames to pass to RayDiffusion (default 8)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun RayDiffusion even if pred_cameras.npz already exists")
    parser.add_argument("--custom_run_name", default="raydiffusion_pose_eval_v3")
    parser.add_argument("--dataset", default="re10k", choices=["re10k", "aria"],
                        help="Kept for CLI compat with V2; aria correction is "
                             "no longer applied at eval (fixed at inference time "
                             "in aria.py for V3).")
    args = parser.parse_args()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device(args.device)
    print(f"Loading RayDiffusion model from {args.model_dir} …")
    model, cfg = load_raydiffusion_model(args.model_dir, device)
    model.eval()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    per_seq: list[dict] = []
    count = 0

    for sample_dir in tqdm(sample_dirs, desc="RayDiffusion pose eval v3"):
        if args.max_samples is not None and count >= args.max_samples:
            break

        gt_rgb_path     = sample_dir / "gt_rgb.mp4"
        gt_cameras_path = sample_dir / "gt_cameras.npz"

        if not gt_rgb_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_rgb.mp4")
            continue
        if not gt_cameras_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_cameras.npz")
            continue

        gt_data = np.load(gt_cameras_path)
        gt_c2w  = gt_data["extrinsics"].astype(np.float64)   # (T, 4, 4) c2w
        gt_w2c  = np.linalg.inv(gt_c2w)

        frames = np.stack(imageio.mimread(str(gt_rgb_path), memtest=False))  # (T, H, W, 3) uint8

        n_gt = len(gt_c2w)
        # Match frames count to cameras count
        if len(frames) > n_gt:
            frames = frames[:n_gt]
        elif len(frames) < n_gt:
            gt_w2c = gt_w2c[:len(frames)]
            gt_c2w = gt_c2w[:len(frames)]
            n_gt = len(frames)

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        cached_pred = sample_out / "pred_cameras.npz"
        if cached_pred.exists() and not args.rerun:
            cache = np.load(cached_pred)
            pred_c2w = cache["extrinsics"].astype(np.float64)
            sub_indices = cache["frame_indices"].astype(np.int64)
        else:
            try:
                pred_c2w, sub_indices = run_raydiffusion_on_frames(model, cfg, frames, device, args.max_frames)
            except Exception as e:
                print(f"Error running RayDiffusion on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue
            np.savez_compressed(str(cached_pred),
                                extrinsics=pred_c2w, frame_indices=sub_indices)

        # Evaluate only on the subsampled frames RayDiffusion produced.
        sub_indices = sub_indices[:len(pred_c2w)]
        gt_c2w_eval = gt_c2w[sub_indices]   # (K, 4, 4)

        try:
            metrics = eval_pose_sequence(
                pred_c2w=pred_c2w,
                gt_c2w=gt_c2w_eval,
                seq=sample_dir.name,
                save_dir=sample_out,
            )
        except Exception as e:
            print(f"Error evaluating {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        with open(sample_out / "pose_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        plot_camera_trajectory(pred_c2w, gt_c2w_eval, str(sample_out / "camera_trajectory.png"))
        wandb.log({f"sample/{k}": metrics[k] for k in POSE_KEYS})
        per_seq.append(metrics)
        count += 1

    final = aggregate(per_seq, POSE_KEYS)
    print(f"\nFinal (n={final['n_samples']})  "
          + "  ".join(f"{k}={final[k]:.4f}" for k in POSE_KEYS))
    wandb.log({f"final/{k}": final[k] for k in POSE_KEYS})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()

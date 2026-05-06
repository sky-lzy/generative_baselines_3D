import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio

from camera_utils import (
    rescale_intrinsics_to_image,
    resolve_seva_frame_window,
    resolve_seva_target_size,
)

# Add SVC to path to avoid installation
sys.path.insert(0, "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/stable-virtual-camera")

# ---------------------------------------------------------------------------
# Aria coordinate-frame correction (same derivation as pose eval scripts)
# c2w_opencv = c2w_aria @ T_fix  where T_fix = R_hardware @ R_image_roll
# ---------------------------------------------------------------------------
_R_ARIA_HARDWARE = np.array([
    [ 0.99606003, -0.04388682,  0.07706079],
    [ 0.08210934,  0.78468796, -0.61442889],
    [-0.03350334,  0.61833547,  0.78519983],
], dtype=np.float64)
_R_ARIA_IMAGE_ROLL = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)

def _aria_fix() -> np.ndarray:
    R = _R_ARIA_HARDWARE @ _R_ARIA_IMAGE_ROLL
    T = np.eye(4, dtype=np.float64); T[:3, :3] = R
    return T

def aria_c2w_to_opencv(c2w: np.ndarray) -> np.ndarray:
    """Convert (T,4,4) c2w from Aria device frame to OpenCV convention."""
    return c2w @ _aria_fix()

try:
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model
    from seva.eval import run_one_scene
except ImportError as e:
    print(f"Failed to import SVC modules: {e}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SVC on Phase 1 NVS outputs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Phase 1 eval outputs (e.g. ignore/eval_outputs/nvs_re10k_on_re10k)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--model_version", type=float, default=1.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Explicitly resize frames to (H, W); overrides --l_short.")
    parser.add_argument("--l_short", type=int, default=576,
                        help="Aspect-preserving SEVA resize: set shortest side to this "
                             "value and round to stride 64. Use 0 to keep native size.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-virtual-camera",
                        help="Hugging Face repo id or local directory containing SEVA weights.")
    parser.add_argument("--weight_name", type=str, default="model.safetensors",
                        help="SEVA weight filename when loading from a local checkpoint directory.")
    parser.add_argument("--intrinsics_scale", default="auto",
                        help="'auto', 'none', a scalar, or 'sx,sy'. Use this when "
                             "gt_cameras.npz intrinsics are stored at ray-map scale "
                             "instead of RGB pixel scale.")
    parser.add_argument("--frame_window", default="21",
                        help="SEVA per-forward frame window T. Official default is 21. "
                             "Use 'full' to process the whole clip in one forward.")
    parser.add_argument("--camera_scale", type=float, default=2.0,
                        help="SEVA translation normalization scale.")
    parser.add_argument("--dataset", default="re10k", choices=["re10k", "aria"],
                        help="'aria' converts GT cameras from Aria device frame to OpenCV "
                             "convention before passing to SEVA.")
    return parser.parse_args()

def compute_metrics(pred_rgb, gt_rgb):
    import lpips
    from skimage.metrics import structural_similarity as ssim
    
    lpips_fn = lpips.LPIPS(net="alex").to(pred_rgb.device).eval()
    
    # pred_rgb, gt_rgb: [T, C, H, W] in [-1, 1] → convert to [0, 1] for standard PSNR
    pred_01 = (pred_rgb * 0.5 + 0.5).clamp(0, 1)
    gt_01   = (gt_rgb   * 0.5 + 0.5).clamp(0, 1)
    mse = torch.mean((pred_01 - gt_01) ** 2, dim=[1, 2, 3])
    psnr = -10 * torch.log10(mse)
    
    lpips_vals = []
    ssim_vals = []
    for i in range(len(pred_rgb)):
        lpips_val = lpips_fn(pred_rgb[i:i+1], gt_rgb[i:i+1]).item()
        lpips_vals.append(lpips_val)
        
        p_np = (pred_rgb[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        g_np = (gt_rgb[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        s_val = ssim(p_np, g_np, data_range=1.0, channel_axis=2)
        ssim_vals.append(s_val)
        
    return psnr.cpu().numpy().tolist(), lpips_vals, ssim_vals

def unnormalize_video(video: torch.Tensor) -> torch.Tensor:
    return (video * 0.5 + 0.5).clamp(0, 1) * 255.0

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading SVC model v{args.model_version} on {args.device}...")
    ae = AutoEncoder(chunk_size=1).to(args.device)
    conditioner = CLIPConditioner().to(args.device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=args.device)
    
    model = load_model(
        model_version=args.model_version,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        weight_name=args.weight_name,
        device="cpu",
    ).eval()
    model = SGMWrapper(model).to(args.device)
    
    input_dir = Path(args.input_dir)
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    if not sample_dirs:
        print(f"No sample directories found in {args.input_dir}")
        sys.exit(0)
        
    print(f"Found {len(sample_dirs)} samples to evaluate.")
    
    all_psnr, all_lpips, all_ssim = [], [], []
    
    for i, sample_dir in enumerate(tqdm(sample_dirs[:args.max_samples], desc="Evaluating SVC")):
        gt_video_path = sample_dir / "gt_rgb.mp4"
        gt_cameras_path = sample_dir / "gt_cameras.npz"
        
        if not gt_video_path.exists() or not gt_cameras_path.exists():
            print(f"Skipping {sample_dir.name} (missing gt_rgb.mp4 or gt_cameras.npz)")
            continue
            
        # Read GT video
        gt_frames_np = np.array(iio.imread(str(gt_video_path))) # [T, H, W, 3] uint8
        T, H, W, _ = gt_frames_np.shape
        
        videos = torch.from_numpy(gt_frames_np).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        videos = videos.unsqueeze(0).to(args.device) # [1, T, 3, H, W]
        
        # Read GT cameras
        cameras = np.load(str(gt_cameras_path))
        c2w_np = cameras["extrinsics"].astype(np.float64)   # (T, 4, 4)
        # Convert from Aria device frame → OpenCV for SEVA conditioning
        if args.dataset == "aria":
            c2w_np = aria_c2w_to_opencv(c2w_np)
        c2ws = torch.from_numpy(c2w_np).float()[:, :3, :4]  # [T, 3, 4]
        K_np, K_scale = rescale_intrinsics_to_image(
            cameras["intrinsics"].astype(np.float64),
            image_hw=(H, W),
            intrinsics_scale=args.intrinsics_scale,
        )
        if K_scale != (1.0, 1.0):
            print(f"{sample_dir.name}: scaled intrinsics to RGB pixels by sx={K_scale[0]:g}, sy={K_scale[1]:g}")
        Ks = torch.from_numpy(K_np).float() # [T, 3, 3]
        
        # Ensure T matches between video and cameras
        T = min(T, len(c2ws))
        videos = videos[:, :T]
        gt_frames_np = gt_frames_np[:T]
        c2ws = c2ws[:T]
        Ks = Ks[:T]
        
        if args.resize_hw is not None:
            resize_hw = (args.resize_hw[0], args.resize_hw[1])
        else:
            resize_hw = None
        target_h, target_w = resolve_seva_target_size(
            image_hw=(H, W),
            resize_hw=resize_hw,
            l_short=args.l_short,
        )
            
        if target_h != H or target_w != W:
            videos = torch.nn.functional.interpolate(
                videos[0], size=(target_h, target_w), mode="bilinear", align_corners=False
            ).unsqueeze(0)
            Ks[:, 0, :] *= target_w / W
            Ks[:, 1, :] *= target_h / H
            H, W = target_h, target_w
            gt_frames_np = (videos[0].permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255.0
            gt_frames_np = gt_frames_np.astype(np.uint8)
        
        out_sample_dir = os.path.join(args.output_dir, sample_dir.name)
        os.makedirs(out_sample_dir, exist_ok=True)
            
        # We condition on first and last frames
        input_indices = [0, T-1]
        anchor_indices = [] # No trajectory prior
        
        image_cond = {
            "img": [gt_frames_np[idx] if idx in input_indices else None for idx in range(T)],
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        
        camera_cond = {
            "c2w": c2ws,
            "K": Ks,
            "input_indices": list(range(T)),
        }
        
        version_dict = {
            "H": H,
            "W": W,
            "T": resolve_seva_frame_window(T, args.frame_window),
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": "nearest-gt",
                "video_save_fps": 10.0,
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": 1,
                "cfg": 2.0,
                "camera_scale": args.camera_scale,
                "num_steps": 50,
                "cfg_min": 1.2,
                "encoding_t": 1,
                "decoding_t": 1,
            },
        }
        
        video_path_generator = run_one_scene(
            task="img2trajvid",
            version_dict=version_dict,
            model=model,
            ae=ae,
            conditioner=conditioner,
            denoiser=denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=out_sample_dir,
            use_traj_prior=False,
            traj_prior_Ks=None,
            traj_prior_c2ws=None,
            seed=args.seed,
        )
        
        for _ in video_path_generator:
            pass
            
        # Load generated frames
        pred_frames = []
        pred_paths = sorted(list(Path(out_sample_dir).glob("samples-rgb/*.png")))
        
        if len(pred_paths) != T - len(input_indices):
            print(f"Warning: Expected {T - len(input_indices)} generated frames, got {len(pred_paths)}")
            continue
            
        pred_idx = 0
        for t in range(T):
            if t in input_indices:
                pred_frames.append(videos[0, t])
            else:
                img = Image.open(pred_paths[pred_idx]).convert("RGB")
                img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
                pred_frames.append(img_t.to(args.device))
                pred_idx += 1
                
        pred_video = torch.stack(pred_frames) # [T, 3, H, W]
        
        # Compute metrics on generated frames only
        gen_indices = [t for t in range(T) if t not in input_indices]
        psnr, lpips, ssim = compute_metrics(pred_video[gen_indices], videos[0, gen_indices])
        
        all_psnr.extend(psnr)
        all_lpips.extend(lpips)
        all_ssim.extend(ssim)
        
        # Save video
        out_video = torch.cat([videos[0], pred_video], dim=-1) # Side by side
        out_video_np = unnormalize_video(out_video).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        iio.imwrite(os.path.join(out_sample_dir, "comparison.mp4"), out_video_np, fps=10)
        
    print(f"Final Results over {len(all_psnr)} frames:")
    print(f"PSNR:  {np.mean(all_psnr):.2f}")
    print(f"LPIPS: {np.mean(all_lpips):.4f}")
    print(f"SSIM:  {np.mean(all_ssim):.4f}")
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "psnr": float(np.mean(all_psnr)),
            "lpips": float(np.mean(all_lpips)),
            "ssim": float(np.mean(all_ssim)),
        }, f, indent=4)

if __name__ == "__main__":
    main()

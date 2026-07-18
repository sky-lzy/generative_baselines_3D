"""Run a BASELINE (Geo4D or DA3) over the EXACT π³ eval frames OUR models saw, and
save per-sequence predictions for scoring by the official π³ metric code.

Per sequence (output -> eval_pi3/preds_baselines/<baseline>/<dataset>/<seq>/):
    pred_c2w.npy          (T, 4, 4)  predicted camera-to-world  (pose datasets)
    pred_depth_metric.npy (T, Hd, Wd) predicted METRIC depth     (depth datasets)
    frames.json           {seq, frame_indices, n_src, n_used}

FRAME PARITY: we select the SAME frame_indices our models used. For datasets where
our preds already exist (sintel/tum/bonn/kitti) we READ frames.json; for the rest we
regenerate them with datasets.pi3seq._choose_frame_count/_frame_indices, which has been
verified to reproduce the existing frames.json byte-for-byte (deterministic).

INPUT RESOLUTION (per the protocol — give each baseline its paper's native input,
no OOD 320x240 handicap):
  - Geo4D: its native video res 320x512 (load_video_batch resizes the native frames).
  - DA3:   process_res=504 (long side), aspect preserved (its any-view default).

Baseline depth is METRIC (NOT our disparity) -> saved directly; the π³ scorer aligns
a single per-sequence scale (align_with_scale). Pose c2w follows each baseline's own
validated convention (Geo4D save_tum_poses; DA3 w2c->invert->c2w, ref_view "first").
"""
import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] repo root via env

import imageio.v3 as iio


# Deterministic frame selection — copied VERBATIM from datasets/pi3seq.py so this script
# never needs REPO on sys.path (which would shadow Geo4D's own `utils` package). Verified
# to reproduce the existing eval_pi3/preds/ours_best_1p3b/*/frames.json byte-for-byte.
def _choose_frame_count(n_src: int, cap: int) -> int:
    m = min(int(n_src), int(cap))
    while m > 2 and (m % 4) != 2:
        m -= 1
    if m < 2:
        raise ValueError(f"sequence too short ({n_src} frames) to satisfy M%4==2")
    return m


def _frame_indices(n_src: int, m: int) -> np.ndarray:
    if m == n_src:
        return np.arange(n_src, dtype=np.int64)
    idx = np.linspace(0, n_src - 1, m).round().astype(np.int64)
    if len(np.unique(idx)) != m:
        idx = np.unique(idx)
        while len(idx) < m:
            missing = [i for i in range(n_src) if i not in set(idx.tolist())]
            idx = np.sort(np.concatenate([idx, np.array(missing[: m - len(idx)])]))
    assert np.all(np.diff(idx) > 0), "frame indices must be strictly increasing"
    return idx


def _select_indices(n_src: int, m: int, frame_mode: str) -> np.ndarray:
    """uniform = Protocol A (default, UNCHANGED); contig_first = Protocol B (first m frames)."""
    if frame_mode == "contig_first":
        assert m <= n_src
        return np.arange(m, dtype=np.int64)
    return _frame_indices(n_src, m)


# Set in main() from --frame_mode. The OURS preds dir whose frames.json we cross-check
# against also switches with the protocol (preds for A, preds_c50 for B).
FRAME_MODE = "uniform"
OURS_PREDS_LABEL_DIR = "eval_pi3/preds/ours_best_1p3b"

PI3_DATA = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose") + "/data"  # [standardized_eval PATCH]

# Frame source per dataset (mirrors configurations/dataset/pi3seq_<ds>.yaml VERBATIM).
DATASET_FRAMES = {
    "sintel":    {"tmpl": f"{PI3_DATA}/sintel/training/final/{{seq}}", "ext": "png"},
    "tum":       {"tmpl": f"{PI3_DATA}/tum/{{seq}}/rgb_90", "ext": "png"},
    "scannetv2": {"tmpl": f"{PI3_DATA}/scannetv2/{{seq}}/color_90", "ext": "jpg"},
    "bonn":      {"tmpl": f"{PI3_DATA}/bonn/rgbd_bonn_dataset/{{seq}}/rgb_110", "ext": "png"},
    "kitti":     {"tmpl": f"{PI3_DATA}/kitti/depth_selection/val_selection_cropped/"
                          f"image_gathered/{{seq}}", "ext": "png"},
    # RE10K angular relpose: the 10 seq-id-map PNGs per seq ({idx:04d}.png -> sorted
    # glob is ascending line-idx == temporal order). Pose-only (angular metric).
    "re10k":     {"tmpl": f"{PI3_DATA}/re10k/{{seq}}/images", "ext": "png"},
    "re10k_hf50": {"tmpl": "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50/{seq}/images", "ext": "png"},
}
FRAME_CAP = 50
POSE_DATASETS = {"sintel", "tum", "scannetv2", "re10k", "re10k_hf50"}
DEPTH_DATASETS = {"sintel", "bonn", "kitti"}


def get_seqs(dataset):
    from omegaconf import OmegaConf
    base = OmegaConf.load(REPO / "configurations/dataset/pi3seq.yaml")
    cfg_file = (REPO / "configurations/dataset/pi3seq.yaml" if dataset == "sintel"
                else REPO / f"configurations/dataset/pi3seq_{dataset}.yaml")
    merged = OmegaConf.merge(base, OmegaConf.load(cfg_file))
    return list(merged.seqs)


def build_record(dataset, seq):
    """Return (selected_files, frame_indices, n_src) matching our model's frames exactly."""
    spec = DATASET_FRAMES[dataset]
    img_dir = spec["tmpl"].format(seq=seq)
    files = sorted(glob.glob(os.path.join(img_dir, f"*.{spec['ext']}")))
    if len(files) < 2:
        raise FileNotFoundError(f"No frames for {dataset}/{seq} under {img_dir}")
    n_src = len(files)
    m = _choose_frame_count(n_src, FRAME_CAP)
    idx = _select_indices(n_src, m, FRAME_MODE)
    # Prefer the EXACT frames.json our model already wrote, if present (sanity-consistency).
    ours_fj = REPO / OURS_PREDS_LABEL_DIR / dataset / seq / "frames.json"
    if ours_fj.exists():
        saved = json.load(open(ours_fj))["frame_indices"]
        assert saved == idx.tolist(), f"{dataset}/{seq}: frames.json {saved[:6]} != det {idx[:6].tolist()}"
        idx = np.asarray(saved, dtype=np.int64)
    return [files[i] for i in idx], idx.tolist(), n_src


def load_frames_uint8(files):
    return [np.asarray(iio.imread(str(p)))[..., :3] for p in files]  # list of (H,W,3) uint8


# ───────────────────────────── DA3 ─────────────────────────────
DA3_CKPT = ("/n/netscratch/ydu_lab/Lab/akiruga/hub/"
            "models--depth-anything--DA3-GIANT-1.1/snapshots/"
            "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19")


def load_da3():
    import torch
    from depth_anything_3.api import DepthAnything3
    dev = torch.device("cuda")
    model = DepthAnything3.from_pretrained(DA3_CKPT).to(device=dev)
    model.device = dev
    model.eval()
    return model


def infer_da3(model, frames_uint8, need_pose, need_depth, process_res=504):
    pred = model.inference(
        frames_uint8, use_ray_pose=True, ref_view_strategy="first",
        process_res=process_res, export_dir=None,
    )
    c2w = depth = None
    if need_pose:
        ext = np.asarray(pred.extrinsics, dtype=np.float64)  # (N,3,4) w2c
        N = ext.shape[0]
        w2c = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        w2c[:, :3, :] = ext
        c2w = np.linalg.inv(w2c)
    if need_depth:
        depth = np.asarray(pred.depth, dtype=np.float32)  # (N, Hd, Wd) metric
    return c2w, depth


# ───────────────────────────── Geo4D ─────────────────────────────
GEO4D_DIR = os.environ.get("GEO4D_DIR", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D")  # [standardized_eval PATCH]


def load_geo4d(gpu_no=0):
    sys.path.insert(0, GEO4D_DIR)
    from collections import OrderedDict
    import torch
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config
    from scripts.evaluation.test_geo4d import load_model_checkpoint

    def _resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else Path(GEO4D_DIR) / p)

    ckpt = os.path.join(GEO4D_DIR, "checkpoints/geo4d/model.ckpt")
    config_path = os.path.join(GEO4D_DIR, "configs/inference_geo4d.yaml")
    config = OmegaConf.load(config_path)
    mc = config.pop("model", OmegaConf.create())
    mc["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(mc).cuda(gpu_no)
    model.perframe_ae = True
    model = load_model_checkpoint(model, _resolve(ckpt))
    model.eval()

    pointmap_vae = None
    if "vae_path" in config:
        config["vae_path"] = _resolve(config["vae_path"])
        pvc = config.pop("pointmap_vae_config", OmegaConf.create())
        pointmap_vae = instantiate_from_config(pvc).eval().cuda(gpu_no)
        from lvdm.basics import disabled_train
        pointmap_vae.train = disabled_train
        for p in pointmap_vae.parameters():
            p.requires_grad = False
        vw = torch.load(config["vae_path"])["state_dict"]
        new_sd = OrderedDict((k[6:], v) for k, v in vw.items() if k.startswith("model."))
        pointmap_vae.load_state_dict(new_sd, strict=True)
    return model, config, pointmap_vae


def infer_geo4d(model, config, pointmap_vae, frames_uint8, need_pose, need_depth,
                ddim_steps=5, height=320, width=512, gpu_no=0):
    """Run the Geo4D window loop + post_optimization ONCE; extract poses and/or depth.
    Window/optimization logic is copied VERBATIM from geo4d/eval_geo4d_{pose,depth}_v3.py."""
    import torch
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from utils.funcs import load_video_batch
    from scripts.evaluation.test_geo4d import (
        image_guided_synthesis, post_optimization, get_sky_mask, get_far_away_mask,
        denormalize_pc_bbox2, raymap_to_camera_matrix,
    )
    seed_everything(123)

    # Write the selected native frames to a temp mp4, then load via Geo4D's own loader
    # (resizes to height×width) — identical input path to the validated v3 scripts.
    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "seq.mp4")
        iio.imwrite(vp, np.stack(frames_uint8), fps=10, codec="libx264",
                    output_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        video_frames, fps_list = load_video_batch([vp], frame_stride=1,
                                                  video_size=(height, width), video_frames=-1)
    B, C, T, H, W = video_frames.shape
    n_real = T
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
    slice_list = [slice(s, s + 16, 1) for s in range(0, T - 16 + 1, stride)]
    if not slice_list or slice_list[-1] != slice(T - 16, T, 1):
        slice_list.append(slice(T - 16, T, 1))

    pred_list, view_list = [], []
    pnt_valid_mask = torch.ones((T, H, W, 1), device=f"cuda:{gpu_no}") > 0
    for sl in slice_list:
        videos = videos_all[:, :, sl, :, :].clone()
        view_list.append(views[sl])
        batch_samples = image_guided_synthesis(
            model, prompts, videos, noise_shape, n_samples=1, ddim_steps=ddim_steps,
            ddim_eta=0.0, unconditional_guidance_scale=1.0, cfg_img=None, fs=fps_list[0],
            text_input=True, multiple_cond_cfg=False, loop=False, interp=False,
            timestep_spacing="uniform_trailing", guidance_rescale=0.7, pointmap_vae=pointmap_vae,
        )
        batch_samples = batch_samples[:, 0]
        raymap = crossmap = inverse_depthmap = traj = None
        if model.modality == "pc_ray_cross_depth":
            raymap = batch_samples[:, 4:7]; crossmap = batch_samples[:, 7:10]
            traj = raymap_to_camera_matrix(raymap, crossmap)
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
        invalid = get_sky_mask(x_recon_reshape, sky_value=1.05, eps=0.35) | \
                  get_far_away_mask(x_recon_reshape, far_away_value=1.99)
        confidence[invalid] = 999.0
        pnt_valid_mask[sl] = pnt_valid_mask[sl] * (~invalid)
        inv_conf = 1 / confidence
        inv_conf[invalid] = 0.0
        x_recon = rearrange(x_recon, "t c h w -> t h w c")
        x_recon = denormalize_pc_bbox2(x_recon, alpha=2.0, beta=2.0)
        pp = {"pts3d": x_recon, "conf": inv_conf}
        if inverse_depthmap is not None:
            pp["inverse_depthmap"] = inverse_depthmap
        if traj is not None:
            pp["traj"] = traj
        pred_list.append(pp)

    scene = post_optimization(view_list, pred_list, config.postprocess,
                              conf_optimize=True, init_method="group", lr=0.03, opt_raydir=False)

    c2w = depth = None
    if need_pose:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            traj_path = tf.name
        scene.save_tum_poses(traj_path)
        from scipy.spatial.transform import Rotation
        poses = []
        for line in open(traj_path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            _, tx, ty, tz, qx, qy, qz, qw = [float(v) for v in line.split()]
            P = np.eye(4); P[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            P[:3, 3] = [tx, ty, tz]; poses.append(P)
        os.unlink(traj_path)
        c2w = np.stack(poses, 0)[:n_real]  # drop padded frames
    if need_depth:
        dms = scene.get_depthmaps()
        depth = torch.stack(dms, 0).detach().cpu().float().numpy()[:n_real]  # (T,Hd,Wd) metric
    return c2w, depth


def run_dataset(dataset, baseline, model, geo_cfg, vae, shard, nshards, out_dir, rerun):
    import time
    need_pose = dataset in POSE_DATASETS
    need_depth = dataset in DEPTH_DATASETS
    assert need_pose or need_depth
    seqs = get_seqs(dataset)
    seqs = [s for i, s in enumerate(seqs) if i % nshards == shard]
    out_root = Path(out_dir) / baseline / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[{baseline}/{dataset}] shard {shard}/{nshards}: {len(seqs)} seqs "
          f"(pose={need_pose} depth={need_depth})", flush=True)
    for seq in seqs:
        sd = out_root / seq
        done = (sd / "frames.json").exists() and \
               ((not need_pose) or (sd / "pred_c2w.npy").exists()) and \
               ((not need_depth) or (sd / "pred_depth_metric.npy").exists())
        if done and not rerun:
            print(f"  skip (done) {seq}", flush=True); continue
        t0 = time.time()
        files, fidx, n_src = build_record(dataset, seq)
        frames = load_frames_uint8(files)
        try:
            if baseline == "da3":
                c2w, depth = infer_da3(model, frames, need_pose, need_depth)
            else:
                c2w, depth = infer_geo4d(model, geo_cfg, vae, frames, need_pose, need_depth)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR {seq}: {e}", flush=True); continue
        sd.mkdir(parents=True, exist_ok=True)
        if need_pose:
            assert c2w is not None and len(c2w) == len(fidx), \
                f"{seq}: pose {None if c2w is None else len(c2w)} vs {len(fidx)} frames"
            np.save(sd / "pred_c2w.npy", c2w.astype(np.float64))
        if need_depth:
            assert depth is not None and len(depth) == len(fidx), \
                f"{seq}: depth {None if depth is None else len(depth)} vs {len(fidx)} frames"
            np.save(sd / "pred_depth_metric.npy", depth.astype(np.float32))
        json.dump({"seq": seq, "frame_indices": fidx, "n_src": n_src, "n_used": len(fidx)},
                  open(sd / "frames.json", "w"), indent=2)
        msg = f"  {seq}: "
        if need_pose: msg += f"c2w {c2w.shape} "
        if need_depth: msg += f"depth {depth.shape} "
        print(msg + f"({time.time()-t0:.1f}s)", flush=True)
    print(f"DONE [{baseline}/{dataset}] shard {shard}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["geo4d", "da3"])
    ap.add_argument("--dataset", help="single dataset")
    ap.add_argument("--datasets", nargs="+", help="multiple datasets (one model load, processed in order)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out_dir", default=str(REPO / "eval_pi3" / "preds_baselines"))
    ap.add_argument("--frame_mode", default="uniform", choices=["uniform", "contig_first"],
                    help="uniform=Protocol A (default, UNCHANGED); contig_first=Protocol B")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    global FRAME_MODE, OURS_PREDS_LABEL_DIR
    FRAME_MODE = args.frame_mode
    if FRAME_MODE == "contig_first":
        OURS_PREDS_LABEL_DIR = "eval_pi3/preds_c50/ours_best_1p3b"

    datasets = args.datasets if args.datasets else [args.dataset]
    assert datasets and all(d in DATASET_FRAMES for d in datasets), f"bad datasets {datasets}"

    if args.baseline == "da3":
        model = load_da3(); geo_cfg = vae = None
    else:
        model, geo_cfg, vae = load_geo4d(0)

    for ds in datasets:
        run_dataset(ds, args.baseline, model, geo_cfg, vae,
                    args.shard, args.nshards, args.out_dir, args.rerun)


if __name__ == "__main__":
    main()

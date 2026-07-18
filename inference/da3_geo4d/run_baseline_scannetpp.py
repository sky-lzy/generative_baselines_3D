"""ScanNet++ video-depth eval for the BASELINES (Geo4D / DA3), scored with the π³
depth metric (utils.depth.depth_evaluation, align_with_scale=True) VERBATIM —
the SAME path our-model bridge (run_eval_scannetpp.py) uses, so it is directly
comparable to our models.

ScanNet++ is NOT a π³ dataset: it uses OUR ScanNetppDataset, which yields per scene
  videos      (T,3,H,W) RGB in [-1,1]  at model res 240x320  -> fed to the baseline
  depths_raw  (T,1,H,W) GT depth in METERS (ViPE/EXR) at 240x320
  depths_valid(T,1,H,W) bool
Because GT exists only at the dataset's 240x320 (already 4:3) crop, EVERY method
(ours + baselines) is fed the same 240x320 frames here — a fair, identical input.
We denormalize videos -> uint8 frames, run the baseline (metric depth), resize to
240x320, and score against depths_raw with depth_evaluation (no crop needed; pred
and GT are already the same 4:3 frames). Baseline depth is metric -> no disparity
decode. Writes eval_pi3/results_baselines/<baseline>_scannetpp.csv.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize
from tqdm import tqdm

REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] repo root via env
sys.path.insert(0, str(REPO))
from datasets.scannetpp import ScanNetppDataset
from eval_pi3.run_baseline_pi3 import load_da3, load_geo4d, infer_da3, infer_geo4d

# π³ depth_evaluation via explicit file load (avoid utils-package name collision; same
# trick as run_eval_scannetpp.py).
import importlib.util as _ilu
_PI3_DEPTH = os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parents[2] / "evaluation" / "pi3_metrics")) + "/utils/depth.py"  # [standardized_eval PATCH] verbatim local metric copy
_spec = _ilu.spec_from_file_location("pi3_utils_depth", _PI3_DEPTH)
_pi3_depth = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pi3_depth)
depth_evaluation = _pi3_depth.depth_evaluation

import cv2


def build_cfg(dataset_cfg, height, width, n_frames):
    overrides = [
        "experiment=exp_video", "algorithm=wan_t2v_ray_depth_mot_concat",
        f"dataset={dataset_cfg}", "experiment.tasks=[test]",
        f"dataset.n_frames={n_frames}", f"dataset.height={height}", f"dataset.width={width}",
        f"dataset.raymap_height={height // 8}", f"dataset.raymap_width={width // 8}",
        "++dataset.use_sky_mask=false", "++dataset.no_augmentations=true",
    ]
    with initialize(version_base=None, config_path="../configurations"):
        cfg = compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["geo4d", "da3", "pi3"])
    ap.add_argument("--pose_only", action="store_true",
                    help="skip depth scoring (existing depth CSVs stay authoritative); "
                         "still saves pred/gt c2w for eval_scannetpp_pose.py")
    ap.add_argument("--pose_dir", default=str(REPO / "eval_pi3" / "preds_spp_pose"))
    ap.add_argument("--save_depth", action="store_true",
                    help="save per-scene metric depth npys (for the report depth-video carousel)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N scenes (0 = all); carousel needs 8")
    ap.add_argument("--dataset_cfg", default="scannetpp_full")
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out_dir", default=str(REPO / "eval_pi3" / "results_baselines"))
    args = ap.parse_args()

    cfg = build_cfg(args.dataset_cfg, args.height, args.width, args.n_frames)
    dataset = ScanNetppDataset(cfg.dataset, split="test")
    print(f"INFO: {len(dataset)} scannetpp scenes; baseline={args.baseline} "
          f"shard {args.shard}/{args.nshards}", flush=True)

    # PRE-EXTRACT all (sharded) scenes WHILE REPO is still importable. Geo4D's `utils`
    # package is a namespace pkg that REPO/utils (a regular pkg) shadows regardless of
    # sys.path order, so we must purge REPO from sys.path BEFORE load_geo4d. We therefore
    # pull every scene's RGB frames + GT here, drop the dataset, purge REPO, then load Geo4D.
    extracted = []  # (scene, frames_list, gt_THW, valid_THW, gt_c2w)
    for i in tqdm(range(len(dataset)), desc="extract scannetpp"):
        if i % args.nshards != args.shard:
            continue
        item = dataset[i]
        scene = item.get("scene_id", f"scene_{i}")
        scene = scene if isinstance(scene, str) else str(scene)
        videos = item["videos"]  # (T,3,H,W) in [-1,1]
        frames = (((videos.permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0) * 255.0)
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        frames_list = [frames[t] for t in range(frames.shape[0])]
        depths_raw = item["depths_raw"].squeeze(1).cpu().numpy()
        valid = item["depths_valid"].squeeze(1).cpu().numpy().astype(bool)
        gt_c2w = item["extrinsics_gt"].cpu().numpy().astype(np.float64)   # (T,4,4) ViPE, 1st-frame-rel
        extracted.append((scene, frames_list, depths_raw, valid, gt_c2w))
        if args.limit and len(extracted) >= args.limit:
            break
    del dataset
    print(f"INFO: extracted {len(extracted)} scenes; purging REPO from sys.path for Geo4D", flush=True)
    for p in [str(REPO), ".", ""]:
        while p in sys.path:
            sys.path.remove(p)

    if args.baseline == "da3":
        model = load_da3(); geo_cfg = vae = None
    elif args.baseline == "geo4d":
        model, geo_cfg, vae = load_geo4d(0)
    else:  # pi3 — official π³. REPO's `utils` package is cached in sys.modules and would
        # shadow π³'s utils.interfaces; purge it (REPO paths were already removed above).
        for k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]
        PI3_ROOT = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")  # [standardized_eval PATCH]
        sys.path.insert(0, PI3_ROOT)
        import rootutils
        rootutils.setup_root(PI3_ROOT, indicator=".project-root", pythonpath=True)
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
        geo_cfg = vae = None

    def _infer_pi3(frames_list):
        """One forward -> (c2w (T,4,4) float64, depth (T,h14,w14) float32). Official π³
        preprocessing (load_and_resize14 @512), same as run_pi3_re10k50/run_pi3_c50 pose."""
        import tempfile, shutil
        from utils.interfaces import load_and_resize14   # π³'s utils (REPO purged above)
        td = tempfile.mkdtemp(prefix="pi3spp_")
        try:
            files = []
            for t, fr in enumerate(frames_list):
                p = os.path.join(td, f"{t:04d}.png")
                cv2.imwrite(p, fr[..., ::-1]); files.append(p)
            imgs = load_and_resize14(files, new_width=512, device="cuda", verbose=False)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(imgs)
            c2w = np.asarray(pred["camera_poses"][0].float().cpu(), np.float64)
            depth = np.asarray(pred["local_points"][0, ..., -1].float().cpu(), np.float32)
            return c2w, depth
        finally:
            shutil.rmtree(td, ignore_errors=True)

    pose_root = Path(args.pose_dir) / args.baseline / "scannetpp"
    pose_root.mkdir(parents=True, exist_ok=True)
    need_depth = (not args.pose_only) or args.save_depth

    rows = []  # (scene, abs_rel, delta1, valid_pixels)
    for scene, frames_list, depths_raw, valid, gt_c2w in tqdm(extracted, desc=f"{args.baseline}/scannetpp"):
        try:
            if args.baseline == "da3":
                c2w, depth = infer_da3(model, frames_list, need_pose=True, need_depth=need_depth)
            elif args.baseline == "geo4d":
                c2w, depth = infer_geo4d(model, geo_cfg, vae, frames_list,
                                         need_pose=True, need_depth=need_depth)
            else:
                c2w, depth = _infer_pi3(frames_list)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR {scene}: {e}", flush=True); continue

        if c2w is not None:
            c2w = c2w.cpu().numpy() if torch.is_tensor(c2w) else np.asarray(c2w)
            np.save(pose_root / f"{scene}_pred_c2w.npy", c2w.astype(np.float64))
            np.save(pose_root / f"{scene}_gt_c2w.npy", gt_c2w)

        if depth is not None and args.save_depth:
            dep = depth.cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
            np.save(pose_root / f"{scene}_pred_depth_metric.npy", dep.astype(np.float16))

        if depth is None or args.pose_only:
            continue
        T, Hg, Wg = depths_raw.shape
        n = min(depth.shape[0], T)
        pred = np.stack([cv2.resize(depth[t].astype(np.float32), (Wg, Hg),
                                    interpolation=cv2.INTER_LINEAR) for t in range(n)], 0).astype(np.float64)
        gt = np.where(valid[:n] & (depths_raw[:n] > 0), depths_raw[:n], 0.0).astype(np.float64)
        results, *_ = depth_evaluation(pred, gt, align_with_scale=True, use_gpu=False, max_depth=None)
        ar = results["Abs Rel"]; d1 = results["δ < 1.25"]; vp = results["valid_pixels"]
        rows.append((scene, ar, d1, vp))
        print(f"{scene:24s}  AbsRel {ar:7.4f}  d1 {d1:7.4f}  vpix {vp}", flush=True)

    if not rows:
        if args.pose_only:
            print(f"pose_only: saved c2w for {len(extracted)} scenes -> {pose_root}"); return
        print("no scenes processed"); return
    abs_arr = np.array([r[1] for r in rows], float)
    d1_arr = np.array([r[2] for r in rows], float)
    w = np.array([r[3] for r in rows], float)
    abs_rel_mean = float(abs_arr.mean()); d1_mean = float(d1_arr.mean())
    abs_rel_w = float(np.average(abs_arr, weights=w)); d1_w = float(np.average(d1_arr, weights=w))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.nshards == 1 else f"_shard{args.shard}of{args.nshards}"
    csv_path = os.path.join(args.out_dir, f"{args.baseline}_scannetpp{suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for scene, ar, d1, vp in rows:
            f.write(f"{scene},,,,{ar:.6f},{d1:.6f},{vp}\n")
        f.write(f"AVERAGE(meanseq),,,,{abs_rel_mean:.6f},{d1_mean:.6f},\n")
        f.write(f"AVERAGE(vpixwt_depth),,,,{abs_rel_w:.6f},{d1_w:.6f},\n")
    print(f"\nbaseline={args.baseline} scannetpp ({len(rows)} scenes)")
    print(f"DEPTH (vpix-weighted): AbsRel {abs_rel_w:.4f}  delta<1.25 {d1_w:.4f}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()

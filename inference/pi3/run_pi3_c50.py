"""PROTOCOL B — PI3-on-50: run PI3's OWN model on the FIRST min(50, len) CONTIGUOUS
frames of each sequence (native stride) and score it with the OFFICIAL π³ metric code
VERBATIM, full-frame native res (load_img_size=512, align=scale) — i.e. EXACTLY the π³
anchor pipeline, the ONLY change being frame selection (first-50 instead of all frames).

This is the Protocol-B PI3 comparison column. It does NOT touch the full-frame ANCHOR
outputs (those stay as the paper-reproduction reference). Writes:
    eval_pi3/results_baselines_c50/pi3_<dataset>.csv
    eval_pi3/preds_c50/pi3/<dataset>/<seq>/{pred_c2w.npy|pred_depth_metric.npy,frames.json}

Pose datasets: tum, scannetv2  (sintel reuses Protocol A — all seqs <=50 => first-50 == all).
Depth datasets: bonn, kitti    (sintel reuses Protocol A likewise).

Run (single dataset, one GPU):
  CUDA_VISIBLE_DEVICES=<g> HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga \
    mamba run -n test2 python3 eval_pi3/run_pi3_c50.py --dataset tum
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] repo root via env
PI3_ROOT = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")  # [standardized_eval PATCH]
sys.path.insert(0, PI3_ROOT)

import rootutils
rootutils.setup_root(PI3_ROOT, indicator=".project-root", pythonpath=True)

# OFFICIAL π³ code — imported and used VERBATIM (same as the anchor scripts).
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_cameras_c2w, infer_videodepth
from relpose.evo_utils import load_traj, get_tum_poses, eval_metrics, calculate_averages
from utils.depth import depth_evaluation, depth_read_sintel, depth_read_bonn, depth_read_kitti

PI3_DATA = f"{PI3_ROOT}/data"
FRAME_CAP = 50

# Frame source per dataset (mirrors run_baseline_pi3.DATASET_FRAMES VERBATIM).
DATASET_FRAMES = {
    "sintel":    {"tmpl": f"{PI3_DATA}/sintel/training/final/{{seq}}", "ext": "png"},
    "tum":       {"tmpl": f"{PI3_DATA}/tum/{{seq}}/rgb_90", "ext": "png"},
    "scannetv2": {"tmpl": f"{PI3_DATA}/scannetv2/{{seq}}/color_90", "ext": "jpg"},
    "bonn":      {"tmpl": f"{PI3_DATA}/bonn/rgbd_bonn_dataset/{{seq}}/rgb_110", "ext": "png"},
    "kitti":     {"tmpl": f"{PI3_DATA}/kitti/depth_selection/val_selection_cropped/"
                          f"image_gathered/{{seq}}", "ext": "png"},
}

# Per-dataset GT layout + π³ depth-eval kwargs (== EVAL_DEPTH_METADATA + relpose configs).
DATASET_SPEC = {
    "sintel": {"pose": {"anno": f"{PI3_DATA}/sintel/training/camdata_left/{{seq}}", "format": "sintel"},
               # depth GT == score_depth_hf.DEPTH["sintel"] VERBATIM (training/depth .dpt, max_depth=70)
               "depth": {"dir": f"{PI3_DATA}/sintel/training/depth/{{seq}}",
                         "ext": "dpt", "read": depth_read_sintel,
                         "kwargs": {"max_depth": 70}}},
    "tum": {"pose": {"anno": f"{PI3_DATA}/tum/{{seq}}/groundtruth_90.txt", "format": "tum"}},
    "scannetv2": {"pose": {"anno": f"{PI3_DATA}/scannetv2/{{seq}}/pose_90.txt", "format": "replica"}},
    # kwargs == EVAL_DEPTH_METADATA[ds]["depth_evaluation_kwargs"] VERBATIM (anchor-faithful):
    #   bonn: max_depth=70 (NO post_clip);  kitti: max_depth=None.
    "bonn": {"depth": {"dir": f"{PI3_DATA}/bonn/rgbd_bonn_dataset/{{seq}}/depth_110",
                       "ext": "png", "read": depth_read_bonn,
                       "kwargs": {"max_depth": 70}}},
    "kitti": {"depth": {"dir": f"{PI3_DATA}/kitti/depth_selection/val_selection_cropped/"
                               f"groundtruth_depth_gathered/{{seq}}", "ext": "png",
                        "read": depth_read_kitti, "kwargs": {"max_depth": None}}},
}


def get_seqs(dataset):
    from omegaconf import OmegaConf
    base = OmegaConf.load(REPO / "configurations/dataset/pi3seq.yaml")
    cfg_file = (REPO / "configurations/dataset/pi3seq.yaml" if dataset == "sintel"
                else REPO / f"configurations/dataset/pi3seq_{dataset}.yaml")
    merged = OmegaConf.merge(base, OmegaConf.load(cfg_file))
    return list(merged.seqs)


def list_first_n(dataset, seq):
    spec = DATASET_FRAMES[dataset]
    files = sorted(glob.glob(os.path.join(spec["tmpl"].format(seq=seq), f"*.{spec['ext']}")))
    if len(files) < 2:
        raise FileNotFoundError(f"No frames for {dataset}/{seq}")
    n_src = len(files)
    n = min(FRAME_CAP, n_src)                      # first-N contiguous, native stride
    idx = list(range(n))
    return [files[i] for i in idx], idx, n_src


def make_cfg():
    return SimpleNamespace(load_img_size=512, device="cuda", verbose=False)


def eval_pose(pspec, seq, fidx, pred_c2w):
    try:
        gt_tum, gt_tt = load_traj(gt_traj_file=pspec["anno"].format(seq=seq),
                                  traj_format=pspec["format"], stride=1)
    except np.linalg.LinAlgError:
        print(f"  [skip] {seq}: GT LinAlgError (matches PI3 harness skip)", flush=True)
        return None
    fi = np.asarray(fidx, dtype=np.int64)
    assert len(gt_tum) >= fi.max() + 1, f"{seq}: GT {len(gt_tum)} need {fi.max()}"
    gt_traj = [gt_tum[fi], gt_tt[fi]]
    pred_traj = get_tum_poses([pred_c2w[i] for i in range(len(pred_c2w))])
    assert len(pred_traj[0]) == len(gt_traj[0])
    fname = str(REPO / "eval_pi3/results_baselines_c50/_tmp_pi3_metric.txt")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ate, rpe_t, rpe_r = eval_metrics(pred_traj, gt_traj, seq=seq, filename=fname)
    return float(ate), float(rpe_t), float(rpe_r)


def eval_depth(dspec, seq, fidx, pred_depth):
    """pred_depth: (N, h14, w14) from PI3 local_points z. Anchor protocol: resize CUBIC to
    full GT res, depth_evaluation align_with_scale (FULL frame — no crop, like the anchor)."""
    gt_files = sorted(glob.glob(os.path.join(dspec["dir"].format(seq=seq), f"*.{dspec['ext']}")))
    assert len(gt_files) >= max(fidx) + 1, f"{seq}: {len(gt_files)} GT need {max(fidx)}"
    gt = np.stack([dspec["read"](gt_files[i]) for i in fidx], axis=0)            # (N,Hg,Wg)
    pr = np.stack([cv2.resize(pred_depth[i], (gt.shape[2], gt.shape[1]),
                              interpolation=cv2.INTER_CUBIC)
                   for i in range(pred_depth.shape[0])], axis=0)
    results, *_ = depth_evaluation(pr, gt, align_with_scale=True, use_gpu=False, **dspec["kwargs"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPEC))
    ap.add_argument("--out_dir", default=str(REPO / "eval_pi3" / "results_baselines_c50"))
    ap.add_argument("--preds_dir", default=str(REPO / "eval_pi3" / "preds_c50"))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    spec = DATASET_SPEC[ds]
    do_pose, do_depth = "pose" in spec, "depth" in spec
    cfg = make_cfg()

    model = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
    print(f"Loaded Pi3 for PI3-on-50 [{ds}] pose={do_pose} depth={do_depth}", flush=True)

    seqs = get_seqs(ds)
    pred_root = Path(args.preds_dir) / "pi3" / ds
    pred_root.mkdir(parents=True, exist_ok=True)

    pose_results, depth_rows = [], []
    for seq in seqs:
        files, fidx, n_src = list_first_n(ds, seq)
        sd = pred_root / seq
        sd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with torch.no_grad():
            if do_pose:
                c2w, _ = infer_cameras_c2w(files, model, cfg)      # (N,3,4) torch
                c2w = np.asarray(c2w.cpu(), dtype=np.float64)
                np.save(sd / "pred_c2w.npy", c2w)
                res = eval_pose(spec["pose"], seq, fidx, c2w)
                if res is not None:
                    pose_results.append((seq, *res))
            if do_depth:
                _, depth_map, _ = infer_videodepth(files, model, cfg)  # (N,h14,w14) torch
                dep = depth_map.float().cpu().numpy()
                np.save(sd / "pred_depth_metric.npy", dep.astype(np.float32))
                dres = eval_depth(spec["depth"], seq, fidx, dep.astype(np.float64))
                depth_rows.append((seq, dres["Abs Rel"], dres["δ < 1.25"], dres["valid_pixels"]))
        json.dump({"seq": seq, "frame_indices": fidx, "n_src": n_src, "n_used": len(fidx)},
                  open(sd / "frames.json", "w"), indent=2)
        msg = f"{seq:34s} ({time.time()-t0:.1f}s)"
        if pose_results and pose_results[-1][0] == seq:
            _, a, rt, rr = pose_results[-1]; msg += f"  ATE {a:.4f} RPE-t {rt:.4f} RPE-r {rr:.4f}"
        if depth_rows and depth_rows[-1][0] == seq:
            _, ar, d1, vp = depth_rows[-1]; msg += f"  AbsRel {ar:.4f} d1 {d1:.4f} vpix {vp}"
        print(msg, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"pi3_{ds}.csv")
    avg_ate = avg_rt = avg_rr = float("nan")
    ar_w = d1_w = ar_m = d1_m = float("nan")
    if do_pose and pose_results:
        avg_ate, avg_rt, avg_rr = calculate_averages(pose_results)
    if do_depth and depth_rows:
        a = np.array([r[1] for r in depth_rows], float); d = np.array([r[2] for r in depth_rows], float)
        w = np.array([r[3] for r in depth_rows], float)
        ar_m, d1_m = float(a.mean()), float(d.mean())
        ar_w, d1_w = float(np.average(a, weights=w)), float(np.average(d, weights=w))
    pr_by = {r[0]: r for r in pose_results}; dr_by = {r[0]: r for r in depth_rows}
    with open(csv_path, "w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for seq in seqs:
            ate, rt, rr = (pr_by[seq][1:] if seq in pr_by else ("", "", ""))
            ar, d1, vp = (dr_by[seq][1:] if seq in dr_by else ("", "", ""))
            def fmt(x): return f"{x:.6f}" if isinstance(x, float) else x
            f.write(f"{seq},{fmt(ate)},{fmt(rt)},{fmt(rr)},{fmt(ar)},{fmt(d1)},{vp}\n")
        if do_pose and pose_results:
            f.write(f"AVERAGE(meanseq),{avg_ate:.6f},{avg_rt:.6f},{avg_rr:.6f},,,\n")
        if do_depth and depth_rows:
            f.write(f"AVERAGE(meanseq),,,,{ar_m:.6f},{d1_m:.6f},\n")
            f.write(f"AVERAGE(vpixwt_depth),,,,{ar_w:.6f},{d1_w:.6f},\n")
    print("\n==================== SUMMARY (PI3-on-50) ====================")
    print(f"dataset={ds} ({len(seqs)} seqs)")
    if do_pose and pose_results:
        print(f"POSE : ATE {avg_ate:.4f}  RPE-t {avg_rt:.4f}  RPE-r {avg_rr:.4f}")
    if do_depth and depth_rows:
        print(f"DEPTH: AbsRel(vpix) {ar_w:.4f}  d1(vpix) {d1_w:.4f}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()

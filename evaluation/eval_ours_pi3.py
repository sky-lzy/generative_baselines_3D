"""Score OUR model's saved predictions with the OFFICIAL π³ metric code VERBATIM.

Per sequence (predictions from run_ours_pi3.py under eval_pi3/preds/<model>/<dataset>/<seq>/):

POSE (Sim3, evo):
  GT  = PI3 `load_traj(camdata_left/{seq}, 'sintel', stride=1)`  (full N-frame c2w traj)
        sliced to the EXACT frame_indices the model saw (frames.json).
  PRED= PI3 `get_tum_poses([c2w_0, c2w_1, ...])` from pred_c2w.npy.
  ATE/RPE-t/RPE-r = PI3 `eval_metrics(pred_traj, gt_traj, ...)`.

DEPTH (scale-ONLY align, the π³ paper video-depth protocol):
  pred [-1,1] modified-disparity -> disp=clip(x*.5+.5,0,1) -> d=1/clip(disp,eps,1-eps)-1
       (proportional to metric depth; the global per-seq scale is solved by align_with_scale),
  then cv2.resize(CUBIC) to GT resolution per frame (exactly as PI3 videodepth/eval.py).
  GT  = PI3 `depth_read_sintel` on the SAME frame_indices' .dpt files.
  metrics = PI3 `depth_evaluation(pred, gt, max_depth=70, post_clip_max=70,
            align_with_scale=True, use_gpu=False)`.

Aggregates per dataset and writes eval_pi3/results_ours/<model>_<dataset>.csv.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PI3_ROOT = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")  # [standardized_eval PATCH] GT-data root
sys.path.insert(0, PI3_ROOT)
sys.path.insert(0, os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parent / "pi3_metrics")))  # [standardized_eval PATCH] metric imports resolve to the local verbatim copy

# OFFICIAL π³ metric functions — imported and used VERBATIM.
from relpose.evo_utils import load_traj, get_tum_poses, eval_metrics, calculate_averages
from utils.depth import (
    depth_evaluation, depth_read_sintel, depth_read_bonn, depth_read_kitti,
)

# AUTHORITATIVE disparity[-1,1] -> metric-depth conversion used by our VALIDATED prior eval.
# The key piece the hand-rolled decode missed is percentile_clip=2.0, which floors disp at
# its 2nd-percentile so the 1/disp inversion doesn't explode on near-zero-disparity (far/sky)
# pixels — without it Sintel depth blows up (AbsRel ~1, d1 ~0.19). scale is absorbed by the
# scale-only alignment downstream, so scale=1.0 is fine.
sys.path.insert(0, os.environ.get("GEN3D_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines_3D"))  # [standardized_eval PATCH]
from eval_common_v3 import disparity_norm_to_metric_depth

# Our model sees an aspect-preserving CENTER-CROP of each frame (pi3seq.py). The GT depth must
# be cropped IDENTICALLY before comparison, else pred (4:3 center crop) is stretched onto the
# full wide GT and the per-pixel depth metrics are meaningless. Same crop util pi3seq.py uses.
sys.path.insert(0, os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] for datasets._crop_utils
from datasets._crop_utils import center_crop_box_for_aspect, apply_spatial_crop

# Per-dataset GT layout + depth-eval kwargs. Each dataset may have a "pose" sub-spec
# (anno path template + π³ traj_format) and/or a "depth" sub-spec (GT dir template, ext,
# π³ depth reader, depth_evaluation kwargs). Presence of a sub-spec => that metric is
# computed for that dataset. All paths/kwargs mirror the π³ configs/data/*.yaml +
# utils/depth.py:EVAL_DEPTH_METADATA VERBATIM.
DATASET_SPEC = {
    "sintel": {
        "pose": {"anno": f"{PI3_ROOT}/data/sintel/training/camdata_left/{{seq}}",
                 "format": "sintel"},
        "depth": {"dir": f"{PI3_ROOT}/data/sintel/training/depth/{{seq}}", "ext": "dpt",
                  "read": depth_read_sintel,
                  "kwargs": {"max_depth": 70, "post_clip_max": 70}},
    },
    "tum": {
        "pose": {"anno": f"{PI3_ROOT}/data/tum/{{seq}}/groundtruth_90.txt",
                 "format": "tum"},
    },
    "scannetv2": {
        "pose": {"anno": f"{PI3_ROOT}/data/scannetv2/{{seq}}/pose_90.txt",
                 "format": "replica"},
    },
    "bonn": {
        "depth": {"dir": f"{PI3_ROOT}/data/bonn/rgbd_bonn_dataset/{{seq}}/depth_110",
                  "ext": "png", "read": depth_read_bonn,
                  "kwargs": {"max_depth": 70}},
    },
    "kitti": {
        "depth": {"dir": f"{PI3_ROOT}/data/kitti/depth_selection/val_selection_cropped/"
                         f"groundtruth_depth_gathered/{{seq}}",
                  "ext": "png", "read": depth_read_kitti,
                  "kwargs": {"max_depth": None}},
    },
}


def decode_pred_depth(pred_norm: np.ndarray) -> np.ndarray:
    """[-1,1] normalized disparity -> metric-proportional depth, VERBATIM via our validated
    conversion (percentile_clip=2.0 bounds the 1/disp inversion on far/sky pixels). The
    per-video percentile is computed across the whole (T,H,W) array, so pass the full seq."""
    return disparity_norm_to_metric_depth(
        depth_norm=pred_norm.astype(np.float32), scale=1.0,
    ).astype(np.float64)


def eval_pose(pspec, seq, frame_indices, pred_c2w):
    gt_tum, gt_tt = load_traj(
        gt_traj_file=pspec["anno"].format(seq=seq),
        traj_format=pspec["format"],
        stride=1,
    )
    fi = np.asarray(frame_indices, dtype=np.int64)
    assert len(gt_tum) >= fi.max() + 1, f"{seq}: GT has {len(gt_tum)} poses, need idx {fi.max()}"
    gt_traj = [gt_tum[fi], gt_tt[fi]]                      # slice GT to the frames the model saw
    pred_traj = get_tum_poses([pred_c2w[i] for i in range(len(pred_c2w))])
    assert len(pred_traj[0]) == len(gt_traj[0]), \
        f"{seq}: pred {len(pred_traj[0])} vs gt {len(gt_traj[0])} poses"
    fname = os.path.join("/tmp", f"benchmark_eval_metric_{os.getpid()}.txt")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ate, rpe_t, rpe_r = eval_metrics(pred_traj, gt_traj, seq=seq, filename=fname)
    return float(ate), float(rpe_t), float(rpe_r)


def eval_depth(dspec, seq, frame_indices, pred_depth_norm):
    gt_files = sorted(glob.glob(os.path.join(dspec["dir"].format(seq=seq), f"*.{dspec['ext']}")))
    fi = list(frame_indices)
    assert len(gt_files) >= max(fi) + 1, \
        f"{seq}: {len(gt_files)} GT depth files, need idx {max(fi)}"
    gt_depth = np.stack([dspec["read"](gt_files[i]) for i in fi], axis=0)        # (M, Hgt, Wgt)
    pred_d = decode_pred_depth(pred_depth_norm)                                  # (M, Hp, Wp)
    Hp, Wp = pred_d.shape[1], pred_d.shape[2]
    # Crop GT to the SAME aspect-preserving center crop the model's input used, then resize
    # pred to that cropped-GT resolution (mirrors PI3 videodepth/eval.py resize-to-GT).
    box = center_crop_box_for_aspect(gt_depth.shape[1], gt_depth.shape[2], Hp, Wp)
    gt_crop = apply_spatial_crop(gt_depth, box)                                  # (M, Hc, Wc)
    Hc, Wc = gt_crop.shape[1], gt_crop.shape[2]
    pred_rs = np.stack([
        cv2.resize(pred_d[i], (Wc, Hc), interpolation=cv2.INTER_CUBIC)
        for i in range(pred_d.shape[0])
    ], axis=0)
    results, *_ = depth_evaluation(
        pred_rs, gt_crop, align_with_scale=True, use_gpu=False, **dspec["kwargs"],
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default=str(Path(__file__).resolve().parent / "preds"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="sintel")
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results_ours"))
    args = ap.parse_args()

    spec = DATASET_SPEC[args.dataset]
    do_pose = "pose" in spec
    do_depth = "depth" in spec
    seq_root = Path(args.preds_dir) / args.model / args.dataset
    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No prediction seqs under {seq_root}")

    pose_results = []      # (seq, ate, rpe_t, rpe_r)
    depth_rows = []        # (seq, abs_rel, delta1, valid_pixels)

    for sd in seq_dirs:
        seq = sd.name
        with open(sd / "frames.json") as f:
            meta = json.load(f)
        frame_indices = meta["frame_indices"]

        ate = rpe_t = rpe_r = float("nan")
        abs_rel = delta1 = float("nan"); vpix = 0
        # Per-scene try/except so one bad scene (e.g. a degenerate GT pose matrix where evo's
        # quaternion_from_matrix eigh fails to converge — a GT-data defect, not a model failure)
        # is SKIPPED rather than crashing the whole eval. Skipped scenes are excluded from the
        # average (same as the π³ harness, which skips scenes it can't score).
        if do_pose:
            try:
                pred_c2w = np.load(sd / "pred_c2w.npy")
                ate, rpe_t, rpe_r = eval_pose(spec["pose"], seq, frame_indices, pred_c2w)
                pose_results.append((seq, ate, rpe_t, rpe_r))
            except Exception as e:
                print(f"{seq:34s}  POSE SKIPPED ({type(e).__name__}: {e})", flush=True)
        if do_depth:
            try:
                pred_depth_norm = np.load(sd / "pred_depth_norm.npy")
                dres = eval_depth(spec["depth"], seq, frame_indices, pred_depth_norm)
                abs_rel = dres["Abs Rel"]; delta1 = dres["δ < 1.25"]; vpix = dres["valid_pixels"]
                depth_rows.append((seq, abs_rel, delta1, vpix))
            except Exception as e:
                print(f"{seq:34s}  DEPTH SKIPPED ({type(e).__name__}: {e})", flush=True)
        print(f"{seq:34s}  ATE {ate:8.4f}  RPE-t {rpe_t:8.4f}  RPE-r {rpe_r:8.4f}  "
              f"| AbsRel {abs_rel:8.4f}  d1 {delta1:8.4f}  vpix {vpix}", flush=True)

    avg_ate = avg_rpe_t = avg_rpe_r = float("nan")
    abs_rel_mean = d1_mean = abs_rel_w = d1_w = float("nan")
    if do_pose:
        avg_ate, avg_rpe_t, avg_rpe_r = calculate_averages(pose_results)
    if do_depth:
        # unweighted mean over seqs AND valid-pixel-weighted (PI3 videodepth/eval.py).
        abs_arr = np.array([r[1] for r in depth_rows], float)
        d1_arr = np.array([r[2] for r in depth_rows], float)
        w = np.array([r[3] for r in depth_rows], float)
        abs_rel_mean = float(abs_arr.mean()); d1_mean = float(d1_arr.mean())
        abs_rel_w = float(np.average(abs_arr, weights=w)); d1_w = float(np.average(d1_arr, weights=w))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.model}_{args.dataset}.csv")
    rows_by_seq = {r[0]: r for r in pose_results}
    drows_by_seq = {r[0]: r for r in depth_rows}
    all_seqs = [sd.name for sd in seq_dirs]
    with open(csv_path, "w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for seq in all_seqs:
            ate, rpe_t, rpe_r = (rows_by_seq[seq][1:] if seq in rows_by_seq else ("", "", ""))
            if seq in drows_by_seq:
                _, ar, d1, vp = drows_by_seq[seq]
            else:
                ar, d1, vp = "", "", ""
            def fmt(x): return f"{x:.6f}" if isinstance(x, float) else x
            f.write(f"{seq},{fmt(ate)},{fmt(rpe_t)},{fmt(rpe_r)},{fmt(ar)},{fmt(d1)},{vp}\n")
        if do_pose:
            f.write(f"AVERAGE(meanseq),{avg_ate:.6f},{avg_rpe_t:.6f},{avg_rpe_r:.6f},"
                    f"{abs_rel_mean if do_depth else ''},{d1_mean if do_depth else ''},\n")
        if do_depth:
            f.write(f"AVERAGE(vpixwt_depth),,,,{abs_rel_w:.6f},{d1_w:.6f},\n")

    print("\n==================== SUMMARY ====================")
    print(f"model={args.model} dataset={args.dataset} ({len(seq_dirs)} seqs)")
    if do_pose:
        print(f"POSE   (mean over seqs): ATE {avg_ate:.4f}  RPE-t {avg_rpe_t:.4f}  RPE-r {avg_rpe_r:.4f}")
    if do_depth:
        print(f"DEPTH  (mean over seqs): AbsRel {abs_rel_mean:.4f}  delta<1.25 {d1_mean:.4f}")
        print(f"DEPTH  (vpix-weighted) : AbsRel {abs_rel_w:.4f}  delta<1.25 {d1_w:.4f}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()

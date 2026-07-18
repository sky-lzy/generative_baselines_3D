"""Score a BASELINE's saved predictions (Geo4D / DA3) with the OFFICIAL π³ metric
code VERBATIM — the SAME harness our-model bridge (eval_ours_pi3.py) uses, so the
numbers are directly comparable.

Differences vs eval_ours_pi3.py (only two, both required by the protocol):
  1. DEPTH is already METRIC (NOT our normalized disparity) -> NO disparity decode.
     The baseline depth is passed straight to π³ depth_evaluation (align_with_scale
     solves the single per-sequence scale).
  2. The baseline saw the FULL native frame (not our 4:3 center-crop), so its depth
     covers the full GT field of view. We therefore bring pred to the GT frame
     (resize to full GT res) and crop BOTH pred AND GT to the same center-4:3 box
     (center_crop_box_for_aspect, aspect 320/240) before depth_evaluation — so the
     metric is computed on the identical region our models are scored on.

POSE is resolution-independent and IDENTICAL to eval_ours_pi3.eval_pose: GT traj
(load_traj) sliced to the model's frame_indices, pred c2w -> get_tum_poses, evo Sim3
eval_metrics (ATE / RPE-t / RPE-r). Aggregation = valid-pixel-weighted mean (depth)
and calculate_averages (pose), matching the π³ harness.
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
from relpose.evo_utils import load_traj, get_tum_poses, eval_metrics, calculate_averages
from utils.depth import (
    depth_evaluation, depth_read_sintel, depth_read_bonn, depth_read_kitti,
)

sys.path.insert(0, os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] for datasets._crop_utils
from datasets._crop_utils import center_crop_box_for_aspect, apply_spatial_crop

# Same DATASET_SPEC as eval_ours_pi3.py (π³ configs/data/*.yaml + EVAL_DEPTH_METADATA VERBATIM).
DATASET_SPEC = {
    "sintel": {
        "pose": {"anno": f"{PI3_ROOT}/data/sintel/training/camdata_left/{{seq}}", "format": "sintel"},
        "depth": {"dir": f"{PI3_ROOT}/data/sintel/training/depth/{{seq}}", "ext": "dpt",
                  "read": depth_read_sintel, "kwargs": {"max_depth": 70, "post_clip_max": 70}},
    },
    "tum": {"pose": {"anno": f"{PI3_ROOT}/data/tum/{{seq}}/groundtruth_90.txt", "format": "tum"}},
    "scannetv2": {"pose": {"anno": f"{PI3_ROOT}/data/scannetv2/{{seq}}/pose_90.txt", "format": "replica"}},
    "bonn": {"depth": {"dir": f"{PI3_ROOT}/data/bonn/rgbd_bonn_dataset/{{seq}}/depth_110",
                       "ext": "png", "read": depth_read_bonn, "kwargs": {"max_depth": 70}}},
    "kitti": {"depth": {"dir": f"{PI3_ROOT}/data/kitti/depth_selection/val_selection_cropped/"
                               f"groundtruth_depth_gathered/{{seq}}", "ext": "png",
                        "read": depth_read_kitti, "kwargs": {"max_depth": None}}},
}

# Target crop aspect = our model resolution (240 high, 320 wide) -> 4:3.
CROP_H, CROP_W = 240, 320


def eval_pose(pspec, seq, frame_indices, pred_c2w):
    # Mirror PI3 relpose/eval_dist.py: a degenerate GT trajectory makes evo's
    # orientations_quat_wxyz raise LinAlgError -> the official harness skips that
    # sequence. Return None so the caller drops it (same scenes the PI3 anchor used).
    try:
        gt_tum, gt_tt = load_traj(gt_traj_file=pspec["anno"].format(seq=seq),
                                  traj_format=pspec["format"], stride=1)
    except np.linalg.LinAlgError:
        print(f"  [skip] {seq}: GT trajectory load failed (LinAlgError) — matches PI3 harness skip")
        return None
    fi = np.asarray(frame_indices, dtype=np.int64)
    assert len(gt_tum) >= fi.max() + 1, f"{seq}: GT {len(gt_tum)} poses, need {fi.max()}"
    gt_traj = [gt_tum[fi], gt_tt[fi]]
    pred_traj = get_tum_poses([pred_c2w[i] for i in range(len(pred_c2w))])
    assert len(pred_traj[0]) == len(gt_traj[0]), f"{seq}: pred {len(pred_traj[0])} vs gt {len(gt_traj[0])}"
    fname = os.path.join(os.path.dirname(__file__), "results_baselines", "_tmp_eval_metric.txt")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ate, rpe_t, rpe_r = eval_metrics(pred_traj, gt_traj, seq=seq, filename=fname)
    return float(ate), float(rpe_t), float(rpe_r)


def eval_depth(dspec, seq, frame_indices, pred_metric):
    """pred_metric: (M, Hd, Wd) METRIC depth covering the full native frame."""
    gt_files = sorted(glob.glob(os.path.join(dspec["dir"].format(seq=seq), f"*.{dspec['ext']}")))
    fi = list(frame_indices)
    assert len(gt_files) >= max(fi) + 1, f"{seq}: {len(gt_files)} GT files, need {max(fi)}"
    gt = np.stack([dspec["read"](gt_files[i]) for i in fi], axis=0).astype(np.float64)  # (M,Hg,Wg)
    M, Hg, Wg = gt.shape
    # Bring baseline depth to the GT frame (full res), then crop BOTH to the same 4:3 box.
    pred_full = np.stack([
        cv2.resize(pred_metric[i].astype(np.float32), (Wg, Hg), interpolation=cv2.INTER_LINEAR)
        for i in range(M)
    ], axis=0).astype(np.float64)
    box = center_crop_box_for_aspect(Hg, Wg, CROP_H, CROP_W)
    gt_crop = apply_spatial_crop(gt, box)
    pred_crop = apply_spatial_crop(pred_full, box)
    results, *_ = depth_evaluation(pred_crop, gt_crop, align_with_scale=True, use_gpu=False,
                                   **dspec["kwargs"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["geo4d", "da3"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--preds_dir", default=str(Path(__file__).resolve().parent / "preds_baselines"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results_baselines"))
    args = ap.parse_args()

    spec = DATASET_SPEC[args.dataset]
    do_pose = "pose" in spec
    do_depth = "depth" in spec
    seq_root = Path(args.preds_dir) / args.baseline / args.dataset
    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No prediction seqs under {seq_root}")

    pose_results, depth_rows = [], []
    for sd in seq_dirs:
        seq = sd.name
        frame_indices = json.load(open(sd / "frames.json"))["frame_indices"]
        ate = rpe_t = rpe_r = float("nan")
        abs_rel = delta1 = float("nan"); vpix = 0
        if do_pose and (sd / "pred_c2w.npy").exists():
            pred_c2w = np.load(sd / "pred_c2w.npy")
            res = eval_pose(spec["pose"], seq, frame_indices, pred_c2w)
            if res is None:
                continue  # GT load failed -> skip (PI3-faithful)
            ate, rpe_t, rpe_r = res
            pose_results.append((seq, ate, rpe_t, rpe_r))
        if do_depth and (sd / "pred_depth_metric.npy").exists():
            pred_metric = np.load(sd / "pred_depth_metric.npy")
            dres = eval_depth(spec["depth"], seq, frame_indices, pred_metric)
            abs_rel = dres["Abs Rel"]; delta1 = dres["δ < 1.25"]; vpix = dres["valid_pixels"]
            depth_rows.append((seq, abs_rel, delta1, vpix))
        print(f"{seq:34s}  ATE {ate:8.4f}  RPE-t {rpe_t:8.4f}  RPE-r {rpe_r:8.4f}  "
              f"| AbsRel {abs_rel:8.4f}  d1 {delta1:8.4f}  vpix {vpix}", flush=True)

    avg_ate = avg_rpe_t = avg_rpe_r = float("nan")
    abs_rel_mean = d1_mean = abs_rel_w = d1_w = float("nan")
    if do_pose and pose_results:
        avg_ate, avg_rpe_t, avg_rpe_r = calculate_averages(pose_results)
    if do_depth and depth_rows:
        abs_arr = np.array([r[1] for r in depth_rows], float)
        d1_arr = np.array([r[2] for r in depth_rows], float)
        w = np.array([r[3] for r in depth_rows], float)
        abs_rel_mean = float(abs_arr.mean()); d1_mean = float(d1_arr.mean())
        abs_rel_w = float(np.average(abs_arr, weights=w)); d1_w = float(np.average(d1_arr, weights=w))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.baseline}_{args.dataset}.csv")
    rows_by_seq = {r[0]: r for r in pose_results}
    drows_by_seq = {r[0]: r for r in depth_rows}
    with open(csv_path, "w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for sd in seq_dirs:
            seq = sd.name
            ate, rpe_t, rpe_r = (rows_by_seq[seq][1:] if seq in rows_by_seq else ("", "", ""))
            if seq in drows_by_seq:
                _, ar, d1, vp = drows_by_seq[seq]
            else:
                ar, d1, vp = "", "", ""
            def fmt(x): return f"{x:.6f}" if isinstance(x, float) else x
            f.write(f"{seq},{fmt(ate)},{fmt(rpe_t)},{fmt(rpe_r)},{fmt(ar)},{fmt(d1)},{vp}\n")
        if do_pose and pose_results:
            f.write(f"AVERAGE(meanseq),{avg_ate:.6f},{avg_rpe_t:.6f},{avg_rpe_r:.6f},"
                    f"{abs_rel_mean if do_depth else ''},{d1_mean if do_depth else ''},\n")
        if do_depth and depth_rows:
            f.write(f"AVERAGE(vpixwt_depth),,,,{abs_rel_w:.6f},{d1_w:.6f},\n")

    print("\n==================== SUMMARY ====================")
    print(f"baseline={args.baseline} dataset={args.dataset} ({len(seq_dirs)} seqs)")
    if do_pose and pose_results:
        print(f"POSE   (mean over seqs): ATE {avg_ate:.4f}  RPE-t {avg_rpe_t:.4f}  RPE-r {avg_rpe_r:.4f}")
    if do_depth and depth_rows:
        print(f"DEPTH  (mean over seqs): AbsRel {abs_rel_mean:.4f}  delta<1.25 {d1_mean:.4f}")
        print(f"DEPTH  (vpix-weighted) : AbsRel {abs_rel_w:.4f}  delta<1.25 {d1_w:.4f}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()

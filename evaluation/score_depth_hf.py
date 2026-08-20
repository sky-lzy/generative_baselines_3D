"""UNIFIED, PIXEL-IDENTICAL depth scorer for the July-3 home-field eval.

Every method (ours + Geo4D/PI3/DA3) is scored on the SAME grid:
  1. GT depth read at native res, cropped to our aspect-preserving 4:3 center box,
     then resized to a COMMON EVAL resolution EVAL_H x EVAL_W (default 240x320 = our res).
  2. pred brought to that SAME 240x320 crop grid:
       ours     : decode normalized disparity -> metric-proportional -> resize to 240x320
       baseline : metric depth (full frame) -> resize to GT res -> crop 4:3 box -> resize 240x320
  => baselines are DOWNSAMPLED to our resolution and evaluated on the EXACT same pixels as ours.
  3. three alignments applied identically (scale / affine_lsq / lads) via align_ablation.

Writes eval_pi3/results_ablation_hf/<method>_<dataset>.csv with one AVERAGE row per alignment.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH]
sys.path.append(str(Path(__file__).resolve().parents[1] / "vendor"))  # self-contained fallback: datasets._geometry_builder resolves here when no VWM checkout is present
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_pi3"))
sys.path.insert(0, os.environ.get("GEN3D_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines_3D"))  # [standardized_eval PATCH]
from eval_common_v3 import disparity_norm_to_metric_depth
from datasets._crop_utils import center_crop_box_for_aspect, apply_spatial_crop
PI3_ROOT = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")  # [standardized_eval PATCH] GT-data root
sys.path.insert(0, PI3_ROOT)
sys.path.insert(0, os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parent / "pi3_metrics")))  # [standardized_eval PATCH] metric imports resolve to the local verbatim copy
from utils.depth import depth_read_sintel, depth_read_bonn, depth_read_kitti
sys.path.insert(0, str(HERE))
from align_ablation import score_all_alignments, ALIGNERS

EVAL_H, EVAL_W = 240, 320          # common eval grid = our resolution; baselines downsampled to it

# depth GT layout + max_depth mask (mirrors π³ EVAL_DEPTH_METADATA)
DEPTH = {
  "sintel": {"dir": f"{PI3_ROOT}/data/sintel/training/depth/{{seq}}", "ext":"dpt", "read":depth_read_sintel, "max_depth":70},
  "bonn":   {"dir": f"{PI3_ROOT}/data/bonn/rgbd_bonn_dataset/{{seq}}/depth_110", "ext":"png", "read":depth_read_bonn, "max_depth":70},
  "kitti":  {"dir": f"{PI3_ROOT}/data/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/{{seq}}", "ext":"png", "read":depth_read_kitti, "max_depth":None},
}

def _norm_meta(seq_dir):
    """Per-scene TRAINING normalization, written by run_ours_pi3.py. Absent => legacy disparity."""
    p = Path(seq_dir) / "norm_meta.json"
    if not p.exists():
        return {}
    try:
        return {k: v for k, v in json.load(open(p)).items() if v is not None}
    except Exception:
        return {}


def decode_ours(pred_norm, seq_dir=None):
    """[-1,1] depth channel -> metric-proportional depth, using the map the model was TRAINED
    with.

    The alignment-based metrics below (scale / affine-LSQ / LADS) absorb a global SCALE, which is
    why per-clip models historically needed no decode flags. They do NOT absorb the NONLINEARITY:
    signed disparity `2/(1+u)-1` and the log map `1-2*clip(a*ln(1+u),0,1)` are different
    functions, so scoring a log-map arm through the disparity inverse is silently wrong -- it
    produces finite, plausible numbers and no error. Hence the dispatch on norm_meta.json.
    """
    meta = _norm_meta(seq_dir) if seq_dir is not None else {}
    mode = meta.get("scale_mode")
    if not mode or mode == "global_metric":
        # Legacy / F / strategy-D: the historical path, byte-for-byte unchanged.
        return disparity_norm_to_metric_depth(pred_norm, scale=1.0, percentile_clip=2.0)
    import torch as _t
    from datasets._geometry_builder import invert_depth_channel

    class _Cfg(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    # scale=1.0: the per-clip scalar is unknown at scoring time and is absorbed by the aligners.
    # Only the shape of the map matters here, which is exactly what we are correcting for.
    d, unrec = invert_depth_channel(_t.from_numpy(np.asarray(pred_norm, dtype=np.float64)),
                                    1.0, _Cfg(meta))
    d = d.numpy()
    # Leave clipped pixels at their BOUNDARY value rather than NaN. The pi3 depth metric masks
    # only on ground_truth>0 (pi3_metrics/utils/depth.py:247) and does NOT mask NaN in the
    # prediction, so NaN propagates through the mean and turns the whole sequence into nan --
    # observed on real Sintel: sequences with any clipped pixel scored AbsRel=nan while
    # clip-free ones scored normally. The clipped value is the best available estimate and the
    # unrecoverable fraction is <0.5% (measured), so this is both finite and faithful; it also
    # matches how the legacy disparity path behaves (percentile_clip bounds, never NaNs).
    _ = unrec
    return d

def gt_on_grid(spec, seq, fi):
    gt_files = sorted(glob.glob(os.path.join(spec["dir"].format(seq=seq), f"*.{spec['ext']}")))
    gt = np.stack([spec["read"](gt_files[i]) for i in fi], axis=0).astype(np.float64)  # (M,Hg,Wg)
    box = center_crop_box_for_aspect(gt.shape[1], gt.shape[2], EVAL_H, EVAL_W)
    gtc = apply_spatial_crop(gt, box)                                                  # (M,Hc,Wc)
    gtr = np.stack([cv2.resize(gtc[i], (EVAL_W, EVAL_H), interpolation=cv2.INTER_NEAREST) for i in range(len(gtc))])
    return gtr, box

def pred_on_grid_ours(pred_norm, seq_dir=None):
    d = decode_ours(pred_norm, seq_dir)                                                          # (M,Hp,Wp)
    return np.stack([cv2.resize(d[i], (EVAL_W, EVAL_H), interpolation=cv2.INTER_CUBIC) for i in range(len(d))])

def pred_on_grid_baseline(pred_metric, box, gtHW):
    Hg, Wg = gtHW
    full = np.stack([cv2.resize(pred_metric[i].astype(np.float32), (Wg, Hg), interpolation=cv2.INTER_LINEAR) for i in range(len(pred_metric))]).astype(np.float64)
    crop = apply_spatial_crop(full, box)
    return np.stack([cv2.resize(crop[i], (EVAL_W, EVAL_H), interpolation=cv2.INTER_CUBIC) for i in range(len(crop))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["ours","baseline"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DEPTH))
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--out_dir", default=str(REPO/"eval_pi3"/"results_ablation_hf"))
    args = ap.parse_args()
    spec = DEPTH[args.dataset]
    root = Path(args.preds_dir)/args.model/args.dataset
    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    per = {k: [] for k in ALIGNERS}     # alignment -> list of (absrel,d1,vpix)
    for sd in seq_dirs:
        try:
            fi = json.load(open(sd/"frames.json"))["frame_indices"]
            gtr, box = gt_on_grid(spec, sd.name, fi)
            gtHW = None
            if args.kind == "ours":
                pn = np.load(sd/"pred_depth_norm.npy"); pr = pred_on_grid_ours(pn, sd)
            else:
                pm = np.load(sd/"pred_depth_metric.npy")
                gt_files = sorted(glob.glob(os.path.join(spec["dir"].format(seq=sd.name), f"*.{spec['ext']}")))
                g0 = spec["read"](gt_files[fi[0]]); gtHW=(g0.shape[0],g0.shape[1])
                pr = pred_on_grid_baseline(pm, box, gtHW)
            res = score_all_alignments(pr, gtr, max_depth=spec["max_depth"])
            for k,(a,d,v) in res.items():
                if np.isfinite(a): per[k].append((a,d,v))
        except Exception as e:
            print(f"  {sd.name}: SKIP ({type(e).__name__}: {e})", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = Path(args.out_dir)/f"{args.model}_{args.dataset}.csv"
    with open(out,"w") as f:
        f.write("alignment,AbsRel_mean,d1_mean,AbsRel_vpixwt,d1_vpixwt,n_seq\n")
        for k in ALIGNERS:
            rows=per[k]
            if not rows: f.write(f"{k},,,,,0\n"); continue
            a=np.array([r[0] for r in rows]); d=np.array([r[1] for r in rows]); w=np.array([r[2] for r in rows],float)
            f.write(f"{k},{a.mean():.6f},{d.mean():.6f},{np.average(a,weights=w):.6f},{np.average(d,weights=w):.6f},{len(rows)}\n")
    print(f"[{args.model}/{args.dataset}] wrote {out}")
    for k in ALIGNERS:
        if per[k]:
            a=np.array([r[0] for r in per[k]]); d=np.array([r[1] for r in per[k]])
            print(f"   {k:11s} AbsRel {a.mean():.3f}  d1 {d.mean():.3f}  (n={len(per[k])})")

if __name__=="__main__":
    main()

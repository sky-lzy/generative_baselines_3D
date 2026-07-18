"""Score ScanNet++ camera pose: {scene}_pred_c2w.npy vs {scene}_gt_c2w.npy (ViPE c2w,
first-frame-relative, saved from the SAME dataset batch by run_eval_scannetpp.py).
Same π³ evo Sim(3) ATE/RPE metric as every other pose dataset (eval_re10k_dist50.py style)."""
import argparse, os, sys
from pathlib import Path
import numpy as np
PI3 = os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parent / "pi3_metrics"))  # [standardized_eval PATCH] metric imports from local verbatim copy
sys.path.insert(0, PI3)
from relpose.evo_utils import get_tum_poses, eval_metrics, calculate_averages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    tag = a.tag or a.model
    root = Path(a.preds_dir)/a.model/"scannetpp"
    scenes = sorted({p.name[:-len("_pred_c2w.npy")] for p in root.glob("*_pred_c2w.npy")},
                    key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
    os.makedirs(a.out_dir, exist_ok=True); tmp = f"{a.out_dir}/_tmp_{tag}.txt"
    res = []
    for s in scenes:
        gf = root/f"{s}_gt_c2w.npy"
        if not gf.exists(): continue
        pred = np.load(root/f"{s}_pred_c2w.npy").astype(np.float64)
        if pred.shape[1:] == (3, 4):   # pad (T,3,4) -> (T,4,4)
            pad = np.tile(np.eye(4), (pred.shape[0], 1, 1)); pad[:, :3] = pred; pred = pad
        gt = np.load(gf).astype(np.float64)
        if pred.shape[0] != gt.shape[0]: continue
        try:
            at,rt,rr = eval_metrics(get_tum_poses([pred[k] for k in range(len(pred))]),
                                    get_tum_poses([gt[k] for k in range(len(gt))]),
                                    seq=s, filename=tmp)
            res.append((s,float(at),float(rt),float(rr)))
        except Exception as e:
            print(f"{s}: SKIPPED ({type(e).__name__}: {e})")
    if not res: print(f"[{tag}] NO scannetpp pose results"); return
    at,rt,rr = calculate_averages(res)
    with open(f"{a.out_dir}/{tag}_scannetpp_pose.csv","w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for s,x,y,z in res: f.write(f"{s},{x:.6f},{y:.6f},{z:.6f},,,\n")
        f.write(f"AVERAGE(meanseq),{at:.6f},{rt:.6f},{rr:.6f},,,\n")
    print(f"[{tag}] scannetpp-pose n={len(res)} ATE {at:.4f} RPE-t {rt:.4f} RPE-r {rr:.4f}")
if __name__=="__main__": main()

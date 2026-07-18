"""Score 50-frame HOME-FIELD RE10K: pred_c2w vs GT = annotations[the 50 contiguous line-indices].
Same π³ evo Sim(3) ATE/RPE metric as eval_re10k_dist.py; only the GT frame set differs (50 middle
contiguous frames instead of the 10 seq-id-map frames)."""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent.parent
PI3 = os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parent / "pi3_metrics"))  # [standardized_eval PATCH] metric imports from local verbatim copy
sys.path.insert(0, PI3)
from relpose.evo_utils import get_tum_poses, eval_metrics, calculate_averages
ANN = os.environ.get("RE10K_ANN", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_pi3frames")  # [standardized_eval PATCH]
HF50 = os.environ.get("RE10K_HF50", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50")  # [standardized_eval PATCH]
def to44(m):
    m = np.asarray(m, float);
    if m.shape==(3,4): o=np.eye(4); o[:3]=m; return o
    return m
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    root = Path(a.preds_dir)/a.model/"re10k_hf50"
    seqs = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    os.makedirs(a.out_dir, exist_ok=True); tmp = f"{a.out_dir}/_tmp_{a.tag}.txt"
    res = []
    for s in seqs:
        pf = root/s/"pred_c2w.npy"
        if not pf.exists(): continue
        idxs = json.load(open(f"{HF50}/{s}/frames.json"))["line_indices"]
        anno = json.load(open(f"{ANN}/{s}/annotations.json"))
        pred = np.load(pf).astype(np.float64)
        idxs = idxs[:pred.shape[0]]
        if pred.shape[0] != len(idxs): continue
        gt = [np.linalg.inv(to44(anno[i]["extrinsics"])) for i in idxs]
        try:
            at,rt,rr = eval_metrics(get_tum_poses([to44(pred[k]) for k in range(len(idxs))]),
                                    get_tum_poses(gt), seq=s, filename=tmp)
            res.append((s,float(at),float(rt),float(rr)))
        except Exception: pass
    if not res: print(f"[{a.tag}] NO results"); return
    at,rt,rr = calculate_averages(res)
    with open(f"{a.out_dir}/{a.tag}_re10k.csv","w") as f:
        f.write("seq,ATE,RPE_trans,RPE_rot,AbsRel,delta_1.25,valid_pixels\n")
        for s,x,y,z in res: f.write(f"{s},{x:.6f},{y:.6f},{z:.6f},,,\n")
        f.write(f"AVERAGE(meanseq),{at:.6f},{rt:.6f},{rr:.6f},,,\n")
    print(f"[{a.tag}] re10k-HF50 n={len(res)} ATE {at:.4f} RPE-t {rt:.4f} RPE-r {rr:.4f}")
if __name__=="__main__": main()

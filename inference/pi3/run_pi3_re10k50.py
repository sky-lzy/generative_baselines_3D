"""Run PI3's OWN model on RE10K (the 10 seq-id-map frames per seq, ascending line-idx)
and save camera-to-world predictions, so RE10K can be scored with BOTH the angular metric
and the new distance metric (eval_re10k_dist.py), apples-to-apples with every other method.

Uses the OFFICIAL π³ inference (utils.interfaces.infer_cameras_c2w, load_img_size=512) —
the SAME call run_pi3_c50.py uses for TUM/ScanNet pose. Pose-only; GT is owned by the
scorer (annotations.json w2c). Writes:
    eval_pi3/preds_baselines/pi3/re10k/<seq>/pred_c2w.npy   (T,3,4 or T,4,4) c2w
    eval_pi3/preds_baselines/pi3/re10k/<seq>/frames.json

Run:
  CUDA_VISIBLE_DEVICES=<g> HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga \
    mamba run -n test2 python3 eval_pi3/run_pi3_re10k.py --seqs_file eval_pi3/re10k_common83.txt
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(os.environ.get("VWM_REPO", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"))  # [standardized_eval PATCH] repo root via env
PI3_ROOT = os.environ.get("PI3_ROOT", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")  # [standardized_eval PATCH]
sys.path.insert(0, PI3_ROOT)
import rootutils
rootutils.setup_root(PI3_ROOT, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_cameras_c2w

IMG_TMPL = os.environ.get("RE10K_HF50", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50") + "/{seq}/images"  # [standardized_eval PATCH]


def get_all_seqs():
    from omegaconf import OmegaConf
    base = OmegaConf.load(REPO / "configurations/dataset/pi3seq.yaml")
    merged = OmegaConf.merge(base, OmegaConf.load(REPO / "configurations/dataset/pi3seq_re10k.yaml"))
    return list(merged.seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs_file", default=None, help="optional subset; default = all configured re10k seqs")
    ap.add_argument("--out_dir", default=str(REPO / "eval_pi3" / "preds_baselines"))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    seqs = ([l.strip() for l in open(args.seqs_file) if l.strip()]
            if args.seqs_file else get_all_seqs())
    cfg = SimpleNamespace(load_img_size=512, device="cuda", verbose=False)
    model = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
    print(f"Loaded Pi3 for RE10K pose ({len(seqs)} seqs)", flush=True)

    root = Path(args.out_dir) / "pi3" / "re10k_hf50"
    done = 0
    for seq in seqs:
        sd = root / seq
        if (sd / "pred_c2w.npy").exists() and not args.rerun:
            done += 1; continue
        files = sorted(glob.glob(os.path.join(IMG_TMPL.format(seq=seq), "*.png")))
        if len(files) < 2:
            print(f"  [skip] {seq}: {len(files)} frames", flush=True); continue
        sd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with torch.no_grad():
            c2w, _ = infer_cameras_c2w(files, model, cfg)
        c2w = np.asarray(c2w.cpu(), dtype=np.float64)
        np.save(sd / "pred_c2w.npy", c2w)
        json.dump({"seq": seq, "n_used": len(files),
                   "frames": [os.path.basename(f) for f in files]},
                  open(sd / "frames.json", "w"))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(seqs)} ({time.time()-t0:.1f}s/seq)", flush=True)
    print(f"DONE pi3 re10k: {done} seqs -> {root}", flush=True)


if __name__ == "__main__":
    main()

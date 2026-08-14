#!/usr/bin/env python3
"""Is each method's k-th output actually AT the k-th requested camera? A fairness instrument.

The scorer pairs a method's k-th predicted frame with the k-th test_id. Two very different things
break that pairing, and PSNR alone cannot tell them apart from "the model is bad":

  PERMUTATION   the engine emits targets in a different order than we assume. Would tank the score
                for a reason that has nothing to do with model quality.
  UNDER/OVER-TRAVEL  the method renders a coherent trajectory but at the wrong SCALE, so frame k
                lands where frame j != k should be. For SEVA that is exactly what --camera_scale
                controls, and for us --moment_scale_mult. A method whose sweep grid does not reach
                its own optimum is being under-searched, which is the classic way a "fair"
                benchmark quietly favours whoever happened to be tuned.

Diagnostic: for each predicted frame, search the WHOLE GT clip for the frame it best matches. Then
  * best-match == assigned id (or an immediate neighbour)  -> correctly placed
  * best-matches monotonically increasing but compressed toward 0 -> UNDER-travel, raise the scale
  * ... stretched beyond the assigned ids -> OVER-travel, lower the scale
Adjacent-frame disagreement is benign: in slow scenes neighbouring GT frames are near-identical, so
argmin picks between near-ties. Only the systematic trend matters.

    python3 nvs/evaluation/verify_frame_alignment.py --cells seva__re10k128_4dim__ncf1__s2 ...
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS / "evaluation"))
from score_nvs_fair import read_gt, read_ours, to_metric_domain  # noqa: E402

SEVA_WORK = Path("/n/lab_storage/ydu_lab/Lab/akiruga/stable-virtual-camera"
                 "/work_dirs/demo/img2img")
CELL = re.compile(r"^(?P<method>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<ncf>\d)__s(?P<scale>[\d.]+)$")


def load_seva(cell, scene, n):
    for d in (Path(NVS / "preds_fair") / cell, SEVA_WORK / f"fair_{cell}"):
        ps = sorted(glob.glob(str(d / scene / "samples-rgb" / "*.png")))
        if len(ps) >= n:
            return torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB")))
                                .float().permute(2, 0, 1) / 255. for p in ps[:n]])
    return None


def analyse_json(cell, scenes_fair, metric_size=256, n_scenes=3):
    """Same measurement as analyse(), returned as data so the report can table it."""
    g = CELL.match(cell)
    if not g:
        return None
    g = g.groupdict(); ds, ncf = g["ds"], int(g["ncf"])
    sroot = Path(scenes_fair) / ds
    nfr = json.load(open(sroot / "_clip.json"))["n_frames"]
    scenes = sorted(d for d in os.listdir(sroot) if (sroot / d).is_dir())
    ratios, exacts, tot = [], 0, 0
    done = 0
    for scene in scenes:
        if done >= n_scenes:
            break
        ids = sorted(json.load(open(sroot / scene / f"train_test_split_{ncf}.json"))["test_ids"])
        if g["method"] == "seva":
            pred = load_seva(cell, scene, len(ids))
        else:
            pred, _ = read_ours(str(Path(NVS / "preds_fair") / cell /
                                    f"sample_{scenes.index(scene):05d}"), ids)
        if pred is None:
            continue
        done += 1
        gt_all = to_metric_domain(read_gt(str(sroot / scene), list(range(nfr))), metric_size)
        p = to_metric_domain(pred, metric_size)
        best = [int(torch.argmin(((gt_all - p[k:k + 1]) ** 2).mean(dim=(1, 2, 3))))
                for k in range(len(ids))]
        a_ = np.array(ids, float); b_ = np.array(best, float)
        ratios.append(float((a_ @ b_) / (a_ @ a_)))
        exacts += sum(1 for k, b in enumerate(best) if b == ids[k]); tot += len(ids)
    if not ratios:
        return None
    return dict(cell=cell, method=g["method"], ds=ds, ncf=ncf, scale=float(g["scale"]),
                travel_ratio=float(np.mean(ratios)), exact=exacts, total=tot, scenes=done)


def analyse(cell, scenes_fair, metric_size=256, n_scenes=3):
    g = CELL.match(cell)
    if not g:
        return f"{cell}: unparseable"
    g = g.groupdict()
    ds, ncf = g["ds"], int(g["ncf"])
    sroot = Path(scenes_fair) / ds
    nfr = json.load(open(sroot / "_clip.json"))["n_frames"]
    scenes = sorted(d for d in os.listdir(sroot) if (sroot / d).is_dir())
    out, agg = [f"\n{cell}"], []
    done = 0
    for scene in scenes:
        if done >= n_scenes:
            break
        ids = sorted(json.load(open(sroot / scene / f"train_test_split_{ncf}.json"))["test_ids"])
        if g["method"] == "seva":
            pred = load_seva(cell, scene, len(ids))
        else:
            idx = scenes.index(scene)
            pred, _ = read_ours(str(Path(NVS / "preds_fair") / cell / f"sample_{idx:05d}"), ids)
        if pred is None:
            continue
        done += 1
        gt_all = to_metric_domain(read_gt(str(sroot / scene), list(range(nfr))), metric_size)
        p = to_metric_domain(pred, metric_size)
        best = []
        for k in range(len(ids)):
            mse = ((gt_all - p[k:k + 1]) ** 2).mean(dim=(1, 2, 3))
            best.append(int(torch.argmin(mse)))
        exact = sum(1 for k, b in enumerate(best) if b == ids[k])
        near = sum(1 for k, b in enumerate(best) if abs(b - ids[k]) <= 1)
        # travel ratio: slope of best-match vs assigned id through the origin
        a = np.array(ids, float); b = np.array(best, float)
        ratio = float((a @ b) / (a @ a)) if (a @ a) > 0 else float("nan")
        agg.append(ratio)
        out.append(f"  {scene}  exact {exact}/{len(ids)}  within±1 {near}/{len(ids)}  "
                   f"travel_ratio {ratio:.3f}")
        out.append(f"     assigned  {ids[:10]}")
        out.append(f"     best-match{best[:10]}")
    if agg:
        m = float(np.mean(agg))
        verdict = ("correctly placed" if 0.9 <= m <= 1.1 else
                   f"{'UNDER' if m < 1 else 'OVER'}-TRAVEL by {abs(1 - m) * 100:.0f}% "
                   f"-> {'raise' if m < 1 else 'lower'} the scale")
        out.append(f"  => mean travel_ratio {m:.3f}  ({verdict})")
    else:
        out.append("  (no scenes readable yet)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--scenes_fair",
                    default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
    ap.add_argument("--n_scenes", type=int, default=3)
    ap.add_argument("--json_out", default=None,
                    help="also write the measurements as JSON for the report")
    a = ap.parse_args()
    if a.json_out:
        rows = [r for r in (analyse_json(c, a.scenes_fair, n_scenes=a.n_scenes) for c in a.cells)
                if r]
        Path(a.json_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {a.json_out} ({len(rows)} cells)")
        return
    print("FRAME-ALIGNMENT / TRAVEL-SCALE DIAGNOSTIC")
    print("travel_ratio ~1.0 = frame k lands at camera k;  <1 under-travels;  >1 over-travels")
    for c in a.cells:
        print(analyse(c, a.scenes_fair, n_scenes=a.n_scenes))


if __name__ == "__main__":
    main()

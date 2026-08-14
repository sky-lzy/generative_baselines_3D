#!/usr/bin/env python3
"""Score the 10-scene tuning cells and pick each method's sampler settings for the full run.

Scores every preds_fair/TUNE__* directory with the SAME scorer the real benchmark uses
(score_nvs_fair.py — one code path, centre square crop, lossless readers), then ranks within each
(method, dataset, ncf) group and writes the winners to nvs/configs/fair_tuned.yaml.

Selection metric is PSNR by default, but LPIPS is printed alongside because guidance trades them
off in opposite directions: raising CFG sharpens (LPIPS improves) while pushing the mean away from
the conditional expectation (PSNR degrades). Picking on PSNR alone and then reporting LPIPS as a win
would be selection bias, so --metric makes the choice explicit and the full grid is always printed.

    python3 nvs/evaluation/score_tune.py                 # score + rank + write fair_tuned.yaml
    python3 nvs/evaluation/score_tune.py --rank_only     # re-rank from existing CSVs
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
SCORER = NVS / "evaluation" / "score_nvs_fair.py"

# TUNE__<method>__<dataset>__ncf<k>__<settings>
PAT = re.compile(r"^TUNE__(?P<method>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<ncf>\d)__(?P<set>.+)$")


def parse_settings(s):
    """'hg0.5_lg0' -> {'hg': 0.5, 'lg': 0.0};  'cfg6' -> {'cfg': 6.0}"""
    out = {}
    for part in s.split("_"):
        m = re.match(r"^([a-z]+)([-\d.]+)$", part)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def read_avg(p):
    for r in csv.DictReader(open(p)):
        if r["scene"] == "AVERAGE":
            return {k: float(r[k]) for k in ("psnr", "ssim", "lpips")} | {"n": int(r["n_views"])}
    return None


def n_scored(p):
    return sum(1 for r in csv.DictReader(open(p)) if r["scene"] != "AVERAGE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(NVS / "preds_fair"))
    ap.add_argument("--results", default=str(NVS / "results_fair"))
    ap.add_argument("--scenes_fair",
                    default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
    ap.add_argument("--metric_size", type=int, default=256)
    ap.add_argument("--metric", default="psnr", choices=["psnr", "lpips", "ssim"],
                    help="which metric selects the winner (higher better for psnr/ssim)")
    ap.add_argument("--rank_only", action="store_true")
    ap.add_argument("--out_yaml", default=str(NVS / "configs" / "fair_tuned.yaml"))
    a = ap.parse_args()

    cells = sorted(d for d in os.listdir(a.preds) if d.startswith("TUNE__"))
    if not cells:
        sys.exit(f"no TUNE__* cells under {a.preds} — nothing has finished yet")
    os.makedirs(a.results, exist_ok=True)

    rows = []
    for c in cells:
        m = PAT.match(c)
        if not m:
            print(f"[skip] unparseable cell name: {c}"); continue
        g = m.groupdict()
        csv_p = os.path.join(a.results, c + ".csv")
        if not a.rank_only:
            cmd = [sys.executable, str(SCORER),
                   "--run_dir", os.path.join(a.preds, c),
                   "--method", "seva" if g["method"] == "seva" else "ours",
                   "--scenes_root", os.path.join(a.scenes_fair, g["ds"]),
                   "--split", g["ncf"], "--metric_size", str(a.metric_size),
                   "--out_csv", csv_p]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[fail] {c}\n  " + "\n  ".join(r.stdout.strip().splitlines()[-4:]))
                continue
        if not os.path.exists(csv_p):
            continue
        avg = read_avg(csv_p)
        if not avg:
            continue
        rows.append(dict(cell=c, scenes=n_scored(csv_p), **g, **parse_settings(g["set"]), **avg))

    if not rows:
        sys.exit("nothing scored")

    # ---- print the FULL grid per group, then the pick ------------------------------------
    better = (lambda x, y: x > y) if a.metric in ("psnr", "ssim") else (lambda x, y: x < y)
    groups, picks = {}, {}
    for r in rows:
        groups.setdefault((r["method"], r["ds"], int(r["ncf"])), []).append(r)

    print(f"\n{'='*96}\nTUNING GRID  (selection metric: {a.metric})\n{'='*96}")
    for key in sorted(groups):
        meth, ds, ncf = key
        rs = sorted(groups[key], key=lambda r: r["set"])
        print(f"\n{meth}  {ds}  ncf{ncf}")
        print(f"  {'settings':<18}{'n':>4}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
        best = None
        for r in rs:
            flag = ""
            print(f"  {r['set']:<18}{r['scenes']:>4}{r['psnr']:>9.3f}{r['ssim']:>9.4f}"
                  f"{r['lpips']:>9.4f}{flag}")
            if best is None or better(r[a.metric], best[a.metric]):
                best = r
        # a grid where cells scored different scene counts is not comparable
        ns = {r["scenes"] for r in rs}
        if len(ns) > 1:
            print(f"  !! UNEQUAL SCENE COUNTS {sorted(ns)} — this grid is NOT comparable yet")
        print(f"  -> pick: {best['set']}  ({a.metric} {best[a.metric]:.4f})")
        picks[key] = best

    # ---- write the tuned config ----------------------------------------------------------
    out = {"_note": f"written by score_tune.py; selection metric={a.metric}; "
                    f"grid ran on {max(r['scenes'] for r in rows)} scenes",
           "tuned": {}}
    for (meth, ds, ncf), b in picks.items():
        d = out["tuned"].setdefault(ds, {}).setdefault(f"ncf{ncf}", {})
        if meth == "seva":
            d["seva"] = {"cfg": b.get("cfg", 2.0)}
        else:
            d[meth] = {"hist_guidance": b.get("hg", 1.0), "lang_guidance": b.get("lg", 0.0)}
    Path(a.out_yaml).write_text(json.dumps(out, indent=2))   # JSON is valid YAML
    print(f"\nwrote {a.out_yaml}")


if __name__ == "__main__":
    main()

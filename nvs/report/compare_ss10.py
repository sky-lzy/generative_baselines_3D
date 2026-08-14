#!/usr/bin/env python3
"""Report the deep 20-vs-20 single-view scale search on the 10 scenes where we currently win.

Prints, per benchmark: both full scale curves, each side's best, and a PAIRED per-scene comparison
on the same 10 scenes. Two guards that matter more than the headline number:

  EDGE CHECK      if a side's best point is the first or last of its grid, that side is STILL
                  truncated and its number remains a lower bound. This is exactly the defect that
                  made our original 50-frame 1-view result (17.220 at the grid's lower edge) an
                  underestimate, so it is checked mechanically rather than eyeballed.

  SELECTION BIAS  these 10 scenes were chosen BECAUSE we beat SEVA on them in the 128-scene pass, so
                  the baseline margin is printed alongside. The interesting quantity is how much of
                  that margin SURVIVES once both sides are fully searched -- not the margin itself.

    python3 nvs/report/compare_ss10.py
"""
import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
CELL = re.compile(r"^(?P<method>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<ncf>\d)"
                  r"__s(?P<scale>[\d.]+)__ss10$")
BENCH = [("re10k128_4dim", 1, "4DiM split · 1 input view"),
         ("re10k128_50f", 1, "50-frame clip · 1 input view")]


def rows_of(p):
    out, avg = {}, None
    for r in csv.DictReader(open(p)):
        (out if r["scene"] != "AVERAGE" else {}).setdefault(r["scene"], r)
        if r["scene"] == "AVERAGE":
            avg = r
    return avg, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(NVS / "results_ss10"))
    a = ap.parse_args()
    man_p = Path(a.results) / "_manifest.json"
    man = json.load(open(man_p)) if man_p.exists() else {}

    data = {}
    for f in sorted(Path(a.results).glob("*__ss10.csv")):
        m = CELL.match(f.stem)
        if not m:
            continue
        g = m.groupdict()
        avg, rows = rows_of(f)
        if not avg or not rows:
            continue
        data.setdefault((g["ds"], int(g["ncf"]), g["method"]), []).append(
            dict(scale=float(g["scale"]), psnr=float(avg["psnr"]), ssim=float(avg["ssim"]),
                 lpips=float(avg["lpips"]), n=len(rows), rows=rows))

    print("=" * 100)
    print("DEEP SCALE SEARCH — 10 single-view scenes where we win at baseline")
    print("SEVA: its paper's sweep, camera_scale 0.1..2.0 step 0.1, --cfg 6.0")
    print("Ours: 20 points placed from the measured curve, hist_guidance 0.5")
    print("=" * 100)

    for ds, ncf, title in BENCH:
        key = f"{ds}__ncf{ncf}"
        print(f"\n{title}")
        info = man.get(key, {})
        if info:
            bl = list(info.get("baseline_margins_dB", {}).values())
            if bl:
                print(f"  baseline margin on these scenes: mean {st.mean(bl):+.3f} dB "
                      f"(range {min(bl):+.3f} … {max(bl):+.3f})")
        picks = {}
        for meth in ("F_5b", "seva"):
            ent = sorted(data.get((ds, ncf, meth), []), key=lambda e: e["scale"])
            if not ent:
                print(f"  {meth}: no results yet"); continue
            b = max(ent, key=lambda e: e["psnr"])
            picks[meth] = b
            lo, hi = ent[0]["scale"], ent[-1]["scale"]
            edge = ("  !! BEST AT GRID EDGE — still truncated, this is a LOWER BOUND"
                    if b["scale"] in (lo, hi) else "")
            print(f"  {meth}: {len(ent)}/20 scales scored | best s{b['scale']:g} "
                  f"PSNR {b['psnr']:.3f} SSIM {b['ssim']:.4f} LPIPS {b['lpips']:.4f}{edge}")
            print("     " + "  ".join(f"{e['scale']:g}:{e['psnr']:.2f}" for e in ent))
        if len(picks) == 2:
            o, s = picks["F_5b"], picks["seva"]
            common = sorted(set(o["rows"]) & set(s["rows"]))
            do = [float(o["rows"][c]["psnr"]) - float(s["rows"][c]["psnr"]) for c in common]
            dl = [float(o["rows"][c]["lpips"]) - float(s["rows"][c]["lpips"]) for c in common]
            wins = sum(1 for x in do if x > 0)
            p = float("nan")
            try:
                from scipy import stats
                p = float(stats.wilcoxon(do).pvalue) if len(do) > 5 else float("nan")
            except Exception:
                pass
            print(f"  => ours - SEVA: PSNR {st.mean(do):+.3f} dB  ({wins}/{len(common)} scenes, "
                  f"p_w={p:.3g}) | LPIPS {st.mean(dl):+.4f}")
            eq = len(data.get((ds, ncf, 'F_5b'), [])) == len(data.get((ds, ncf, 'seva'), []))
            print(f"  => sweep depth ours {len(data.get((ds,ncf,'F_5b'),[]))} vs SEVA "
                  f"{len(data.get((ds,ncf,'seva'),[]))}"
                  + ("" if eq else "  !! UNEQUAL — not like-for-like yet"))
    print()


if __name__ == "__main__":
    main()

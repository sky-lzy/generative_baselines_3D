#!/usr/bin/env python3
"""Deep single-view scale search, reported as ONE primary comparison plus labelled sensitivity runs.

Why the split matters
---------------------
A claim of "we beat SEVA" has to be measured against SEVA as its authors specify it. Tuning the
baseline beyond its own paper produces a stronger-than-published SEVA, which is not the thing anyone
would compare against; conversely, searching OUR side deeper than the baseline turns a tuning
advantage into an apparent modelling one. So exactly one configuration is the headline and everything
else is labelled sensitivity:

  PRIMARY   SEVA at its documented single-view RealEstate10K protocol -- camera_scale swept
            0.1..2.0 in steps of 0.1 (20 points), --cfg 6.0 -- against OUR 20-point grid at a single
            a-priori guidance (hist_guidance 1.0, the engine default, independently validated on 128
            scenes). 20 configs vs 20 configs, one guidance per side.

  SENS-A    identical grids, SEVA at --cfg 2.0 instead. This is still inside the paper: 2.0 is
            demo.py's own default, and our 10-scene grid preferred it over 6.0 by ~0.32 dB. The
            headline result is NOT robust to this choice, which is the single most important caveat
            in the whole comparison.

  SENS-B    both sides unconstrained: every scale run including the sub-0.1 points added beyond
            SEVA's paper, and both cfg values. Included to show the beyond-paper extension changed
            nothing -- SEVA's optimum stayed at 0.1 -- so it never manufactured a margin for us.

The 10 scenes were selected BECAUSE we beat SEVA on them in the 128-scene pass. Everything here is
therefore an upper bound on our advantage, not an estimate of it.

    python3 nvs/report/compare_ss10_protocol.py
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
CELL = re.compile(r"^(?P<m>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<n>\d)"
                  r"__s(?P<s>[\d.]+)__(?P<p>ss10b?)$")
# SEVA's published single-view sweep, verbatim.
PAPER = [round(0.1 * i, 1) for i in range(1, 21)]
# Our 20 canonical points per benchmark (placement justified in submit_scalesearch10.py).
OURS20 = {
    ("re10k128_4dim", "1"): [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                             0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.50],
    ("re10k128_50f", "1"): [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.25, 1.50],
}
BENCH = [("re10k128_4dim", "1", "4DiM split · 1 input view"),
         ("re10k128_50f", "1", "50-frame clip · 1 input view")]


def load(res):
    D = {}
    for f in sorted(glob.glob(f"{res}/*__ss10*.csv")):
        g = CELL.match(os.path.basename(f)[:-4])
        if not g:
            continue
        g = g.groupdict()
        rows, avg = {}, None
        for r in csv.DictReader(open(f)):
            if r["scene"] == "AVERAGE":
                avg = r
            else:
                rows[r["scene"]] = r
        if avg and rows:
            D.setdefault((g["ds"], g["n"], g["m"], g["p"]), {})[float(g["s"])] = (
                float(avg["psnr"]), float(avg["lpips"]), rows)
    return D


def best_of(D, key, allowed):
    d = D.get(key, {})
    c = {s: v for s, v in d.items()
         if allowed is None or any(abs(s - a) < 1e-9 for a in allowed)}
    if not c:
        return None
    s = max(c, key=lambda s: c[s][0])
    return s, c[s], len(c)


def report(o, s, label):
    (os_, ov, on), (ss_, sv, sn) = o, s
    common = sorted(set(ov[2]) & set(sv[2]))
    dp = [float(ov[2][c]["psnr"]) - float(sv[2][c]["psnr"]) for c in common]
    dl = [float(ov[2][c]["lpips"]) - float(sv[2][c]["lpips"]) for c in common]
    p = float("nan")
    try:
        from scipy import stats
        p = float(stats.wilcoxon(dp).pvalue)
    except Exception:
        pass
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {label}")
    print(f"    ours  s{os_:<5g} PSNR {ov[0]:7.3f}  LPIPS {ov[1]:.4f}   ({on} configs)")
    print(f"    SEVA  s{ss_:<5g} PSNR {sv[0]:7.3f}  LPIPS {sv[1]:.4f}   ({sn} configs)"
          + ("   [best at grid edge -> lower bound]" if ss_ in (min(PAPER), max(PAPER)) else ""))
    print(f"    => ours-SEVA {st.mean(dp):+.3f} dB | {sum(1 for x in dp if x > 0)}/{len(common)} "
          f"scenes | p_w={p:.3g} ({sig}) | LPIPS {st.mean(dl):+.4f}")
    if on != sn:
        print(f"    !! config counts differ ({on} vs {sn}) — not yet like-for-like")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(NVS / "results_ss10"))
    a = ap.parse_args()
    D = load(a.results)
    print("=" * 98)
    print("DEEP SINGLE-VIEW SCALE SEARCH — 10 scenes selected BECAUSE we win on them")
    print("Upper bound on our advantage, not an estimate of it.")
    print("=" * 98)
    for ds, n, title in BENCH:
        print(f"\n=== {title} ===")
        ours = best_of(D, (ds, n, "F_5b", "ss10b"), OURS20[(ds, n)])
        if not ours:
            print("  ours: not ready"); continue
        s6 = best_of(D, (ds, n, "seva", "ss10"), PAPER)
        if s6:
            report(ours, s6, "PRIMARY  SEVA at its documented protocol (0.1–2.0 step 0.1, cfg 6.0)")
        s2 = best_of(D, (ds, n, "seva", "ss10b"), PAPER)
        if s2:
            report(ours, s2, "SENS-A   same grids, SEVA at cfg 2.0 (demo.py's default)")
        ou = max((x for x in (best_of(D, (ds, n, "F_5b", p), None) for p in ("ss10", "ss10b")) if x),
                 key=lambda x: x[1][0], default=None)
        su = max((x for x in (best_of(D, (ds, n, "seva", p), None) for p in ("ss10", "ss10b")) if x),
                 key=lambda x: x[1][0], default=None)
        if ou and su:
            report(ou, su, "SENS-B   both sides unconstrained (incl. sub-0.1, both cfgs)")
    print()


if __name__ == "__main__":
    main()

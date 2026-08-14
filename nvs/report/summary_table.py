#!/usr/bin/env python3
"""Print the fair-NVS headline table as text, plus per-scene significance against SEVA.

Same selection rule as the HTML report (each method at its OWN best scale over an equally deep
sweep), so the two can never disagree. Adds what a mean alone cannot support:

  * a PAIRED per-scene comparison against SEVA -- Wilcoxon signed-rank and a paired t-test on the
    same scenes, plus how many of the 128 each method wins. A 0.3 dB mean gap on 128 scenes whose
    per-scene spread is 8 dB is not a result, and only the paired test says so.
  * the sweep DEPTH per side, because a best-of-5 against a best-of-1 is not a comparison.

    python3 nvs/report/summary_table.py
"""
import argparse
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS / "report"))
from build_report import BENCH, PRETTY, collect, best  # noqa: E402


def paired(ours_rows, seva_rows, key="psnr"):
    common = sorted(set(ours_rows) & set(seva_rows))
    if not common:
        return None
    do = [float(ours_rows[c][key]) - float(seva_rows[c][key]) for c in common]
    n = len(do)
    mean = sum(do) / n
    wins = sum(1 for d in do if d > 0)
    p_t = p_w = float("nan")
    try:
        from scipy import stats
        p_t = float(stats.ttest_rel([float(ours_rows[c][key]) for c in common],
                                    [float(seva_rows[c][key]) for c in common]).pvalue)
        if n > 5:
            p_w = float(stats.wilcoxon(do).pvalue)
    except Exception:
        pass
    return dict(n=n, mean=mean, wins=wins, p_t=p_t, p_w=p_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(NVS / "results_fair"))
    ap.add_argument("--metric", default="psnr")
    a = ap.parse_args()
    data = collect(a.results)
    methods = [m for m in ["F_5b", "mot13b_full", "seva"] if any(k[2] == m for k in data)]

    print("=" * 104)
    print("FAIR NVS BENCHMARK — 128 RE10K scenes · identical input pixels · one scorer · lossless "
          "both sides")
    print("=" * 104)
    for ds, ncf, title, desc in BENCH:
        picks = {m: best(data.get((ds, ncf, m), []), a.metric) for m in methods}
        depth = {m: len(data.get((ds, ncf, m), [])) for m in methods}
        print(f"\n{title}\n  {desc}")
        if not any(picks.values()):
            print("  (not yet scored)")
            continue
        print(f"  {'method':<22}{'scale':>7}{'n':>5}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}"
              f"{'sweep':>7}")
        for m in methods:
            e = picks[m]
            if not e:
                print(f"  {PRETTY.get(m, m):<22}{'—':>7}"); continue
            print(f"  {PRETTY.get(m, m):<22}{e['scale']:>7g}{e['n']:>5}{e['psnr']:>9.3f}"
                  f"{e['ssim']:>9.4f}{e['lpips']:>9.4f}{depth[m]:>7}")
        sv = picks.get("seva")
        if sv:
            for m in methods:
                if m == "seva" or not picks[m]:
                    continue
                st = paired(picks[m]["rows"], sv["rows"], "psnr")
                stl = paired(picks[m]["rows"], sv["rows"], "lpips")
                if not st:
                    continue
                print(f"  vs SEVA — {PRETTY.get(m, m)}: PSNR {st['mean']:+.3f} dB "
                      f"({st['wins']}/{st['n']} scenes, p_t={st['p_t']:.3g}, p_w={st['p_w']:.3g})"
                      f" | LPIPS {stl['mean']:+.4f} (lower is better)")
            ok = len(set(depth.values())) == 1
            if not ok:
                print(f"  !! UNEQUAL SWEEP DEPTH {depth} — not a like-for-like comparison")
    print()


if __name__ == "__main__":
    main()

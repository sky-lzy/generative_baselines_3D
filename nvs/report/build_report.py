#!/usr/bin/env python3
"""Assemble the fair-NVS HTML report: headline table + qualitative panels + the scale searches.

Layout, top to bottom:
  1. FAIR COMPARISON TABLE — all 4 benchmark cells x every method, at each method's OWN best scale.
     Best-of-sweep on one side against a default on the other is the failure mode this whole
     benchmark exists to avoid, so the chosen scale is printed in every cell and any cell whose
     sweep depth differs from its row-mates is flagged rather than quietly averaged.
  2. FAIRNESS LEDGER — what is controlled, and the residual that is not.
  3. PER-BENCHMARK PANELS — best / median / worst scenes ranked by OUR PSNR, as GT|ours|SEVA videos.
     Ranking by our score (not the gap) means the "worst" row is genuinely our worst output rather
     than a cherry-picked SEVA failure.
  4. SCALE SEARCHES — every scale rendered for a sample scene, with the pick marked, so the search
     is visible instead of asserted.

    python3 nvs/report/build_report.py
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent

CELL = re.compile(r"^(?P<method>.+?)__(?P<ds>re10k128_\w+?)__ncf(?P<ncf>\d)__s(?P<scale>[\d.]+)$")

BENCH = [
    ("re10k128_50f", 2, "50-frame clip · 2 input views",
     "condition on frames 0 and 49, score the 48 interior frames"),
    ("re10k128_4dim", 1, "4DiM split · 1 input view",
     "condition on frame 0 only, score frames 10,20,…,60"),
    ("re10k128_4dim", 2, "4DiM split · 2 input views",
     "condition on frames 0 and 69, score frames 10,20,…,60"),
    ("re10k128_50f", 1, "50-frame clip · 1 input view",
     "condition on frame 0 only, score frames 1…49"),
]

PRETTY = {"F_5b": "Ours — F (5B)", "mot13b_full": "Ours — MoT (1.3B)",
          "ov1_1p3b": "Ours — MoT overfit (1.3B)", "seva": "SEVA"}

CSS = """
:root{--bg:#ffffff;--fg:#16181d;--mut:#5c6370;--line:#e3e6ec;--card:#f7f8fa;
      --ours:#0b6bcb;--seva:#b4530a;--good:#0d8050;--bad:#c0392b}
:root:not([data-theme=light]){}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#111317;--fg:#e8eaed;--mut:#9aa1ad;--line:#272b33;--card:#181b21;
  --ours:#5aa9f5;--seva:#e2953f;--good:#4cc38a;--bad:#f07a6a}}
:root[data-theme=dark]{--bg:#111317;--fg:#e8eaed;--mut:#9aa1ad;--line:#272b33;--card:#181b21;
  --ours:#5aa9f5;--seva:#e2953f;--good:#4cc38a;--bad:#f07a6a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:32px 20px 80px;
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:44px 0 6px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px}
.sub{color:var(--mut);margin:0 0 20px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:640px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;
         letter-spacing:.5px;border-bottom:2px solid var(--line)}
tbody tr:hover{background:var(--card)}
.win{color:var(--good);font-weight:600}.lose{color:var(--bad)}
.sc{color:var(--mut);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:16px 0}
video{width:100%;border-radius:8px;background:#000;display:block}
.vid{margin:18px 0}
.cap{color:var(--mut);font-size:13px;margin:6px 0 0}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
      background:var(--line);color:var(--mut);margin-left:6px}
.pill.pick{background:var(--good);color:#fff}
ul{margin:8px 0;padding-left:20px}li{margin:4px 0}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def read_csv(p):
    rows, avg = {}, None
    for r in csv.DictReader(open(p)):
        if r["scene"] == "AVERAGE":
            avg = r
        else:
            rows[r["scene"]] = r
    return avg, rows


def collect(results):
    """-> {(ds,ncf,method): [ {scale, psnr, ssim, lpips, n, rows} ... ]}"""
    out = {}
    for f in sorted(Path(results).glob("*.csv")):
        m = CELL.match(f.stem)
        if not m:
            continue
        g = m.groupdict()
        avg, rows = read_csv(f)
        if not avg:
            continue
        out.setdefault((g["ds"], int(g["ncf"]), g["method"]), []).append(dict(
            scale=float(g["scale"]), psnr=float(avg["psnr"]), ssim=float(avg["ssim"]),
            lpips=float(avg["lpips"]), n=len(rows), rows=rows, cell=f.stem))
    return out


def best(entries, metric="psnr"):
    if not entries:
        return None
    key = (lambda e: e[metric]) if metric in ("psnr", "ssim") else (lambda e: -e[metric])
    return max(entries, key=key)


def h(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(NVS / "results_fair"))
    ap.add_argument("--videos", default=str(NVS / "report" / "videos"))
    ap.add_argument("--out", default=str(NVS / "report" / "fair_nvs_report.html"))
    ap.add_argument("--metric", default="psnr")
    a = ap.parse_args()

    data = collect(a.results)
    methods = [m for m in ["F_5b", "mot13b_full", "ov1_1p3b", "seva"]
               if any(k[2] == m for k in data)]
    vdir = Path(a.videos)

    P = []
    P.append(f"<title>Fair NVS Benchmark</title><style>{CSS}</style><div class=wrap>")
    P.append("<h1>Fair NVS Benchmark — ours vs SEVA</h1>")
    P.append("<p class=sub>128 RealEstate10K scenes · identical input pixels · one scorer · "
             "lossless on both sides</p>")

    # ---------------------------------------------------------------- headline table
    P.append("<h2>1 · Fair comparison</h2>")
    P.append("<p class=sub>Every method at its OWN best scale over an equally deep sweep. "
             "The scale that won is printed under each score.</p><div class=scroll><table>")
    P.append("<thead><tr><th>Benchmark</th><th>Metric</th>"
             + "".join(f"<th>{h(PRETTY.get(m, m))}</th>" for m in methods) + "</tr></thead><tbody>")
    for ds, ncf, title, desc in BENCH:
        picks = {m: best(data.get((ds, ncf, m), []), a.metric) for m in methods}
        if not any(picks.values()):
            P.append(f"<tr><td>{h(title)}</td><td colspan={len(methods)+1} class=sc>"
                     f"not yet scored</td></tr>")
            continue
        for mi, met in enumerate(["psnr", "ssim", "lpips"]):
            hi = met != "lpips"
            vals = [picks[m][met] for m in methods if picks[m]]
            bestv = max(vals) if hi else min(vals)
            row = [f"<td>{h(title) if mi == 0 else ''}</td>", f"<td>{met.upper()}</td>"]
            for m in methods:
                e = picks[m]
                if not e:
                    row.append("<td class=sc>—</td>"); continue
                cls = "win" if abs(e[met] - bestv) < 1e-9 else ""
                fmt = f"{e[met]:.3f}" if met == "psnr" else f"{e[met]:.4f}"
                extra = (f"<br><span class=sc>s={e['scale']:g} · n={e['n']}</span>"
                         if mi == 0 else "")
                row.append(f"<td class={cls}>{fmt}{extra}</td>")
            P.append("<tr>" + "".join(row) + "</tr>")
    P.append("</tbody></table></div>")

    # sweep-depth audit — the one thing that can silently unfair the table above
    P.append("<div class=card><b>Sweep-depth audit.</b><ul>")
    for ds, ncf, title, _ in BENCH:
        depths = {m: len(data.get((ds, ncf, m), [])) for m in methods if data.get((ds, ncf, m))}
        if not depths:
            continue
        ok = len(set(depths.values())) == 1
        P.append(f"<li>{h(title)}: " + ", ".join(f"{PRETTY.get(m,m)} {d} scale(s)"
                                                 for m, d in depths.items())
                 + (" — equal depth ✓" if ok else
                    " — <span class=lose>UNEQUAL: best-of-sweep vs shallower search</span>") + "</li>")
    P.append("</ul></div>")

    # ---------------------------------------------------------------- fairness ledger
    P.append("<h2>2 · What makes this fair</h2><div class=card><ul>"
             "<li><b>Same input pixels.</b> One 384×288 4:3 scene store, intrinsics rebased for the "
             "crop+resize. SEVA upsamples it 2× to 576×768 internally and rescales K in the same "
             "call, so it runs at its trained scale on <i>our</i> information.</li>"
             "<li><b>Same camera.</b> Verified numerically through SEVA's own "
             "<code>transform_img_and_K</code>: FOV 74.2478°×59.1682° for every method, principal "
             "point centred, crop a no-op, relative trajectories identical to 0.0e+00.</li>"
             "<li><b>No codec gap.</b> Ours is scored from <code>raw_arrays.npz</code> (pre-H.264), "
             "SEVA from its PNGs. Neither side passes through a lossy codec.</li>"
             "<li><b>One scorer.</b> Both go through the same centre-square-crop → 256² path; "
             "synthetic tests put the two reader paths at max|Δ| = 0.0e+00 on identical content.</li>"
             "<li><b>Both sides tuned.</b> Our history-CFG and SEVA's <code>--cfg</code> were each "
             "chosen on the same 10 held-out scenes, then the scale swept to equal depth.</li>"
             "<li><b>Residual we cannot remove:</b> SEVA renders 576² into the 256² metric domain "
             "while we render 288²; on identical content that resampling is worth ~+0.0 to +0.3 dB "
             "to SEVA. It is measured (T3 in <code>test_score_fair.py</code>), not assumed away.</li>"
             "</ul></div>")

    # ---------------------------------------------------------------- qualitative panels
    P.append("<h2>3 · Qualitative — best / median / worst</h2>")
    P.append("<p class=sub>Scenes ranked by OUR PSNR, so the worst row is our genuine worst case. "
             "Every panel is cropped and resized exactly as the scorer does.</p>")
    for ds, ncf, title, desc in BENCH:
        vids = sorted(vdir.glob(f"cmp__{ds}__ncf{ncf}__*.mp4"))
        P.append(f"<h3>{h(title)}</h3><p class=sub>{h(desc)}</p>")
        if not vids:
            P.append("<p class=sc>no videos rendered yet</p>"); continue
        for v in vids:
            meta = vdir / (v.stem + ".json")
            cap = ""
            if meta.exists():
                d = json.load(open(meta))
                cap = (f"{d.get('rank','')} · scene {d.get('scene','')} · "
                       + " · ".join(f"{k} {v2:.2f}" for k, v2 in d.get("psnr", {}).items()))
            P.append(f"<div class=vid><video src='videos/{h(v.name)}' controls loop muted "
                     f"playsinline preload=metadata></video><p class=cap>{h(cap)}</p></div>")

    # ---------------------------------------------------------------- scale searches
    P.append("<h2>4 · Scale searches</h2>")
    P.append("<p class=sub>Every scale that was run, for both sides, with the pick marked.</p>")
    for ds, ncf, title, _ in BENCH:
        ent = {m: sorted(data.get((ds, ncf, m), []), key=lambda e: e["scale"]) for m in methods}
        if not any(len(v) > 1 for v in ent.values()):
            continue
        P.append(f"<h3>{h(title)}</h3><div class=scroll><table><thead><tr><th>Method</th>"
                 "<th>scale</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>n</th>"
                 "</tr></thead><tbody>")
        for m in methods:
            b = best(ent[m], a.metric)
            for e in ent[m]:
                pick = " <span class='pill pick'>picked</span>" if b and e["cell"] == b["cell"] else ""
                P.append(f"<tr><td>{h(PRETTY.get(m,m))}</td><td>{e['scale']:g}{pick}</td>"
                         f"<td>{e['psnr']:.3f}</td><td>{e['ssim']:.4f}</td>"
                         f"<td>{e['lpips']:.4f}</td><td>{e['n']}</td></tr>")
        P.append("</tbody></table></div>")
        for v in sorted(vdir.glob(f"scales__{ds}__ncf{ncf}__*.mp4")):
            P.append(f"<div class=vid><video src='videos/{h(v.name)}' controls loop muted "
                     f"playsinline preload=metadata></video>"
                     f"<p class=cap>{h(v.stem)}</p></div>")

    P.append("</div>")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(P))
    print(f"wrote {a.out}  ({len(data)} scored cells, {len(list(vdir.glob('*.mp4')))} videos)")


if __name__ == "__main__":
    main()

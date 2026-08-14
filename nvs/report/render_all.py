#!/usr/bin/env python3
"""Pick the scenes worth showing and render every video the report needs.

Selection: rank scenes by OUR PSNR in the winning cell and take the best, the median band and the
worst, spread evenly (default 20 per benchmark). Ranking by our own score rather than by the gap to
SEVA is deliberate — ranking by gap would hand-pick SEVA's failures and turn the qualitative section
into an advert.

Also renders one scale-search strip per swept cell: a single mid-ranked scene at every scale that
was run, so the search can be seen rather than taken on trust.

    python3 nvs/report/render_all.py                    # everything that has results
    python3 nvs/report/render_all.py --per_bench 6      # quick pass
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NVS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NVS / "report"))
from build_report import BENCH, PRETTY, collect, best  # noqa: E402

MAKE = NVS / "report" / "make_videos.py"


def pick_scenes(rows, k):
    """best -> worst by psnr, sampled so the head, the middle band and the tail are all present."""
    order = sorted(rows.items(), key=lambda kv: -float(kv[1]["psnr"]))
    n = len(order)
    if n <= k:
        idx = list(range(n))
    else:
        nb = max(1, k // 3)
        nw = max(1, k // 3)
        nm = k - nb - nw
        mid0 = (n - nm) // 2
        idx = list(range(nb)) + list(range(mid0, mid0 + nm)) + list(range(n - nw, n))
        idx = sorted(set(idx))
    tag = {}
    for i in idx:
        tag[i] = "BEST" if i < max(1, k // 3) else ("WORST" if i >= n - max(1, k // 3) else "MEDIAN")
    return [(order[i][0], tag[i], float(order[i][1]["psnr"]), i + 1, n) for i in idx]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   " + (r.stdout or r.stderr).strip().splitlines()[-1][:160])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(NVS / "results_fair"))
    ap.add_argument("--videos", default=str(NVS / "report" / "videos"))
    ap.add_argument("--scenes_fair",
                    default="/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair")
    ap.add_argument("--per_bench", type=int, default=20)
    ap.add_argument("--scale_strips", type=int, default=1, help="scenes per swept cell")
    ap.add_argument("--primary", default="F_5b", help="the 'ours' model that drives ranking")
    ap.add_argument("--only_bench", default=None,
                    help="'<dataset>:<ncf>' — render just this benchmark, so the four can be "
                         "rendered as four concurrent processes instead of one serial pass "
                         "(~80 videos at ~30s each is 40 minutes serially)")
    a = ap.parse_args()

    data = collect(a.results)
    vdir = Path(a.videos); vdir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["F_5b", "mot13b_full", "seva"] if any(k[2] == m for k in data)]

    bench = BENCH
    if a.only_bench:
        want_ds, want_ncf = a.only_bench.split(":")
        bench = [b for b in BENCH if b[0] == want_ds and b[1] == int(want_ncf)]
        if not bench:
            sys.exit(f"ERROR: no benchmark matches {a.only_bench}")
    for ds, ncf, title, _ in bench:
        prim = best(data.get((ds, ncf, a.primary), []))
        if not prim:
            print(f"[skip] {title}: no {a.primary} results"); continue
        cells = []
        for m in methods:
            b = best(data.get((ds, ncf, m), []))
            if b:
                cells.append(f"{PRETTY.get(m, m)}={b['cell']}")
        picks = pick_scenes(prim["rows"], a.per_bench)
        print(f"\n{title}: {len(picks)} scenes x {len(cells)} panels")
        for scene, rank, psnr, pos, n in picks:
            name = f"cmp__{ds}__ncf{ncf}__{rank}__{scene}"
            ok = run([sys.executable, str(MAKE), "--dataset", ds, "--ncf", str(ncf),
                      "--scene", scene, "--cells", ",".join(cells), "--name", name,
                      "--out", str(vdir), "--scenes_fair", a.scenes_fair])
            if ok:
                meta = {"rank": f"{rank} (#{pos} of {n} by our PSNR)", "scene": scene,
                        "psnr": {}}
                for m in methods:
                    b = best(data.get((ds, ncf, m), []))
                    if b and scene in b["rows"]:
                        meta["psnr"][PRETTY.get(m, m)] = float(b["rows"][scene]["psnr"])
                (vdir / (name + ".json")).write_text(json.dumps(meta))
            print(f"  {'ok ' if ok else 'FAIL'} {rank:<6} {scene}  ourPSNR {psnr:.2f}")

        # ---- scale strips ---------------------------------------------------------------
        for m in methods:
            ent = sorted(data.get((ds, ncf, m), []), key=lambda e: e["scale"])
            if len(ent) < 2:
                continue
            b = best(ent)
            mids = [p[0] for p in picks if p[1] == "MEDIAN"][:a.scale_strips] or [picks[0][0]]
            for scene in mids:
                cols = ",".join(f"s={e['scale']:g}{' PICK' if e['cell']==b['cell'] else ''}={e['cell']}"
                                for e in ent)
                name = f"scales__{ds}__ncf{ncf}__{m}__{scene}"
                ok = run([sys.executable, str(MAKE), "--dataset", ds, "--ncf", str(ncf),
                          "--scene", scene, "--cells", cols, "--name", name,
                          "--out", str(vdir), "--scenes_fair", a.scenes_fair])
                print(f"  {'ok ' if ok else 'FAIL'} scales {m} {scene} ({len(ent)} scales)")


if __name__ == "__main__":
    main()

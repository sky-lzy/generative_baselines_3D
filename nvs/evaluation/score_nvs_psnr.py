#!/usr/bin/env python3
"""NVS scorer — per-scene PSNR CSV for one (method, dataset, conditioning) cell.

NO pixels are recomputed: the inference engine already computed each scene's PSNR on the raw
tensors (interior frames only — frames 0 / T-1 are conditioning) BEFORE mp4 compression and
saved it to <cell>/sample_XXXXX/rgb_metrics.json. This scorer maps sample index -> scene name
(sorted scene dirs, exactly datasets/svc_scenes.py::_load_records order), aggregates, and
writes <out_dir>/<cell>.csv with per-scene rows + an AVERAGE row.

Exactness: mean(per-scene psnr) == final_stats.json "psnr" to float64 (verified across the
campaign; the engine's final_stats IS the running mean of the same per-scene values). When
final_stats.json is complete the scorer cross-checks and FAILS on any mismatch > 1e-6 dB.
A cell missing any scene's rgb_metrics.json writes NO csv and exits 1 (count-aware: partial
cells never produce a result that looks complete).
"""
import argparse
import json
import sys
from pathlib import Path


def scene_list(data_root: str, min_frames: int = 26):
    """Replicates datasets/svc_scenes.py::_load_records ordering/filter."""
    scenes = []
    for sd in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        tj = sd / "transforms.json"
        if not tj.exists():
            continue
        if len(json.load(open(tj))["frames"]) >= min_frames:
            scenes.append(sd.name)
    return scenes


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-scene NVS PSNR into a CSV.")
    ap.add_argument("--run_dir", required=True, help="preds cell dir <tag>__<ds>__ncf<k>")
    ap.add_argument("--data_root", required=True, help="<scenes_root>/<ds> (scene-name mapping)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    scenes = scene_list(args.data_root)
    if not scenes:
        sys.exit(f"ERROR: no scenes under {args.data_root}")

    rows, missing = [], []
    for i, scene in enumerate(scenes):
        mj = run / f"sample_{i:05d}" / "rgb_metrics.json"
        if not mj.exists():
            missing.append(scene)
            continue
        rows.append((scene, float(json.load(open(mj))["psnr"])))
    if missing:
        sys.exit(f"ERROR: {run.name}: {len(missing)}/{len(scenes)} scenes missing "
                 f"rgb_metrics.json ({', '.join(missing[:5])}...) — cell incomplete, no CSV written")

    avg = sum(p for _, p in rows) / len(rows)

    # cross-check against the engine's own aggregate when the cell completed in one pass
    fs = run / "final_stats.json"
    if fs.exists():
        stats = json.load(open(fs))
        if stats.get("count", 0) == len(scenes):
            diff = abs(stats["psnr"] - avg)
            if diff > 1e-6:
                sys.exit(f"ERROR: {run.name}: scorer avg {avg:.6f} != final_stats "
                         f"{stats['psnr']:.6f} (diff {diff:.2e}) — refusing to write CSV")
            print(f"[nvs-score] cross-check OK vs final_stats.json (diff {diff:.2e} dB)")
        else:
            print(f"[nvs-score] final_stats covers {stats.get('count', 0)}/{len(scenes)} scenes "
                  f"(--resume run); CSV built from per-scene rgb_metrics.json")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("scene,psnr\n")
        for scene, p in rows:
            f.write(f"{scene},{p:.6f}\n")
        f.write(f"AVERAGE,{avg:.6f}\n")
    print(f"[nvs-score] {run.name}: AVERAGE {avg:.4f} dB over {len(rows)} scenes -> {out}")


if __name__ == "__main__":
    main()

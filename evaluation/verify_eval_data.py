#!/usr/bin/env python3
"""Pin the benchmark inputs byte-for-byte.

`--write` walks every image/GT file the pi3 depth/pose benchmark reads (the exact seq lists and
directory templates the campaign used) and records relpath, size and md5 into
evaluation/eval_data_manifest.json. Without flags it re-walks and FAILS on any missing, extra,
resized or bit-changed file, so a run can prove it is scoring the identical clips.

Frame selection is a pure function of the sorted file list per sequence (pi3seq.py), so
"same files byte-for-byte" => same frames, same GT, same numbers.
"""
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

PI3 = os.environ.get("PI3_ROOT",
    "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose")
D = f"{PI3}/data"

# The EXACT benchmark definition: seq lists verbatim from the VWM pi3seq_*.yaml configs the
# campaign ran with, plus the GT templates from eval_ours_pi3.DATASET_SPEC. Duplicated here on
# purpose: this file makes the eval repo self-describing instead of trusting a VWM checkout.
SEQS = {
    "sintel": ["alley_2","ambush_4","ambush_5","ambush_6","cave_2","cave_4","market_2",
               "market_5","market_6","shaman_3","sleeping_1","sleeping_2","temple_2","temple_3"],
    "bonn": ["rgbd_bonn_balloon2","rgbd_bonn_crowd2","rgbd_bonn_crowd3",
             "rgbd_bonn_person_tracking2","rgbd_bonn_synchronous"],
    "kitti": None,   # all dirs under the gathered root (13)
    "tum": None,     # all dirs with groundtruth_90.txt (8)
    "scannetv2": None,  # all dirs with pose_90.txt (100)
}
ROOTS = {
    "sintel":    [f"{D}/sintel/training/final/{{s}}", f"{D}/sintel/training/camdata_left/{{s}}",
                  f"{D}/sintel/training/depth/{{s}}"],
    "bonn":      [f"{D}/bonn/rgbd_bonn_dataset/{{s}}/rgb_110",
                  f"{D}/bonn/rgbd_bonn_dataset/{{s}}/depth_110"],
    "kitti":     [f"{D}/kitti/depth_selection/val_selection_cropped/image_gathered/{{s}}",
                  f"{D}/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/{{s}}"],
    "tum":       [f"{D}/tum/{{s}}/rgb_90", f"{D}/tum/{{s}}/groundtruth_90.txt"],
    "scannetv2": [f"{D}/scannetv2/{{s}}/color_90", f"{D}/scannetv2/{{s}}/pose_90.txt"],
}
DISCOVER = {
    "kitti": f"{D}/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/*",
    "tum": f"{D}/tum/*/groundtruth_90.txt",
    "scannetv2": f"{D}/scannetv2/*/pose_90.txt",
}

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def walk():
    out = {}
    for ds, seqs in SEQS.items():
        if seqs is None:
            seqs = sorted({Path(x).parent.name if x.endswith(".txt") else Path(x).name
                           for x in glob.glob(DISCOVER[ds])})
        for s in seqs:
            for tpl in ROOTS[ds]:
                root = tpl.format(s=s)
                files = [root] if os.path.isfile(root) else \
                        sorted(glob.glob(os.path.join(root, "*")))
                for f in files:
                    if os.path.isfile(f):
                        rel = os.path.relpath(f, D)
                        out[rel] = {"size": os.path.getsize(f), "md5": md5(f)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--manifest", default=str(Path(__file__).parent / "eval_data_manifest.json"))
    a = ap.parse_args()
    cur = walk()
    if a.write:
        json.dump(cur, open(a.manifest, "w"), indent=0, sort_keys=True)
        print(f"wrote {a.manifest}: {len(cur)} files")
        return
    ref = json.load(open(a.manifest))
    missing = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
    changed = sorted(k for k in set(ref) & set(cur) if ref[k] != cur[k])
    for label, lst in [("MISSING", missing), ("EXTRA", extra), ("CHANGED", changed)]:
        for k in lst[:10]:
            print(f"{label}: {k}")
    if missing or extra or changed:
        sys.exit(f"FAIL: {len(missing)} missing, {len(extra)} extra, {len(changed)} changed "
                 f"of {len(ref)} manifest files")
    print(f"OK: all {len(ref)} benchmark files present and byte-identical")

if __name__ == "__main__":
    main()

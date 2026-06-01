#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_raw_vs_oracle_tables import _format_table, _summarize_method


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TUM RGB-D raw/oracle benchmark tables.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "outputs" / "datasets" / "tum_rgbd"
    specs = [
        ("Pi3", "TUM RGB-D 50-view", root / "pi3_50view_s50", "stack"),
        ("Ours/VWM", "TUM RGB-D 50-view", root / "ours_50view_s50_s40", "stack"),
        ("Geo4D", "TUM RGB-D 50-view", root / "geo4d_50view_s50_s5", "stack"),
        ("Pi3", "TUM RGB-D 2-view long", root / "pi3_2view_long2_perseq", "pair"),
        ("Ours/VWM", "TUM RGB-D 2-view long", root / "ours_2view_long2_perseq_s40", "pair"),
        ("Geo4D", "TUM RGB-D 2-view long", root / "geo4d_2view_long2_perseq_s5", "pair"),
    ]
    rows = [_summarize_method(*spec) for spec in specs]
    report = {
        "note": "TUM RGB-D rows use 50 raw RGB-D trajectory windows with 50 frames each. 50-view rows evaluate all C(50,2)=1225 relative-pose pairs per sequence. 2-view rows use 2 deterministic long-baseline pairs per sequence; Geo4D has 99 valid pairs because one pair hit an OpenCV PnP degeneracy. Oracle rows use a GT-assisted per-sequence SO3 camera-basis fit from predicted relative translation directions to GT directions.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = "\n".join(
        [
            "# TUM RGB-D Raw vs Oracle Tables",
            "",
            str(report["note"]),
            "",
            "## Raw Pi3 Metrics",
            "",
            _format_table(rows, "raw"),
            "",
            "## Oracle Per-Sequence Camera-Basis SO3",
            "",
            _format_table(rows, "oracle_per_sequence_camera_basis_so3"),
            "",
        ]
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_raw_vs_oracle_tables import _format_table, _summarize_method


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OOD raw/oracle benchmark tables.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "outputs" / "datasets"
    specs = [
        ("Pi3", "DL3DV-Eval 50-view", root / "dl3dv_eval" / "pi3_50view_s50", "stack"),
        ("Ours/VWM", "DL3DV-Eval 50-view", root / "dl3dv_eval" / "ours_50view_s50_s40", "stack"),
        ("Geo4D", "DL3DV-Eval 50-view", root / "dl3dv_eval" / "geo4d_50view_s50_s5", "stack"),
        ("Pi3", "7Scenes 50-view", root / "7scenes" / "pi3_50view_s50", "stack"),
        ("Ours/VWM", "7Scenes 50-view", root / "7scenes" / "ours_50view_s50_s40", "stack"),
        ("Geo4D", "7Scenes 50-view", root / "7scenes" / "geo4d_50view_s50_s5", "stack"),
        ("Pi3", "DL3DV-Eval 2-view long", root / "dl3dv_eval" / "pi3_2view_long2_perseq", "pair"),
        ("Ours/VWM", "DL3DV-Eval 2-view long", root / "dl3dv_eval" / "ours_2view_long2_perseq_s40", "pair"),
        ("Geo4D", "DL3DV-Eval 2-view long", root / "dl3dv_eval" / "geo4d_2view_long2_perseq_s5", "pair"),
        ("Pi3", "7Scenes 2-view long", root / "7scenes" / "pi3_2view_long2_perseq", "pair"),
        ("Ours/VWM", "7Scenes 2-view long", root / "7scenes" / "ours_2view_long2_perseq_s40", "pair"),
        ("Geo4D", "7Scenes 2-view long", root / "7scenes" / "geo4d_2view_long2_perseq_s5", "pair"),
    ]
    rows = [_summarize_method(*spec) for spec in specs]
    report = {
        "note": "OOD rows include 50-sequence 50-view runs with all C(50,2)=1225 relative-pose pairs per sequence, plus 50-sequence 2-view long-baseline runs with 100 sampled pairs per dataset/method. Oracle rows use a GT-assisted per-sequence SO3 camera-basis fit from predicted relative translation directions to GT directions.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = "\n".join(
        [
            "# OOD Raw vs Oracle Tables",
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

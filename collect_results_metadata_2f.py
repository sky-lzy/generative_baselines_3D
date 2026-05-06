#!/usr/bin/env python3
"""Collect metadata two-frame Geo4D/RayDiffusion metrics into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materialize_metadata_2f import DEFAULT_DATASETS


POSE_METHODS = {
    "geo4d_pose": "Geo4D",
    "raydiffusion_pose": "RayDiffusion",
}
DEPTH_METHODS = {
    "geo4d_depth": "Geo4D",
}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _collect_pose(path: Path) -> str:
    data = _read_json(path / "final_stats.json")
    if not data:
        return ""
    if all(k in data for k in ("ate", "rpe_trans", "rpe_rot")):
        return f"{data['ate']:.3f} / {data['rpe_trans']:.3f} / {data['rpe_rot']:.2f} (n={data.get('n_samples', 0)})"
    return ""


def _collect_depth(path: Path) -> str:
    data = _read_json(path / "final_stats.json")
    if not data:
        return ""
    if all(k in data for k in ("abs_rel", "rmse", "delta_1")):
        return f"{data['abs_rel']:.3f} / {data['rmse']:.3f} / {data['delta_1']:.3f} (n={data.get('n_samples', 0)})"
    return ""


def _table(title: str, methods: dict[str, str], datasets: list[str], root: Path, collect_fn) -> list[str]:
    lines = [title, ""]
    header = "| Method | " + " | ".join(datasets) + " |"
    sep = "|---|" + "|".join("---" for _ in datasets) + "|"
    lines.extend([header, sep])
    for subdir, name in methods.items():
        row = [name]
        for dataset in datasets:
            row.append(collect_fn(root / dataset / subdir))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/metadata_2f")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    datasets = args.dataset or DEFAULT_DATASETS
    output = Path(args.output) if args.output else root / "RESULTS_METADATA_2F.md"

    lines = [
        "# Metadata 2-Frame Results",
        "",
        "Inputs are first and last frames from the metadata-defined sequences.",
        "Pose: ATE / RPE_trans / RPE_rot. Depth: Abs Rel / RMSE / delta<1.25.",
        "",
    ]
    lines.extend(_table("## Camera Pose", POSE_METHODS, datasets, root, _collect_pose))
    lines.extend(_table("## Depth", DEPTH_METHODS, datasets, root, _collect_depth))
    lines.append("Empty cell = not run, failed, or unavailable.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

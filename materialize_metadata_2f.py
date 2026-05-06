#!/usr/bin/env python3
"""Create first/last-frame baseline inputs from metadata JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metadata_v3_dataset import load_metadata_records, materialize_two_frame_record


DEFAULT_DATASETS = [
    "aria",
    "dl3dv",
    "dl3dv_test",
    "re10k",
    "scannetpp",
    "scenenet_depth",
    "spatialvid_nvs",
    "tanksandtemples",
    "vkitti2",
    "7scenes",
    "tum",
    "bonn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path",
        action="append",
        default=None,
        help="Metadata JSON path. Can be repeated.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to materialize. Can be repeated. Defaults to all selected datasets.",
    )
    parser.add_argument("--results_root", default="results/metadata_2f")
    parser.add_argument("--input_subdir", default="inputs_pose_depth_2f")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    metadata_paths = args.metadata_path or ["metadata_v3_n50.json", "metadata_zeroshot.json"]
    datasets = args.dataset or DEFAULT_DATASETS
    results_root = Path(args.results_root)

    records = load_metadata_records(
        metadata_paths=metadata_paths,
        dataset_names=datasets,
        exclude_datasets={"agibot_world"},
        max_samples=args.max_samples,
    )
    if not records:
        raise SystemExit(
            "No metadata records found for "
            f"datasets={datasets} metadata_paths={metadata_paths}"
        )

    manifests: dict[str, list[dict]] = {}
    for record in records:
        input_dir = results_root / record.dataset / args.input_subdir
        sample_out = materialize_two_frame_record(record, input_dir, fps=args.fps)
        frame_info = json.loads((sample_out / "frame_indices.json").read_text(encoding="utf-8"))
        manifests.setdefault(record.dataset, []).append(
            {
                "sample_id": record.sample_id,
                "sample_dir": str(sample_out),
                "metadata_path": str(record.metadata_path),
                "frame_indices": frame_info["frame_indices"],
                "source_frame_count": frame_info["source_frame_count"],
            }
        )

    for dataset, rows in manifests.items():
        input_dir = results_root / dataset / args.input_subdir
        (input_dir / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"[{dataset}] materialized {len(rows)} samples -> {input_dir}")


if __name__ == "__main__":
    main()

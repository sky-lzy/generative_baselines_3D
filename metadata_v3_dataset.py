#!/usr/bin/env python3
"""Metadata-backed dataset for test-time-search v3.

The paper V3 split is defined by
``generative_baselines_3D/metadata_v3_n50.json``.  This dataset loads those
canonical per-sample artifacts directly instead of rediscovering examples from
the original dataset roots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import imageio
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TaskKind = Literal["nvs", "pose_depth"]


@dataclass(frozen=True)
class MetadataSample:
    dataset: str
    sample_id: str
    prompt: str
    block: dict[str, Any]


@dataclass(frozen=True)
class MetadataRecord:
    metadata_path: Path
    dataset: str
    sample_id: str
    prompt: str
    block: dict[str, Any]
    raw_record: dict[str, Any]


def _read_video(path: str | Path, n_frames: int | None = None) -> torch.Tensor:
    arr = iio.imread(path)
    if arr.ndim != 4:
        raise ValueError(f"Expected THWC video from {path}, got {arr.shape}")
    if n_frames is not None:
        arr = _fit_frames_np(arr, n_frames)
    arr = arr.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _fit_frames_np(arr: np.ndarray, n_frames: int) -> np.ndarray:
    if arr.shape[0] == n_frames:
        return arr
    if arr.shape[0] > n_frames:
        return arr[:n_frames]
    pad = np.repeat(arr[-1:], n_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _prompt_from_record(record: dict[str, Any], block: dict[str, Any]) -> str:
    for source in (record, block):
        prompt = source.get("prompt")
        if isinstance(prompt, str):
            return prompt
    for source in (record, block):
        prompt_txt = source.get("prompt_txt")
        if isinstance(prompt_txt, str) and prompt_txt:
            path = Path(prompt_txt)
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
            return prompt_txt
    return ""


def _block_from_record(record: dict[str, Any], task: TaskKind) -> dict[str, Any] | None:
    block_key = "nvs" if task == "nvs" else "pose_depth"
    block = record.get(block_key)
    if not block and "gt_rgb_mp4" in record:
        block = record
    if not block:
        return None
    return dict(block)


def load_metadata_records(
    metadata_paths: Sequence[str | Path],
    dataset_names: Sequence[str] | None = None,
    *,
    task: TaskKind = "pose_depth",
    exclude_datasets: Iterable[str] = (),
    max_samples: int | None = None,
) -> list[MetadataRecord]:
    """Load canonical metadata records from nested v3 or flat zeroshot JSON."""
    wanted = list(dataset_names) if dataset_names is not None else None
    excluded = set(exclude_datasets)
    records: list[MetadataRecord] = []

    for metadata_path_like in metadata_paths:
        metadata_path = Path(metadata_path_like)
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        datasets = meta.get("datasets", {})
        names = wanted if wanted is not None else list(datasets.keys())

        for dataset_name in names:
            if dataset_name in excluded or dataset_name not in datasets:
                continue
            count_for_dataset = 0
            for record in datasets[dataset_name].get("samples", []):
                block = _block_from_record(record, task)
                if not block:
                    continue
                records.append(
                    MetadataRecord(
                        metadata_path=metadata_path,
                        dataset=dataset_name,
                        sample_id=str(record["sample_id"]),
                        prompt=_prompt_from_record(record, block),
                        block=block,
                        raw_record=record,
                    )
                )
                count_for_dataset += 1
                if max_samples is not None and count_for_dataset >= int(max_samples):
                    break
    return records


def _first_last_indices(n_frames: int) -> list[int]:
    if n_frames <= 0:
        raise ValueError("Cannot materialize a two-frame sample from an empty sequence")
    if n_frames == 1:
        return [0, 0]
    return [0, n_frames - 1]


def _temporal_length_from_npz(path: str | Path | None, preferred_key: str) -> int | None:
    if not path:
        return None
    data = np.load(path)
    if preferred_key in data and data[preferred_key].ndim >= 1:
        return int(data[preferred_key].shape[0])
    return None


def _slice_npz_file(
    source_path: str | Path,
    output_path: str | Path,
    frame_indices: Sequence[int],
    temporal_lengths: Iterable[int],
) -> None:
    source = np.load(source_path)
    temporal_lengths = set(int(v) for v in temporal_lengths if v is not None)
    max_index = max(frame_indices)
    out: dict[str, np.ndarray] = {}
    temporal_1d_keys = {"timestamps", "frame_indices"}
    for key in source.files:
        arr = np.asarray(source[key])
        should_slice = (
            arr.ndim >= 3 and arr.shape[0] in temporal_lengths and arr.shape[0] > max_index
        ) or (
            key in temporal_1d_keys
            and arr.ndim >= 1
            and arr.shape[0] in temporal_lengths
            and arr.shape[0] > max_index
        )
        out[key] = arr[list(frame_indices)] if should_slice else arr
    np.savez_compressed(output_path, **out)


def materialize_two_frame_record(
    record: MetadataRecord,
    output_dir: str | Path,
    *,
    fps: int = 10,
) -> Path:
    """Write one first/last-frame sample dir consumable by the baseline evaluators."""
    output_dir = Path(output_dir)
    sample_out = output_dir / record.sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    rgb_path = record.block.get("gt_rgb_mp4")
    camera_path = record.block.get("gt_cameras_npz")
    depth_path = record.block.get("gt_depth_raw_npz")
    if not rgb_path:
        raise KeyError(f"{record.dataset}/{record.sample_id} missing gt_rgb_mp4")
    if not camera_path:
        raise KeyError(f"{record.dataset}/{record.sample_id} missing gt_cameras_npz")
    if not depth_path:
        raise KeyError(f"{record.dataset}/{record.sample_id} missing gt_depth_raw_npz")

    rgb = iio.imread(rgb_path)
    if rgb.ndim != 4:
        raise ValueError(f"Expected THWC video from {rgb_path}, got {rgb.shape}")

    video_len = int(rgb.shape[0])
    camera_len = _temporal_length_from_npz(camera_path, "extrinsics")
    depth_len = _temporal_length_from_npz(depth_path, "depth")
    source_len = min(v for v in (video_len, camera_len, depth_len) if v is not None)
    frame_indices = _first_last_indices(source_len)

    imageio.mimwrite(
        sample_out / "gt_rgb.mp4",
        list(rgb[frame_indices]),
        fps=fps,
        macro_block_size=1,
    )
    temporal_lengths = [video_len, camera_len, depth_len]
    _slice_npz_file(camera_path, sample_out / "gt_cameras.npz", frame_indices, temporal_lengths)
    _slice_npz_file(depth_path, sample_out / "gt_depth_raw.npz", frame_indices, temporal_lengths)

    (sample_out / "prompt.txt").write_text(record.prompt, encoding="utf-8")
    (sample_out / "frame_indices.json").write_text(
        json.dumps(
            {
                "frame_indices": frame_indices,
                "source_frame_count": source_len,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (sample_out / "source_metadata.json").write_text(
        json.dumps(
            {
                "metadata_path": str(record.metadata_path),
                "dataset": record.dataset,
                "sample_id": record.sample_id,
                "prompt": record.prompt,
                "source_paths": record.block,
                "frame_indices": frame_indices,
                "source_frame_count": source_len,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return sample_out


def _fit_frames_torch(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    if x.shape[0] == n_frames:
        return x
    if x.shape[0] > n_frames:
        return x[:n_frames]
    pad = x[-1:].repeat((n_frames - x.shape[0],) + (1,) * (x.ndim - 1))
    return torch.cat([x, pad], dim=0)


def _resize_video(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.shape[-2:] == (height, width):
        return x
    return F.interpolate(x.float(), size=(height, width), mode="bilinear", align_corners=False)


def _read_depth(block: dict[str, Any], n_frames: int, height: int, width: int) -> tuple[torch.Tensor, float]:
    path = block.get("gt_depth_raw_npz")
    if not path:
        return torch.zeros((n_frames, 1, height, width), dtype=torch.float32), 1.0
    z = np.load(path)
    depth = torch.from_numpy(z["depth"].astype(np.float32))
    scale = float(z["scale"]) if "scale" in z else 1.0
    depth = _fit_frames_torch(depth, n_frames)
    if depth.ndim == 3:
        depth = depth[:, None]
    elif depth.ndim != 4:
        raise ValueError(f"Unexpected depth shape {tuple(depth.shape)} in {path}")
    depth = _resize_video(depth, height, width)
    return depth.contiguous(), scale


def _read_optional_ray_video(
    path: str | Path | None,
    n_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if not path:
        return torch.zeros((n_frames, 3, height, width), dtype=torch.float32)
    return _resize_video(_read_video(path, n_frames), height, width)


class MetadataV3Dataset(Dataset):
    """Loads canonical V3 samples for NVS or RGB-to-geometry prediction."""

    def __init__(
        self,
        metadata_path: str | Path,
        dataset_name: str,
        task: TaskKind,
        n_frames: int,
        height: int,
        width: int,
        raymap_height: int,
        raymap_width: int,
        max_samples: int | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.dataset_name = dataset_name
        self.task = task
        self.n_frames = int(n_frames)
        self.height = int(height)
        self.width = int(width)
        self.raymap_height = int(raymap_height)
        self.raymap_width = int(raymap_width)
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if dataset_name not in meta["datasets"]:
            raise KeyError(f"{dataset_name!r} not in {self.metadata_path}")
        samples: list[MetadataSample] = []
        block_key = "nvs" if task == "nvs" else "pose_depth"
        for record in meta["datasets"][dataset_name]["samples"]:
            block = record.get(block_key)
            if not block and "gt_rgb_mp4" in record:
                block = record
            if not block:
                continue
            samples.append(
                MetadataSample(
                    dataset=dataset_name,
                    sample_id=str(record["sample_id"]),
                    prompt=str(record.get("prompt") or record.get("prompt_txt") or ""),
                    block=block,
                )
            )
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        block = sample.block
        rgb = _resize_video(
            _read_video(block["gt_rgb_mp4"], self.n_frames),
            self.height,
            self.width,
        )
        ray_d = _read_optional_ray_video(
            block.get("gt_ray_d_mp4") or block.get("gt_rays_d_mp4"),
            self.n_frames,
            self.raymap_height,
            self.raymap_width,
        )
        ray_m = _read_optional_ray_video(
            block.get("gt_ray_m_mp4") or block.get("gt_rays_m_mp4"),
            self.n_frames,
            self.raymap_height,
            self.raymap_width,
        )
        depth, scale = _read_depth(block, self.n_frames, self.height, self.width)
        return {
            "videos": rgb.float(),
            "raymaps": torch.cat([ray_d.float(), ray_m.float()], dim=1),
            "depths": depth.float(),
            "raymap_scale": torch.tensor(scale, dtype=torch.float32),
            "prompts": sample.prompt,
            "sample_ids": sample.sample_id,
            "metadata_dataset": sample.dataset,
            "has_raymap": torch.tensor(bool(block.get("gt_ray_d_mp4") or block.get("gt_rays_d_mp4"))),
            "has_depth": torch.tensor(bool(block.get("gt_depth_raw_npz"))),
        }

import json
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metadata_v3_dataset import load_metadata_records, materialize_two_frame_record


def _write_video(path: Path, n_frames: int = 4) -> np.ndarray:
    frames = np.zeros((n_frames, 16, 16, 3), dtype=np.uint8)
    for i in range(n_frames):
        frames[i, :, :, :] = i * 40
    imageio.mimwrite(path, list(frames), fps=4, macro_block_size=1)
    return frames


def _write_camera_npz(path: Path, n_frames: int = 4) -> np.ndarray:
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], n_frames, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], n_frames, axis=0)
    for i in range(n_frames):
        extrinsics[i, 0, 3] = float(i)
        intrinsics[i, 0, 0] = float(i + 1)
    np.savez(path, extrinsics=extrinsics, intrinsics=intrinsics, scalar=np.array(7.0))
    return extrinsics


def _write_depth_npz(path: Path, n_frames: int = 4) -> np.ndarray:
    depth = np.arange(n_frames * 4 * 4, dtype=np.float32).reshape(n_frames, 4, 4)
    np.savez(path, depth=depth, scale=np.array(2.5, dtype=np.float32))
    return depth


def test_load_metadata_records_supports_nested_and_flat_records(tmp_path):
    nested_video = tmp_path / "nested.mp4"
    flat_video = tmp_path / "flat.mp4"
    _write_video(nested_video)
    _write_video(flat_video)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "nested": {
                        "samples": [
                            {
                                "sample_id": "sample_00000",
                                "prompt": "nested prompt",
                                "pose_depth": {"gt_rgb_mp4": str(nested_video)},
                            }
                        ]
                    },
                    "flat": {
                        "samples": [
                            {
                                "sample_id": "sample_00001",
                                "prompt_txt": None,
                                "gt_rgb_mp4": str(flat_video),
                            }
                        ]
                    },
                    "agibot_world": {
                        "samples": [
                            {
                                "sample_id": "sample_99999",
                                "pose_depth": {"gt_rgb_mp4": str(nested_video)},
                            }
                        ]
                    },
                }
            }
        )
    )

    records = load_metadata_records(
        metadata_paths=[metadata_path],
        dataset_names=["nested", "flat", "agibot_world"],
        exclude_datasets={"agibot_world"},
    )

    assert [(r.dataset, r.sample_id) for r in records] == [
        ("nested", "sample_00000"),
        ("flat", "sample_00001"),
    ]
    assert records[0].prompt == "nested prompt"
    assert records[0].block["gt_rgb_mp4"] == str(nested_video)
    assert records[1].block["gt_rgb_mp4"] == str(flat_video)


def test_materialize_two_frame_record_slices_rgb_cameras_and_depth(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    video_path = source_dir / "gt_rgb.mp4"
    camera_path = source_dir / "gt_cameras.npz"
    depth_path = source_dir / "gt_depth_raw.npz"
    _write_video(video_path, n_frames=4)
    extrinsics = _write_camera_npz(camera_path, n_frames=4)
    depth = _write_depth_npz(depth_path, n_frames=4)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "demo": {
                        "samples": [
                            {
                                "sample_id": "sample_00000",
                                "prompt": "demo prompt",
                                "pose_depth": {
                                    "gt_rgb_mp4": str(video_path),
                                    "gt_cameras_npz": str(camera_path),
                                    "gt_depth_raw_npz": str(depth_path),
                                },
                            }
                        ]
                    }
                }
            }
        )
    )
    record = load_metadata_records([metadata_path], ["demo"])[0]

    sample_out = materialize_two_frame_record(record, tmp_path / "out")

    assert sample_out.name == "sample_00000"
    written_frames = np.stack(imageio.mimread(sample_out / "gt_rgb.mp4", memtest=False))
    assert written_frames.shape[0] == 2

    camera_data = np.load(sample_out / "gt_cameras.npz")
    np.testing.assert_allclose(camera_data["extrinsics"], extrinsics[[0, 3]])
    np.testing.assert_allclose(camera_data["intrinsics"][0, 0, 0], 1.0)
    np.testing.assert_allclose(camera_data["intrinsics"][1, 0, 0], 4.0)
    assert float(camera_data["scalar"]) == 7.0

    depth_data = np.load(sample_out / "gt_depth_raw.npz")
    np.testing.assert_allclose(depth_data["depth"], depth[[0, 3]])
    assert float(depth_data["scale"]) == 2.5

    assert json.loads((sample_out / "frame_indices.json").read_text())["frame_indices"] == [0, 3]
    assert (sample_out / "prompt.txt").read_text() == "demo prompt"

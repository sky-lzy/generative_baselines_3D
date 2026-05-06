from pathlib import Path
import importlib.util

import numpy as np


def _load_camera_utils():
    module_path = Path(__file__).resolve().parents[1] / "seva" / "camera_utils.py"
    spec = importlib.util.spec_from_file_location("seva_camera_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_auto_rescales_ray_grid_intrinsics_to_rgb_pixels():
    camera_utils = _load_camera_utils()
    K = np.array(
        [
            [[24.629694, 0.0, 20.0], [0.0, 24.629696, 15.0], [0.0, 0.0, 1.0]],
            [[24.629694, 0.0, 20.0], [0.0, 24.629696, 15.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    scaled, scale_xy = camera_utils.rescale_intrinsics_to_image(
        K, image_hw=(240, 320), intrinsics_scale="auto"
    )

    assert scale_xy == (8.0, 8.0)
    np.testing.assert_allclose(scaled[:, 0, :], K[:, 0, :] * 8.0)
    np.testing.assert_allclose(scaled[:, 1, :], K[:, 1, :] * 8.0)
    np.testing.assert_allclose(scaled[:, 2, :], K[:, 2, :])
    np.testing.assert_allclose(K[:, 0, 2], 20.0)


def test_auto_keeps_pixel_intrinsics_unchanged():
    camera_utils = _load_camera_utils()
    K = np.array(
        [[[197.0, 0.0, 160.0], [0.0, 197.0, 120.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )

    scaled, scale_xy = camera_utils.rescale_intrinsics_to_image(
        K, image_hw=(240, 320), intrinsics_scale="auto"
    )

    assert scale_xy == (1.0, 1.0)
    np.testing.assert_allclose(scaled, K)


def test_auto_keeps_normalized_intrinsics_unchanged():
    camera_utils = _load_camera_utils()
    K = np.array(
        [[[0.62, 0.0, 0.5], [0.0, 0.82, 0.5], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )

    scaled, scale_xy = camera_utils.rescale_intrinsics_to_image(
        K, image_hw=(240, 320), intrinsics_scale="auto"
    )

    assert scale_xy == (1.0, 1.0)
    np.testing.assert_allclose(scaled, K)


def test_default_seva_frame_window_uses_official_context_length_not_clip_length():
    camera_utils = _load_camera_utils()

    assert camera_utils.resolve_seva_frame_window(num_frames=50, frame_window=21) == 21


def test_seva_target_size_uses_official_short_side_with_aspect_ratio():
    camera_utils = _load_camera_utils()

    assert camera_utils.resolve_seva_target_size(image_hw=(240, 320), l_short=576) == (
        576,
        768,
    )

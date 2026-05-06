import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common_v3 import eval_pose_sequence


def test_eval_pose_sequence_falls_back_for_degenerate_two_frame_alignment(tmp_path):
    pred = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    gt = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    gt[1, 0, 3] = 2.0

    metrics = eval_pose_sequence(pred, gt, seq="degenerate", save_dir=tmp_path)

    assert metrics["n_frames"] == 2
    assert metrics["alignment"] == "first_pose_scale_fallback"
    np.testing.assert_allclose(metrics["ate"], np.sqrt(2.0))
    np.testing.assert_allclose(metrics["rpe_trans"], 2.0)
    np.testing.assert_allclose(metrics["rpe_rot"], 0.0)
    assert "fallback" in (tmp_path / "degenerate_eval_metric.txt").read_text()

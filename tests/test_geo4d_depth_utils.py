import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo4d.depth_utils import match_depth_to_gt


def test_match_depth_to_gt_trims_padded_predictions_and_resizes_spatial_shape():
    pred = np.ones((16, 4, 6), dtype=np.float32)
    gt = np.ones((2, 8, 10), dtype=np.float32)

    matched = match_depth_to_gt(pred, gt)

    assert matched.shape == gt.shape

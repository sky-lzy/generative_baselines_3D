import math
import unittest

import numpy as np

from pi3_pairwise_benchmark.metrics import (
    calculate_auc_np,
    compute_pairwise_errors,
    invert_se3_stack,
    pairwise_indices,
)


def _pose(tx=0.0, ty=0.0, tz=0.0, angle_deg=0.0):
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[:3, 3] = [tx, ty, tz]
    return out


class MetricsTests(unittest.TestCase):
    def test_pairwise_indices_match_ten_view_pi3_contract(self):
        pairs = pairwise_indices(10)
        self.assertEqual(len(pairs), 45)
        self.assertEqual(pairs[0], (0, 1))
        self.assertEqual(pairs[-1], (8, 9))
        self.assertEqual(len(set(pairs)), len(pairs))

    def test_identity_predictions_have_perfect_auc(self):
        gt_w2c = np.stack([_pose(tx=float(i)) for i in range(10)])
        errors = compute_pairwise_errors(pred_w2c=gt_w2c, gt_w2c=gt_w2c)

        self.assertTrue(np.allclose(errors.rotation_deg, 0.0))
        self.assertTrue(np.allclose(errors.translation_deg, 0.0, atol=1e-5))
        auc, hist = calculate_auc_np(errors.rotation_deg, errors.translation_deg, max_threshold=30)
        self.assertEqual(auc, 1.0)
        self.assertEqual(hist[0], 1.0)

    def test_known_relative_rotation_is_reported_in_degrees(self):
        gt_w2c = np.stack([_pose(), _pose(tx=1.0)])
        pred_w2c = np.stack([_pose(), _pose(tx=1.0, angle_deg=10.0)])

        errors = compute_pairwise_errors(pred_w2c=pred_w2c, gt_w2c=gt_w2c)

        self.assertEqual(errors.rotation_deg.shape, (1,))
        self.assertTrue(np.isclose(errors.rotation_deg[0], 10.0, atol=1e-6))

    def test_translation_angle_uses_pi3_sign_ambiguity(self):
        gt_w2c = np.stack([_pose(), _pose(tx=1.0)])
        pred_w2c = np.stack([_pose(), _pose(tx=-1.0)])

        errors = compute_pairwise_errors(pred_w2c=pred_w2c, gt_w2c=gt_w2c)

        self.assertTrue(np.isclose(errors.translation_deg[0], 0.0, atol=1e-5))

    def test_c2w_w2c_round_trip(self):
        c2w = np.stack([_pose(tx=1.0, ty=2.0, tz=3.0, angle_deg=15.0), _pose(tx=-1.0)])
        w2c = invert_se3_stack(c2w)

        self.assertTrue(np.allclose(invert_se3_stack(w2c), c2w))


if __name__ == "__main__":
    unittest.main()

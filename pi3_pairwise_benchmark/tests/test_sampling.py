import unittest

from pi3_pairwise_benchmark.sampling import select_pairs


class SamplingTests(unittest.TestCase):
    def test_all_policy_matches_pi3_unordered_pairs(self):
        pairs = select_pairs(num_frames=10, policy="all")

        self.assertEqual(len(pairs), 45)
        self.assertEqual(pairs[0], (0, 1))
        self.assertEqual(pairs[-1], (8, 9))

    def test_long_policy_uses_large_frame_gaps(self):
        pairs = select_pairs(
            num_frames=50,
            policy="long",
            max_pairs=12,
            seed=7,
            sequence_key="seq-a",
            long_gap_min=30,
        )

        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(j - i >= 30 for i, j in pairs))

    def test_sampling_is_deterministic_per_sequence(self):
        kwargs = {
            "num_frames": 50,
            "policy": "long",
            "max_pairs": 8,
            "seed": 42,
            "sequence_key": "seq-a",
            "long_gap_min": 30,
        }

        self.assertEqual(select_pairs(**kwargs), select_pairs(**kwargs))
        self.assertNotEqual(select_pairs(**kwargs), select_pairs(**{**kwargs, "sequence_key": "seq-b"}))

    def test_short_medium_long_policies_use_requested_ranges(self):
        short = select_pairs(num_frames=50, policy="short", max_pairs=20, short_gap_min=1, short_gap_max=5)
        medium = select_pairs(num_frames=50, policy="medium", max_pairs=20, medium_gap_min=10, medium_gap_max=20)
        long = select_pairs(num_frames=50, policy="long", max_pairs=20, long_gap_min=30)

        self.assertTrue(all(1 <= j - i <= 5 for i, j in short))
        self.assertTrue(all(10 <= j - i <= 20 for i, j in medium))
        self.assertTrue(all(j - i >= 30 for i, j in long))

    def test_empty_policy_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No pairs"):
            select_pairs(num_frames=10, policy="long", long_gap_min=30)


if __name__ == "__main__":
    unittest.main()

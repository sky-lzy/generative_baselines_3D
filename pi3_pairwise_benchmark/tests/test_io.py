import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi3_pairwise_benchmark.io import (
    ManifestRow,
    load_manifest,
    write_json,
    write_prediction_pair,
    write_prediction_sequence,
)


def _pose_stack(n):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    poses[:, 0, 3] = np.arange(n, dtype=np.float64)
    return poses


class IoTests(unittest.TestCase):
    def test_manifest_row_requires_at_least_two_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {
                "dataset": "Re10K",
                "sequence_name": "seq",
                "image_ids": [0],
                "image_paths": [str(root / "0.png")],
                "gt_w2c": _pose_stack(1).tolist(),
                "gt_c2w": _pose_stack(1).tolist(),
            }

            with self.assertRaisesRegex(ValueError, "at least 2"):
                ManifestRow.from_json(payload)

    def test_manifest_row_rejects_mismatched_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {
                "dataset": "Re10K",
                "sequence_name": "seq",
                "image_ids": list(range(3)),
                "image_paths": [str(root / f"{i}.png") for i in range(2)],
                "gt_w2c": _pose_stack(3).tolist(),
                "gt_c2w": _pose_stack(3).tolist(),
            }

            with self.assertRaisesRegex(ValueError, "same number"):
                ManifestRow.from_json(payload)

    def test_manifest_row_accepts_fifty_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {
                "dataset": "Re10K",
                "sequence_name": "seq",
                "image_ids": list(range(50)),
                "image_paths": [str(root / f"{i}.png") for i in range(50)],
                "gt_w2c": _pose_stack(50).tolist(),
                "gt_c2w": _pose_stack(50).tolist(),
            }

            row = ManifestRow.from_json(payload)

            self.assertEqual(len(row.image_paths), 50)
            self.assertEqual(row.gt_w2c.shape, (50, 4, 4))

    def test_load_manifest_parses_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(10):
                (root / f"{i}.png").write_bytes(b"fake")
            row = {
                "dataset": "Re10K",
                "sequence_name": "seq",
                "image_ids": list(range(10)),
                "image_paths": [str(root / f"{i}.png") for i in range(10)],
                "gt_w2c": _pose_stack(10).tolist(),
                "gt_c2w": _pose_stack(10).tolist(),
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = load_manifest(manifest)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sequence_name, "seq")
            self.assertEqual(rows[0].gt_w2c.shape, (10, 4, 4))

    def test_write_prediction_pair_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "preds"
            pred_c2w = _pose_stack(2)
            gt_w2c = _pose_stack(2)

            write_prediction_pair(
                output_dir=out_dir,
                sequence_name="seq/a",
                pair=(0, 9),
                pred_c2w=pred_c2w,
                gt_w2c=gt_w2c,
                metrics={"Auc_30": 100.0},
            )

            pair_dir = out_dir / "seq_a" / "pair_00_09"
            self.assertTrue((pair_dir / "pred_c2w.npy").is_file())
            self.assertTrue((pair_dir / "pred_w2c.npy").is_file())
            self.assertTrue((pair_dir / "gt_w2c.npy").is_file())
            self.assertEqual(
                json.loads((pair_dir / "metrics.json").read_text())["Auc_30"],
                100.0,
            )

    def test_write_prediction_sequence_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "preds"
            pred_c2w = _pose_stack(50)
            gt_w2c = _pose_stack(50)

            write_prediction_sequence(
                output_dir=out_dir,
                sequence_name="seq/a",
                pred_c2w=pred_c2w,
                gt_w2c=gt_w2c,
                metrics={"Auc_30": 90.0, "num_pairs": 1225.0},
            )

            seq_dir = out_dir / "seq_a"
            self.assertTrue((seq_dir / "pred_c2w.npy").is_file())
            self.assertTrue((seq_dir / "pred_w2c.npy").is_file())
            self.assertTrue((seq_dir / "gt_w2c.npy").is_file())
            metrics = json.loads((seq_dir / "metrics.json").read_text())
            self.assertEqual(metrics["Auc_30"], 90.0)
            self.assertEqual(metrics["num_pairs"], 1225.0)

    def test_write_json_creates_parent(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "payload.json"
            write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()

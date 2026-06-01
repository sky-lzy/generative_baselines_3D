import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_re10k_contiguous_seq_map import build_contiguous_seq_map


def _write_re10k_txt(path: Path, num_frames: int) -> None:
    pose = " ".join(str(float(i == j)) for i in range(3) for j in range(4))
    intr = "0.5 0.5 0.5 0.5 0.0 0.0"
    lines = ["https://www.youtube.com/watch?v=fake_video"]
    for idx in range(num_frames):
        lines.append(f"{idx * 33333} {intr} {pose}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ContiguousSeqMapTests(unittest.TestCase):
    def test_builds_centered_contiguous_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_re10k_txt(root / "RealEstate10K" / "test" / "seq_a.txt", 100)

            seq_map = build_contiguous_seq_map(
                source_root=root,
                sequence_names=["seq_a"],
                split="test",
                num_frames=50,
                max_sequences=1,
                window="center",
            )

            self.assertEqual(seq_map, {"seq_a": list(range(25, 75))})

    def test_skips_sequences_without_enough_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_re10k_txt(root / "RealEstate10K" / "test" / "seq_a.txt", 20)

            seq_map = build_contiguous_seq_map(
                source_root=root,
                sequence_names=["seq_a"],
                split="test",
                num_frames=50,
                max_sequences=1,
                window="start",
            )

            self.assertEqual(seq_map, {})


if __name__ == "__main__":
    unittest.main()

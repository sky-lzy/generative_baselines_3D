import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo4d.path_utils import resolve_geo4d_path


def test_resolve_geo4d_path_prefers_geo4d_dir_for_relative_paths(tmp_path):
    geo4d_dir = tmp_path / "Geo4D"
    target = geo4d_dir / "checkpoints" / "geo4d" / "vae.ckpt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"checkpoint")

    resolved = resolve_geo4d_path("checkpoints/geo4d/vae.ckpt", geo4d_dir)

    assert resolved == target


def test_resolve_geo4d_path_preserves_absolute_paths(tmp_path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"checkpoint")

    resolved = resolve_geo4d_path(str(target), tmp_path / "Geo4D")

    assert resolved == target

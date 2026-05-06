"""Path helpers for vendored Geo4D evaluators."""

from __future__ import annotations

from pathlib import Path


def resolve_geo4d_path(path: str | Path, geo4d_dir: str | Path) -> Path:
    """Resolve absolute, cwd-relative, or Geo4D-root-relative paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return Path(geo4d_dir) / candidate

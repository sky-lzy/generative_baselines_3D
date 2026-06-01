from __future__ import annotations

import sys
from pathlib import Path


def setup_root(search_from, indicator=".project-root", pythonpath=False, cwd=False):
    """Small compatibility shim for Pi3's use of `rootutils.setup_root`.

    It walks upward from `search_from` until it finds `indicator`, optionally
    prepends that directory to `sys.path`, and optionally changes cwd.
    """
    start = Path(search_from).resolve()
    cur = start if start.is_dir() else start.parent
    while True:
        if (cur / indicator).exists():
            root = cur
            break
        if cur.parent == cur:
            root = start if start.is_dir() else start.parent
            break
        cur = cur.parent
    if pythonpath and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if cwd:
        import os

        os.chdir(root)
    return root


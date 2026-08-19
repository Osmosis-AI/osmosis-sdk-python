"""Locate the ``uv`` executable that ships alongside this interpreter.

Two callers need it: the Harbor bundle packager builds wheels with it, and
local evaluation launches the rollout server through it. It lives here rather
than in ``osmosis_ai.packaging`` because that module requires the ``harbor``
extra, and local evaluation must not.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def _uv_executable() -> str:
    """Find uv installed with this interpreter, then fall back to PATH."""
    script_dirs = (Path(sysconfig.get_path("scripts")), Path(sys.executable).parent)
    for script_dir in dict.fromkeys(script_dirs):
        for name in ("uv", "uv.exe"):
            candidate = script_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError(
            "uv is required to build rollout bundles; install osmosis-ai[harbor]"
        )
    return executable

"""Internal utilities for atomic file operations."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_secure_dir(directory: Path) -> None:
    """Ensure *directory* exists with 0o700 permissions (owner-only access).

    Creates the directory (and parents) if missing, and tightens permissions
    if group/other bits are set.
    """
    try:
        current_mode = directory.stat().st_mode
        if current_mode & 0o077:  # group or others have permissions
            os.chmod(directory, 0o700)
    except FileNotFoundError:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)


def atomic_write_json(
    path: Path,
    data: Any,
    mode: int = 0o600,
    indent: int = 2,
) -> None:
    """Atomically write JSON data to a file with proper permissions.

    Uses mkstemp + os.replace for atomic writes, ensuring credentials
    are never in a world-readable state.

    Args:
        path: Target file path.
        data: Data to serialize as JSON.
        mode: File permissions (default: 0o600 for owner-only read/write).
        indent: JSON indentation (default: 2).
    """
    parent = path.parent
    ensure_secure_dir(parent)

    # Create temp file in same directory for atomic rename.
    # Set permissions on the fd BEFORE os.replace() so the file already has
    # the desired mode when it appears at the final path — no race window.
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fchmod(f.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

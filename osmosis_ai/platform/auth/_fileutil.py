"""Internal utilities for atomic file operations."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


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

    # Ensure parent directory exists with secure permissions
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
    else:
        # Fix permissions if needed
        current_mode = parent.stat().st_mode
        if current_mode & 0o077:  # If group or others have any permissions
            os.chmod(parent, 0o700)

    # Create temp file in same directory for atomic rename
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
        os.chmod(path, mode)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

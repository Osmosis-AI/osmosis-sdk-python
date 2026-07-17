"""Validation for untrusted ids (e.g. ``rollout_id``) that get joined onto paths."""

from __future__ import annotations

from pathlib import Path

# Reject both POSIX and Windows separators so ids stay portable single segments
# even when validated on Linux (where ``os.altsep`` is None).
_PATH_SEPARATORS = ("/", "\\")


def is_single_path_segment(name: str) -> bool:
    """True when ``name`` is a plain path component, not a traversal or root escape."""
    if not name or name in (".", ".."):
        return False
    # NUL truncates C-level paths and can turn a later open() into a sink error
    # instead of the intended 422 at the request boundary.
    if "\x00" in name or any(sep in name for sep in _PATH_SEPARATORS):
        return False
    return Path(name).name == name


def ensure_single_path_segment(value: str, *, label: str = "id") -> str:
    """Pydantic field-validator body: return ``value`` or raise ``ValueError``."""
    if not is_single_path_segment(value):
        raise ValueError(
            f"{label} must be a single path segment "
            "(no path separators, and not '.' or '..')"
        )
    return value

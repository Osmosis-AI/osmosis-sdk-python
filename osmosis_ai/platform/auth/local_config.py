"""Local configuration for legacy auth-local cache files.

File: ~/.config/osmosis/config.json

Cache data (subscription status) is stored as individual JSON files under
~/.config/osmosis/cache/ so that concurrent writes to different workspaces
never conflict.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ._fileutil import atomic_write_json
from .config import CACHE_DIR, CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.json"


def _safe_ws_name(name: str) -> str:
    """Sanitise a workspace name for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()


def _write_cache(path: Path, data: Any) -> None:
    """Atomically write *data* as JSON to *path* (tempfile + os.replace)."""
    atomic_write_json(path, data, mode=0o600)


def _read_cache(path: Path) -> Any | None:
    """Read a single JSON cache file. Returns None on missing / corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Workspace cleanup ────────────────────────────────────────────


def clear_workspace_data(workspace_name: str) -> None:
    """Remove cached subscription data for a workspace."""
    safe = _safe_ws_name(workspace_name)
    path = CACHE_DIR / f"subscription_{safe}.json"
    with contextlib.suppress(OSError):
        path.unlink()


# ── Subscription status cache ────────────────────────────────────


def save_subscription_status(workspace_name: str, has_subscription: bool) -> None:
    """Cache the subscription status for a workspace."""
    safe = _safe_ws_name(workspace_name)
    path = CACHE_DIR / f"subscription_{safe}.json"
    _write_cache(
        path, {"has_subscription": has_subscription, "refreshed_at": time.time()}
    )


def load_subscription_status(
    workspace_name: str, max_age: float | None = None
) -> bool | None:
    """Load cached subscription status for a workspace.

    Args:
        workspace_name: Name of the workspace.
        max_age: Maximum cache age in seconds. If the cached entry is older
            than this, ``None`` is returned so the caller can refresh.
            Pass ``None`` to accept any age.

    Returns:
        True/False if cached (and within *max_age*), or None if missing/expired.
    """
    safe = _safe_ws_name(workspace_name)
    path = CACHE_DIR / f"subscription_{safe}.json"
    data = _read_cache(path)
    if not isinstance(data, dict):
        return None
    if max_age is not None:
        refreshed_at = data.get("refreshed_at")
        if (
            not isinstance(refreshed_at, (int, float))
            or (time.time() - refreshed_at) > max_age
        ):
            return None
    return data.get("has_subscription")


def clear_all_local_data() -> None:
    """Remove legacy auth-local preferences and cache files.

    This is intentionally scoped to ~/.config/osmosis/config.json and cache
    files. Project-to-workspace mappings live under ~/.osmosis/config.json and
    must survive logout/session reset.
    """
    with contextlib.suppress(OSError):
        CONFIG_FILE.unlink()
    with contextlib.suppress(FileNotFoundError):
        for path in CACHE_DIR.iterdir():
            with contextlib.suppress(OSError):
                path.unlink()


def reset_session() -> None:
    """Complete session teardown: credentials and legacy preferences/cache.

    Single entry point for all "end session" paths (logout, login --force,
    401 expiry, user identity change) to ensure no stale auth-local state
    remains.
    """
    from .credentials import delete_credentials

    delete_credentials()
    clear_all_local_data()

"""Local configuration for user preferences (default project, etc.).

Stored separately from credentials to keep secrets and preferences apart.
File: ~/.config/osmosis/config.json

Cache data (project lists, subscription status) is stored as individual
files under ~/.config/osmosis/cache/ so that concurrent writes to different
workspaces never conflict.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.json"


# ── Internal helpers ────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        import sys

        print(
            f"Warning: config file is corrupted ({CONFIG_FILE}), using defaults.",
            file=sys.stderr,
        )
        return {}
    except OSError as exc:
        import sys

        print(
            f"Warning: cannot read config file ({CONFIG_FILE}): {exc}",
            file=sys.stderr,
        )
        return {}


def _save_config(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_FILE.with_suffix(".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp.rename(CONFIG_FILE)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def _safe_ws_name(name: str) -> str:
    """Sanitise a workspace name for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()


def _write_cache(path: Path, data: Any) -> None:
    """Atomically write *data* as JSON to *path* (tempfile + os.replace)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_cache(path: Path) -> Any | None:
    """Read a single JSON cache file. Returns None on missing / corrupt."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── Default project ──────────────────────────────────────────────


def get_default_project(workspace_name: str) -> dict[str, str] | None:
    """Get the default project for a workspace.

    Returns:
        Dict with 'project_id' and 'project_name', or None.
    """
    config = _load_config()
    defaults = config.get("defaults", {})
    entry = defaults.get(workspace_name)
    if (
        entry
        and isinstance(entry, dict)
        and "project_id" in entry
        and "project_name" in entry
    ):
        return entry
    # Corrupted entry (missing required keys) — treat as unset and clean up.
    if entry is not None:
        clear_default_project(workspace_name)
    return None


def set_default_project(
    workspace_name: str, project_id: str, project_name: str
) -> None:
    """Set the default project for a workspace."""
    config = _load_config()
    if "defaults" not in config:
        config["defaults"] = {}
    config["defaults"][workspace_name] = {
        "project_id": project_id,
        "project_name": project_name,
    }
    _save_config(config)


def clear_default_project(workspace_name: str) -> None:
    """Remove the default project for a workspace."""
    config = _load_config()
    defaults = config.get("defaults", {})
    if workspace_name in defaults:
        del defaults[workspace_name]
        _save_config(config)


# ── Workspace cleanup ────────────────────────────────────────────


def clear_workspace_data(workspace_name: str) -> None:
    """Remove all local data for a workspace (defaults + cache files)."""
    # Clear default project from config.json
    config = _load_config()
    defaults = config.get("defaults", {})
    if workspace_name in defaults:
        del defaults[workspace_name]
        _save_config(config)

    # Remove cache files
    safe = _safe_ws_name(workspace_name)
    for prefix in ("projects", "subscription"):
        path = CACHE_DIR / f"{prefix}_{safe}.json"
        with contextlib.suppress(OSError):
            path.unlink()


# ── Project cache ────────────────────────────────────────────────


def save_workspace_projects(
    workspace_name: str, projects: list[dict[str, Any]]
) -> None:
    """Cache the project list for a workspace."""
    safe = _safe_ws_name(workspace_name)
    path = CACHE_DIR / f"projects_{safe}.json"
    _write_cache(path, {"projects": projects, "refreshed_at": time.time()})


def load_workspace_projects(
    workspace_name: str,
) -> tuple[list[dict[str, Any]], float | None]:
    """Load cached projects and their refresh timestamp for a workspace.

    Returns:
        Tuple of (projects list, refreshed_at timestamp or None).
    """
    safe = _safe_ws_name(workspace_name)
    path = CACHE_DIR / f"projects_{safe}.json"
    data = _read_cache(path)
    if not isinstance(data, dict):
        return [], None
    return data.get("projects", []), data.get("refreshed_at")


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
        if refreshed_at is None or (time.time() - refreshed_at) > max_age:
            return None
    return data.get("has_subscription")

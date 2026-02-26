"""Local configuration for user preferences (default project, etc.).

Stored separately from credentials to keep secrets and preferences apart.
File: ~/.config/osmosis/config.json
"""

from __future__ import annotations

import json
import time
from typing import Any

from .config import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.json"


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
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


# ── Default project ──────────────────────────────────────────────


def get_default_project(workspace_name: str) -> dict[str, str] | None:
    """Get the default project for a workspace.

    Returns:
        Dict with 'project_id' and 'project_name', or None.
    """
    config = _load_config()
    defaults = config.get("defaults", {})
    entry = defaults.get(workspace_name)
    if entry and isinstance(entry, dict) and "project_id" in entry:
        return entry
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
    """Clear the default project for a workspace."""
    config = _load_config()
    defaults = config.get("defaults", {})
    if workspace_name in defaults:
        del defaults[workspace_name]
        _save_config(config)


# ── Project cache ────────────────────────────────────────────────


def save_workspace_projects(workspace_name: str, projects: list[dict]) -> None:
    """Cache the project list for a workspace."""
    config = _load_config()
    if "project_cache" not in config:
        config["project_cache"] = {}
    config["project_cache"][workspace_name] = {
        "projects": projects,
        "refreshed_at": time.time(),
    }
    _save_config(config)


def load_workspace_projects(
    workspace_name: str,
) -> tuple[list[dict], float | None]:
    """Load cached projects and their refresh timestamp for a workspace.

    Returns:
        Tuple of (projects list, refreshed_at timestamp or None).
    """
    config = _load_config()
    cache = config.get("project_cache", {}).get(workspace_name, {})
    return cache.get("projects", []), cache.get("refreshed_at")

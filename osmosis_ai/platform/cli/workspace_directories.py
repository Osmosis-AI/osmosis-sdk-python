"""Remembered local directories for each workspace on this machine.

Callers must verify a recalled path against the filesystem; reality wins.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from osmosis_ai.platform.auth.config import CONFIG_DIR, get_platform_url
from osmosis_ai.platform.auth.fileutil import atomic_write_json

WORKSPACE_DIRECTORIES_FILE = CONFIG_DIR / "workspace-directories.json"

_VERSION = 1


def _read() -> dict[str, dict[str, str]]:
    try:
        with open(WORKSPACE_DIRECTORIES_FILE, encoding="utf-8") as f:
            data: Any = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(data, dict) or data.get("version") != _VERSION:
        return {}
    platforms = data.get("platforms")
    if not isinstance(platforms, dict):
        return {}
    return {
        platform_url: {
            workspace_id: path
            for workspace_id, path in entries.items()
            if isinstance(workspace_id, str) and isinstance(path, str)
        }
        for platform_url, entries in platforms.items()
        if isinstance(platform_url, str) and isinstance(entries, dict)
    }


def _write(platforms: dict[str, dict[str, str]]) -> None:
    with contextlib.suppress(OSError):
        atomic_write_json(
            WORKSPACE_DIRECTORIES_FILE,
            {"version": _VERSION, "platforms": platforms},
            mode=0o600,
        )


def remember_workspace_directory(workspace_id: str, path: Path) -> None:
    platforms = _read()
    entries = platforms.setdefault(get_platform_url(), {})
    entries[workspace_id] = str(path.expanduser().resolve())
    _write(platforms)


def recall_workspace_directory(workspace_id: str) -> Path | None:
    path = _read().get(get_platform_url(), {}).get(workspace_id)
    return Path(path) if path else None


def forget_workspace_directory(workspace_id: str) -> None:
    platforms = _read()
    entries = platforms.get(get_platform_url())
    if not entries or entries.pop(workspace_id, None) is None:
        return
    _write(platforms)


__all__ = [
    "forget_workspace_directory",
    "recall_workspace_directory",
    "remember_workspace_directory",
]

"""Local configuration for legacy auth-local session files."""

from __future__ import annotations

import contextlib

from .config import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.json"


def _clear_legacy_config_file() -> None:
    with contextlib.suppress(OSError):
        CONFIG_FILE.unlink()


def clear_all_local_data() -> None:
    """Clear local CLI credentials and non-workspace-directory runtime state."""
    reset_session()


def reset_session() -> None:
    """Complete session teardown: credentials and legacy session config.

    Single entry point for all "end session" paths (logout, login --force,
    401 expiry, user identity change) to ensure no stale auth-local session
    state remains.
    """
    from .credentials import delete_credentials

    delete_credentials()
    _clear_legacy_config_file()

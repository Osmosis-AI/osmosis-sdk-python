"""Local configuration for legacy auth-local session files."""

from __future__ import annotations

import contextlib

from .config import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.json"


def _clear_legacy_config_file() -> None:
    with contextlib.suppress(OSError):
        CONFIG_FILE.unlink()


def reset_session() -> None:
    """Complete current-platform session teardown: credentials and legacy config.

    Logout is the only automatic caller; request failures never mutate stored
    credentials.
    """
    from .credentials import delete_credentials

    delete_credentials()
    _clear_legacy_config_file()

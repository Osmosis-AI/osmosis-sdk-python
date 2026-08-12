"""Shared Typer Option factories for list/logs pagination flags."""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_LOG_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

_CURSOR_HELP = "Page further back using the next_cursor value from a previous page."
_LOG_LIMIT_HELP = "Maximum number of recent log entries to show."


def limit_option(help: str, *, max_value: int = MAX_PAGE_SIZE) -> Any:
    return typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=max_value,
        help=help,
    )


def log_limit_option(help: str = _LOG_LIMIT_HELP) -> Any:
    return limit_option(help, max_value=MAX_LOG_PAGE_SIZE)


def all_option(help: str) -> Any:
    return typer.Option(False, "--all", help=help)


def cursor_option(help: str = _CURSOR_HELP) -> Any:
    return typer.Option(None, "--cursor", help=help)

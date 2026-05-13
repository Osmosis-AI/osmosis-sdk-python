"""Shared human-display helpers for CLI output."""

from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Any


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _localize(dt: datetime, *, tz: tzinfo | None = None) -> datetime:
    if tz is not None:
        return dt.astimezone(tz)
    return dt.astimezone()


def local_timezone_label(
    *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    if tz is not None:
        base_now = now or datetime.now(tz)
        local_now = base_now.astimezone(tz) if base_now.tzinfo else base_now
    else:
        local_now = now or datetime.now().astimezone()
    return local_now.tzname() or "Local"


def created_column_label(
    *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    return f"Created ({local_timezone_label(now=now, tz=tz)})"


def format_local_date(
    value: str | None, *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)[:10]
    return _localize(parsed, tz=tz).strftime("%Y-%m-%d %H:%M")


def format_local_datetime(
    value: str | None, *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)
    return _localize(parsed, tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_reward(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)

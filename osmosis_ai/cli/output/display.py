"""Shared human-display helpers for CLI output."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _localize(dt: datetime, *, now: datetime | None = None) -> datetime:
    local_tz = (now or datetime.now().astimezone()).tzinfo
    if dt.tzinfo is None:
        return dt
    if local_tz is None:
        return dt.astimezone()
    return dt.astimezone(local_tz)


def local_timezone_label(*, now: datetime | None = None) -> str:
    local_now = now or datetime.now().astimezone()
    return local_now.tzname() or "Local"


def created_column_label(*, now: datetime | None = None) -> str:
    return f"Created ({local_timezone_label(now=now)})"


def format_local_date(value: str | None, *, now: datetime | None = None) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)[:10]
    return _localize(parsed, now=now).strftime("%Y-%m-%d %H:%M")


def format_local_datetime(value: str | None, *, now: datetime | None = None) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)
    return _localize(parsed, now=now).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_reward(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)

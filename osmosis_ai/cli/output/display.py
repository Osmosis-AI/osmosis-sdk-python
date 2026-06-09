"""Shared human-display helpers for CLI output."""

from __future__ import annotations

from datetime import datetime, tzinfo


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


def _twelve_hour_time(dt: datetime, *, with_seconds: bool = False) -> str:
    """12-hour clock time with AM/PM and no leading-zero hour (e.g. ``6:16 PM``)."""
    hour = dt.hour % 12 or 12
    meridiem = "AM" if dt.hour < 12 else "PM"
    seconds = f":{dt.second:02d}" if with_seconds else ""
    return f"{hour}:{dt.minute:02d}{seconds} {meridiem}"


def format_local_date(
    value: str | None, *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)[:10]
    local = _localize(parsed, tz=tz)
    return f"{local.strftime('%Y-%m-%d')} {_twelve_hour_time(local)} {local.strftime('%Z')}"


def format_local_datetime(
    value: str | None, *, now: datetime | None = None, tz: tzinfo | None = None
) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "" if value is None else str(value)
    local = _localize(parsed, tz=tz)
    return (
        f"{local.strftime('%Y-%m-%d')} "
        f"{_twelve_hour_time(local, with_seconds=True)} {local.strftime('%Z')}"
    )


def format_duration_ms(duration_ms: float) -> str:
    """Human-readable duration from milliseconds (e.g. ``2h 47m``)."""
    duration_ms = max(0.0, duration_ms)
    total_seconds = duration_ms / 1000
    if total_seconds < 60:
        return (
            f"{total_seconds:.1f}s" if total_seconds % 1 else f"{int(total_seconds)}s"
        )

    total_seconds_int = int(total_seconds)
    minutes, seconds = divmod(total_seconds_int, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"

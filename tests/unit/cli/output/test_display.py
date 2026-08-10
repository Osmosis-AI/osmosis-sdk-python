from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from osmosis_ai.cli.output.display import (
    format_duration_ms,
    format_elapsed,
    format_local_date,
    format_local_datetime,
    format_relative_time,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


def test_format_local_date_uses_explicit_timezone() -> None:
    formatted = format_local_date("2026-05-13T12:34:56Z", tz=ZoneInfo("UTC"))

    assert formatted == "2026-05-13 12:34 PM UTC"


def test_format_local_date_includes_per_timestamp_timezone_rules() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    assert (
        format_local_date("2026-01-01T12:00:00Z", tz=pacific)
        == "2026-01-01 4:00 AM PST"
    )
    assert (
        format_local_date("2026-07-01T12:00:00Z", tz=pacific)
        == "2026-07-01 5:00 AM PDT"
    )


def test_format_local_datetime_falls_back_for_invalid_input() -> None:
    assert format_local_datetime("not-a-date") == "not-a-date"


def test_format_local_datetime_uses_per_timestamp_timezone_rules() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    assert (
        format_local_datetime("2026-01-01T12:00:00Z", tz=pacific)
        == "2026-01-01 4:00:00 AM PST"
    )
    assert (
        format_local_datetime("2026-07-01T12:00:00Z", tz=pacific)
        == "2026-07-01 5:00:00 AM PDT"
    )


def test_format_local_datetime_does_not_use_now_offset_for_conversion() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    july_fixed_offset_now = datetime(
        2026,
        7,
        1,
        5,
        0,
        tzinfo=timezone(timedelta(hours=-7), "PDT"),
    )

    assert (
        format_local_datetime(
            "2026-01-01T12:00:00Z",
            now=july_fixed_offset_now,
            tz=pacific,
        )
        == "2026-01-01 4:00:00 AM PST"
    )
    assert (
        format_local_datetime("2026-01-01T12:00:00Z", now=july_fixed_offset_now)
        != "2026-01-01 5:00:00 AM PDT"
    )


def test_format_local_datetime_does_not_localize_offsetless_input() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    assert (
        format_local_datetime("2026-01-01T12:00:00", tz=pacific)
        == "2026-01-01T12:00:00"
    )


def test_format_local_date_uses_compact_fallback_for_offsetless_input() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    assert format_local_date("2026-01-01T12:00:00", tz=pacific) == "2026-01-01"


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (-500, "0s"),
        (1500, "1.5s"),
        (45000, "45s"),
        (60000, "1m"),
        (754000, "12m 34s"),
        (3600000, "1h"),
        (9900000, "2h 45m"),
        (86400000, "1d"),
        (90000000, "1d 1h"),
    ],
)
def test_format_duration_ms(duration_ms: float, expected: str) -> None:
    assert format_duration_ms(duration_ms) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-08-04T11:59:30Z", "just now"),
        ("2026-08-04T11:35:00Z", "25m ago"),
        ("2026-08-04T07:00:00Z", "5h ago"),
        ("2026-08-02T12:00:00Z", "2d ago"),
        ("2026-07-14T12:00:00Z", "3w ago"),
        ("2026-05-04T12:00:00Z", "3mo ago"),
        ("2024-08-04T12:00:00Z", "2y ago"),
    ],
)
def test_format_relative_time(value: str, expected: str) -> None:
    assert format_relative_time(value, now=NOW) == expected


def test_format_relative_time_falls_back_for_invalid_input() -> None:
    assert format_relative_time("not-a-date", now=NOW) == "not-a-date"
    assert format_relative_time(None, now=NOW) == ""


def test_format_elapsed_measures_to_completion_or_now() -> None:
    assert (
        format_elapsed("2026-08-04T09:15:00Z", "2026-08-04T11:00:00Z", now=NOW)
        == "1h 45m"
    )
    assert format_elapsed("2026-08-04T11:00:00Z", None, now=NOW) == "1h"
    assert format_elapsed(None, "2026-08-04T11:00:00Z", now=NOW) is None

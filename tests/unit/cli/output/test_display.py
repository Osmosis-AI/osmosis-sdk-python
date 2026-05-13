from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from osmosis_ai.cli.output.display import (
    format_local_date,
    format_local_datetime,
    format_reward,
    local_timezone_label,
)


def test_format_reward_uses_two_decimal_places() -> None:
    assert format_reward(0.875) == "0.88"
    assert format_reward(None) == ""


def test_local_timezone_label_returns_non_empty_text() -> None:
    assert local_timezone_label()


def test_format_local_date_uses_explicit_timezone() -> None:
    formatted = format_local_date("2026-05-13T12:34:56Z", tz=ZoneInfo("UTC"))

    assert formatted == "2026-05-13 12:34"


def test_format_local_datetime_falls_back_for_invalid_input() -> None:
    assert format_local_datetime("not-a-date") == "not-a-date"


def test_format_local_datetime_uses_per_timestamp_timezone_rules() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        pytest.skip("America/Los_Angeles timezone data is unavailable")

    assert (
        format_local_datetime("2026-01-01T12:00:00Z", tz=pacific)
        == "2026-01-01 04:00:00 PST"
    )
    assert (
        format_local_datetime("2026-07-01T12:00:00Z", tz=pacific)
        == "2026-07-01 05:00:00 PDT"
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
        == "2026-01-01 04:00:00 PST"
    )
    assert (
        format_local_datetime("2026-01-01T12:00:00Z", now=july_fixed_offset_now)
        != "2026-01-01 05:00:00 PDT"
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

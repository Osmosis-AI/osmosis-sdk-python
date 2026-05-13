from __future__ import annotations

from datetime import UTC, datetime

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


def test_format_local_date_uses_injected_now_timezone() -> None:
    now = datetime(2026, 5, 13, 12, 0, tzinfo=UTC).astimezone()
    formatted = format_local_date("2026-05-13T12:34:56Z", now=now)

    assert formatted.startswith("2026-05-13 ")
    assert len(formatted) == 16


def test_format_local_datetime_falls_back_for_invalid_input() -> None:
    assert format_local_datetime("not-a-date") == "not-a-date"

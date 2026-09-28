"""Tests for the CommandResult shapes."""

from __future__ import annotations

from typing import Literal, get_args, get_origin, get_type_hints

from osmosis_ai.cli.output.result import (
    DetailResult,
    ListColumn,
    ListResult,
    SectionedListResult,
)


def test_detail_result_preserves_positional_exit_code() -> None:
    assert DetailResult("T", {"id": "x"}, [], 7).exit_code == 7


def test_list_result_preserves_positional_exit_code() -> None:
    result = ListResult("T", [], 0, False, None, [], {}, None, 7)

    assert result.exit_code == 7


def test_list_column_overflow_supports_rich_ignore() -> None:
    overflow_type = get_type_hints(ListColumn)["overflow"]
    literal_values: list[str] = []
    for arg in get_args(overflow_type):
        if get_origin(arg) is Literal:
            literal_values.extend(get_args(arg))

    assert "ignore" in literal_values


def test_sectioned_list_result_preserves_positional_exit_code() -> None:
    assert SectionedListResult([], {}, 7).exit_code == 7

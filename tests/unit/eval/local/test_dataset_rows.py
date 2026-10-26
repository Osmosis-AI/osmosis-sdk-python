"""dataset_rows.jsonl line format: the prompt and label text the platform searches."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from osmosis_ai.eval.local.dataset import normalize_row
from osmosis_ai.eval.local.dataset_rows import (
    _dataset_row_label,
    _dataset_row_prompt,
    render_dataset_rows,
)


def _render(*raw_rows: dict[str, Any]) -> bytes:
    return render_dataset_rows(
        normalize_row(raw, row_index=index, source_row_index=index)
        for index, raw in enumerate(raw_rows)
    )


def _only_line(output: bytes) -> dict[str, Any]:
    return json.loads(output.decode("utf-8"))


# --------------------------------------------------------------------------- #
# Rows loaded by the SDK
# --------------------------------------------------------------------------- #


def test_prompt_mode_row_has_prompt_and_label() -> None:
    assert _render({"user_prompt": "What is 2+2?", "ground_truth": "4"}) == (
        b'{"row_index": 0, "prompt": "What is 2+2?", "label": "4"}'
    )


def test_system_prompt_is_not_part_of_the_prompt() -> None:
    raw = {
        "system_prompt": "Be terse",
        "user_prompt": "What is 2+2?",
        "ground_truth": "4",
    }
    assert _render(raw) == b'{"row_index": 0, "prompt": "What is 2+2?", "label": "4"}'


def test_label_alias_is_used_when_ground_truth_is_absent() -> None:
    assert _only_line(_render({"user_prompt": "q", "label": "L"})) == {
        "row_index": 0,
        "prompt": "q",
        "label": "L",
    }


def test_metadata_only_row_uses_sorted_metadata_json() -> None:
    line = _only_line(_render({"metadata": {"b": 1, "a": "x"}}))

    assert line["prompt"] == '{"a": "x", "b": 1}'


def test_metadata_values_without_a_json_form_are_stringified() -> None:
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    line = _only_line(_render({"metadata": {"when": when, "a": 1}}))

    assert line["prompt"] == '{"a": 1, "when": "2026-01-02 03:04:05+00:00"}'


def test_user_prompt_takes_precedence_over_metadata() -> None:
    line = _only_line(_render({"user_prompt": "q", "metadata": {"a": 1}}))

    assert line == {"row_index": 0, "prompt": "q"}


def test_system_prompt_with_metadata_still_uses_the_metadata_text() -> None:
    line = _only_line(_render({"system_prompt": "Be terse", "metadata": {"a": 1}}))

    assert line == {"row_index": 0, "prompt": '{"a": 1}'}


@pytest.mark.parametrize(
    ("length", "kept"), [(1999, 1999), (2000, 2000), (2001, 2000), (3000, 2000)]
)
def test_prompt_is_cut_at_2000_characters(length: int, kept: int) -> None:
    line = _only_line(_render({"user_prompt": "a" * length, "ground_truth": "x"}))

    assert line["prompt"] == "a" * kept


@pytest.mark.parametrize(
    ("length", "kept"), [(499, 499), (500, 500), (501, 500), (600, 500)]
)
def test_label_is_cut_at_500_characters(length: int, kept: int) -> None:
    line = _only_line(_render({"user_prompt": "q", "ground_truth": "b" * length}))

    assert line["label"] == "b" * kept


def test_a_row_without_a_label_omits_the_key() -> None:
    assert _render({"metadata": {"a": 1}}) == (
        b'{"row_index": 0, "prompt": "{\\"a\\": 1}"}'
    )


def test_non_ascii_text_is_written_as_utf8_not_escaped() -> None:
    output = _render({"user_prompt": "héllo 世界 🙂", "ground_truth": "→ ok"})

    expected = '{"row_index": 0, "prompt": "héllo 世界 🙂", "label": "→ ok"}'
    assert output == expected.encode("utf-8")
    assert b"\\u" not in output


def test_truncation_counts_characters_not_bytes() -> None:
    line = _only_line(_render({"user_prompt": "é" * 2001, "ground_truth": "ü" * 501}))

    assert line["prompt"] == "é" * 2000
    assert line["label"] == "ü" * 500


def test_lone_surrogates_are_replaced_so_the_file_stays_valid_utf8() -> None:
    output = _render({"user_prompt": "a\ud800b", "ground_truth": "c\udfffd"})

    assert output == b'{"row_index": 0, "prompt": "a?b", "label": "c?d"}'


def test_rows_are_joined_by_newlines_with_no_trailing_newline() -> None:
    output = _render(
        {"user_prompt": "first", "ground_truth": "1"},
        {"user_prompt": "second", "ground_truth": "2"},
        {"user_prompt": "third", "ground_truth": "3"},
    )

    assert output == (
        b'{"row_index": 0, "prompt": "first", "label": "1"}\n'
        b'{"row_index": 1, "prompt": "second", "label": "2"}\n'
        b'{"row_index": 2, "prompt": "third", "label": "3"}'
    )


def test_row_index_is_the_selected_position_not_the_source_offset() -> None:
    row = normalize_row(
        {"user_prompt": "q", "ground_truth": "a"}, row_index=0, source_row_index=7
    )

    assert render_dataset_rows([row]) == (
        b'{"row_index": 0, "prompt": "q", "label": "a"}'
    )


def test_no_rows_render_to_an_empty_file() -> None:
    assert render_dataset_rows([]) == b""


# --------------------------------------------------------------------------- #
# Raw row semantics (rows shaped as chat messages or with a prompt alias)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            {"system_prompt": "Be terse", "user_prompt": "What is 2+2?"},
            "What is 2+2?",
            id="system-prompt-excluded",
        ),
        pytest.param({"prompt": "p"}, "p", id="prompt-alias"),
        pytest.param(
            {
                "messages": [
                    {"role": "system", "content": "SYSTEMONLY"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                            "bare string block",
                        ],
                    },
                    {"role": "assistant", "content": "   "},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
            "Hello → Hi",
            id="messages-with-text-blocks",
        ),
        pytest.param(
            {"messages": ["bare", {"role": "user", "content": "q"}]},
            "q",
            id="messages-skip-non-dict-entries",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": None, "tool_calls": []},
                ]
            },
            "q",
            id="messages-skip-non-text-content",
        ),
        pytest.param(
            {"metadata": {"b": 1, "a": "x"}}, '{"a": "x", "b": 1}', id="metadata"
        ),
        pytest.param(
            {"user_prompt": "q", "metadata": {"a": 1}}, "q", id="prompt-over-metadata"
        ),
        pytest.param({}, "{}", id="empty-row-falls-back-to-its-repr"),
    ],
)
def test_prompt_flattens_the_non_system_messages_of_a_row(
    row: dict[str, Any], expected: str
) -> None:
    assert _dataset_row_prompt(row) == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"ground_truth": 4}, "4"),
        ({"ground_truth": 0}, "0"),
        ({"ground_truth": False}, "False"),
        ({"ground_truth": 0, "label": "L"}, "0"),
        ({"ground_truth": "", "label": "L"}, "L"),
        ({"ground_truth": None, "label": "L"}, "L"),
        ({"label": "only"}, "only"),
        ({"ground_truth": None, "label": ""}, None),
        ({}, None),
    ],
)
def test_label_is_omitted_only_when_none_or_empty(
    row: dict[str, Any], expected: str | None
) -> None:
    assert _dataset_row_label(row) == expected

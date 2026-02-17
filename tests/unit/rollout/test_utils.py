# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.utils."""

from __future__ import annotations

from typing import Any

import pytest

from osmosis_ai.rollout import (
    count_messages_by_role,
    get_message_content,
    get_message_role,
    is_assistant_message,
    is_tool_message,
    is_user_message,
    normalize_stop,
    parse_tool_calls,
)

# =============================================================================
# parse_tool_calls Tests
# =============================================================================


def test_parse_tool_calls_with_calls(
    sample_assistant_message_with_tool_calls: dict[str, Any],
) -> None:
    """Verify parse_tool_calls extracts tool_calls."""
    tool_calls = parse_tool_calls(sample_assistant_message_with_tool_calls)
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_123"


def test_parse_tool_calls_no_calls(sample_assistant_message: dict[str, Any]) -> None:
    """Verify parse_tool_calls returns empty list when no tool_calls."""
    tool_calls = parse_tool_calls(sample_assistant_message)
    assert tool_calls == []


def test_parse_tool_calls_none_value() -> None:
    """Verify parse_tool_calls handles None tool_calls."""
    message = {"role": "assistant", "content": "Hi", "tool_calls": None}
    assert parse_tool_calls(message) == []


def test_parse_tool_calls_missing_key() -> None:
    """Verify parse_tool_calls handles missing tool_calls key."""
    message = {"role": "assistant", "content": "Hi"}
    assert parse_tool_calls(message) == []


def test_parse_tool_calls_not_a_list() -> None:
    """Verify parse_tool_calls handles non-list tool_calls."""
    message = {"role": "assistant", "tool_calls": "not a list"}
    assert parse_tool_calls(message) == []


def test_parse_tool_calls_empty_list() -> None:
    """Verify parse_tool_calls handles empty tool_calls list."""
    message = {"role": "assistant", "tool_calls": []}
    assert parse_tool_calls(message) == []


# =============================================================================
# normalize_stop Tests
# =============================================================================


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (None, None),
        ("stop", ["stop"]),
        (["a", "b"], ["a", "b"]),
        ([1, 2, 3], ["1", "2", "3"]),
        ("", [""]),
        ([], []),
    ],
)
def test_normalize_stop(input_val: Any, expected: Any) -> None:
    """Verify normalize_stop normalizes various input formats."""
    assert normalize_stop(input_val) == expected


def test_normalize_stop_invalid_type() -> None:
    """Verify normalize_stop returns None for invalid types."""
    assert normalize_stop(123) is None
    assert normalize_stop({"a": 1}) is None


# =============================================================================
# get_message_content Tests
# =============================================================================


def test_get_message_content_present() -> None:
    """Verify get_message_content extracts content."""
    message = {"role": "user", "content": "Hello"}
    assert get_message_content(message) == "Hello"


def test_get_message_content_none() -> None:
    """Verify get_message_content returns empty string for None."""
    message = {"role": "assistant", "content": None}
    assert get_message_content(message) == ""


def test_get_message_content_missing() -> None:
    """Verify get_message_content returns empty string if missing."""
    message = {"role": "assistant"}
    assert get_message_content(message) == ""


def test_get_message_content_empty() -> None:
    """Verify get_message_content returns empty string for empty content."""
    message = {"role": "user", "content": ""}
    assert get_message_content(message) == ""


def test_get_message_content_not_string() -> None:
    """Verify get_message_content returns empty for non-string content."""
    message = {"role": "user", "content": 123}
    assert get_message_content(message) == ""


# =============================================================================
# get_message_role Tests
# =============================================================================


@pytest.mark.parametrize(
    "message,expected",
    [
        ({"role": "user", "content": "Hi"}, "user"),
        ({"role": "assistant", "content": "Hello"}, "assistant"),
        ({"role": "tool", "content": "42", "tool_call_id": "call_123"}, "tool"),
        ({"content": "Hi"}, "unknown"),
        ({"role": 123}, "unknown"),
    ],
)
def test_get_message_role(message: dict[str, Any], expected: str) -> None:
    """Verify get_message_role extracts role or returns 'unknown'."""
    assert get_message_role(message) == expected


# =============================================================================
# is_* Role Check Tests
# =============================================================================


@pytest.mark.parametrize(
    "func,role,expected",
    [
        (is_assistant_message, "assistant", True),
        (is_assistant_message, "user", False),
        (is_tool_message, "tool", True),
        (is_tool_message, "user", False),
        (is_user_message, "user", True),
        (is_user_message, "assistant", False),
    ],
)
def test_is_role_message(func: Any, role: str, expected: bool) -> None:
    """Verify is_*_message functions return correct bool for given role."""
    msg: dict[str, Any] = {"role": role, "content": "test"}
    if role == "tool":
        msg["tool_call_id"] = "call_123"
    assert func(msg) == expected


# =============================================================================
# count_messages_by_role Tests
# =============================================================================


def test_count_messages_by_role_empty() -> None:
    """Verify count_messages_by_role handles empty list."""
    assert count_messages_by_role([]) == {}


def test_count_messages_by_role_single() -> None:
    """Verify count_messages_by_role counts single message."""
    messages = [{"role": "user", "content": "Hi"}]
    assert count_messages_by_role(messages) == {"user": 1}


def test_count_messages_by_role_multiple() -> None:
    """Verify count_messages_by_role counts multiple messages."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine"},
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]
    counts = count_messages_by_role(messages)
    assert counts == {"system": 1, "user": 2, "assistant": 2, "tool": 1}


def test_count_messages_by_role_unknown() -> None:
    """Verify count_messages_by_role handles missing roles."""
    messages = [{"content": "No role"}]
    counts = count_messages_by_role(messages)
    assert counts == {"unknown": 1}

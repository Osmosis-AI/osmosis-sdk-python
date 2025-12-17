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

"""Tests for osmosis_ai.rollout.tools."""

from __future__ import annotations

import pytest

from osmosis_ai.rollout import (
    ToolArgumentError,
    ToolExecutionError,
    create_tool_error_result,
    create_tool_result,
    execute_tool_calls,
    get_tool_call_info,
    parse_tool_arguments,
    serialize_tool_result,
)


# =============================================================================
# create_tool_result Tests
# =============================================================================


def test_create_tool_result_basic() -> None:
    """Verify create_tool_result creates correct structure."""
    result = create_tool_result("call_123", "42")
    assert result == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "call_123",
    }


def test_create_tool_result_empty_content() -> None:
    """Verify create_tool_result handles empty content."""
    result = create_tool_result("call_123", "")
    assert result["content"] == ""
    assert result["tool_call_id"] == "call_123"


def test_create_tool_result_multiline_content() -> None:
    """Verify create_tool_result handles multiline content."""
    content = "Line 1\nLine 2\nLine 3"
    result = create_tool_result("call_123", content)
    assert result["content"] == content


# =============================================================================
# serialize_tool_result Tests
# =============================================================================


def test_serialize_tool_result_int() -> None:
    """Verify serialize_tool_result handles integers."""
    assert serialize_tool_result(42) == "42"
    assert serialize_tool_result(0) == "0"
    assert serialize_tool_result(-1) == "-1"


def test_serialize_tool_result_float() -> None:
    """Verify serialize_tool_result handles floats."""
    assert serialize_tool_result(3.14) == "3.14"
    assert serialize_tool_result(0.0) == "0.0"
    assert serialize_tool_result(-1.5) == "-1.5"


def test_serialize_tool_result_string() -> None:
    """Verify serialize_tool_result passes strings through."""
    assert serialize_tool_result("hello") == "hello"
    assert serialize_tool_result("") == ""
    assert serialize_tool_result("  spaces  ") == "  spaces  "


def test_serialize_tool_result_dict() -> None:
    """Verify serialize_tool_result JSON-encodes dicts."""
    result = serialize_tool_result({"a": 1, "b": 2})
    assert result == '{"a": 1, "b": 2}'


def test_serialize_tool_result_list() -> None:
    """Verify serialize_tool_result JSON-encodes lists."""
    result = serialize_tool_result([1, 2, 3])
    assert result == "[1, 2, 3]"


def test_serialize_tool_result_none() -> None:
    """Verify serialize_tool_result handles None."""
    assert serialize_tool_result(None) == "null"


def test_serialize_tool_result_bool() -> None:
    """Verify serialize_tool_result JSON-encodes booleans."""
    assert serialize_tool_result(True) == "true"
    assert serialize_tool_result(False) == "false"


# =============================================================================
# parse_tool_arguments Tests
# =============================================================================


def test_parse_tool_arguments_dict() -> None:
    """Verify parse_tool_arguments passes dicts through."""
    args = {"a": 5, "b": 3}
    result = parse_tool_arguments(args)
    assert result == {"a": 5, "b": 3}


def test_parse_tool_arguments_json_string() -> None:
    """Verify parse_tool_arguments parses JSON strings."""
    result = parse_tool_arguments('{"a": 5, "b": 3}')
    assert result == {"a": 5, "b": 3}


def test_parse_tool_arguments_empty_dict_json() -> None:
    """Verify parse_tool_arguments handles empty JSON object."""
    result = parse_tool_arguments("{}")
    assert result == {}


def test_parse_tool_arguments_nested_json() -> None:
    """Verify parse_tool_arguments handles nested JSON."""
    json_str = '{"nested": {"a": 1}, "list": [1, 2, 3]}'
    result = parse_tool_arguments(json_str)
    assert result == {"nested": {"a": 1}, "list": [1, 2, 3]}


def test_parse_tool_arguments_invalid_json_raises() -> None:
    """Verify parse_tool_arguments raises on invalid JSON."""
    with pytest.raises(ToolArgumentError, match="Invalid JSON"):
        parse_tool_arguments("not valid json")


def test_parse_tool_arguments_json_array_raises() -> None:
    """Verify parse_tool_arguments raises if JSON is not a dict."""
    with pytest.raises(ToolArgumentError, match="Expected dict"):
        parse_tool_arguments("[1, 2, 3]")


def test_parse_tool_arguments_wrong_type_raises() -> None:
    """Verify parse_tool_arguments raises on wrong type."""
    with pytest.raises(ToolArgumentError, match="Expected str or dict"):
        parse_tool_arguments(123)  # type: ignore


# =============================================================================
# get_tool_call_info Tests
# =============================================================================


def test_get_tool_call_info_basic() -> None:
    """Verify get_tool_call_info extracts info correctly."""
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": '{"a": 5, "b": 3}',
        },
    }
    call_id, name, args = get_tool_call_info(tool_call)
    assert call_id == "call_123"
    assert name == "add"
    assert args == {"a": 5, "b": 3}


def test_get_tool_call_info_dict_arguments() -> None:
    """Verify get_tool_call_info handles dict arguments."""
    tool_call = {
        "id": "call_456",
        "type": "function",
        "function": {
            "name": "multiply",
            "arguments": {"x": 2, "y": 3},
        },
    }
    call_id, name, args = get_tool_call_info(tool_call)
    assert call_id == "call_456"
    assert name == "multiply"
    assert args == {"x": 2, "y": 3}


def test_get_tool_call_info_missing_id() -> None:
    """Verify get_tool_call_info uses 'unknown' for missing id."""
    tool_call = {
        "type": "function",
        "function": {
            "name": "test",
            "arguments": "{}",
        },
    }
    call_id, name, args = get_tool_call_info(tool_call)
    assert call_id == "unknown"


def test_get_tool_call_info_invalid_arguments() -> None:
    """Verify get_tool_call_info raises on invalid arguments."""
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "test",
            "arguments": "not valid json",
        },
    }
    with pytest.raises(ToolArgumentError, match="Invalid JSON"):
        get_tool_call_info(tool_call)


# =============================================================================
# execute_tool_calls Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execute_tool_calls_empty() -> None:
    """Verify execute_tool_calls handles empty list."""

    async def executor(tc):
        return create_tool_result(tc["id"], "result")

    results = await execute_tool_calls([], executor)
    assert results == []


@pytest.mark.asyncio
async def test_execute_tool_calls_single() -> None:
    """Verify execute_tool_calls handles single call."""

    async def executor(tc):
        return create_tool_result(tc["id"], "42")

    tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
    results = await execute_tool_calls(tool_calls, executor)

    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call_1"
    assert results[0]["content"] == "42"


@pytest.mark.asyncio
async def test_execute_tool_calls_multiple() -> None:
    """Verify execute_tool_calls handles multiple calls in order."""

    async def executor(tc):
        return create_tool_result(tc["id"], f"result_{tc['id']}")

    tool_calls = [
        {"id": "call_1", "function": {"name": "a", "arguments": "{}"}},
        {"id": "call_2", "function": {"name": "b", "arguments": "{}"}},
        {"id": "call_3", "function": {"name": "c", "arguments": "{}"}},
    ]
    results = await execute_tool_calls(tool_calls, executor)

    assert len(results) == 3
    assert results[0]["content"] == "result_call_1"
    assert results[1]["content"] == "result_call_2"
    assert results[2]["content"] == "result_call_3"


@pytest.mark.asyncio
async def test_execute_tool_calls_concurrent() -> None:
    """Verify execute_tool_calls runs concurrently."""
    import asyncio

    call_times = []

    async def executor(tc):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.1)
        return create_tool_result(tc["id"], "done")

    tool_calls = [
        {"id": f"call_{i}", "function": {"name": "test", "arguments": "{}"}}
        for i in range(3)
    ]

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(tool_calls, executor)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 3
    # Should complete in ~0.1s if concurrent, not ~0.3s if sequential
    assert elapsed < 0.2


# =============================================================================
# create_tool_error_result Tests
# =============================================================================


def test_create_tool_error_result_basic() -> None:
    """Verify create_tool_error_result creates error message."""
    result = create_tool_error_result("call_123", "Division by zero")
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "Error: Division by zero"


# =============================================================================
# Exception Tests
# =============================================================================


def test_tool_execution_error_basic() -> None:
    """Verify ToolExecutionError basic usage."""
    error = ToolExecutionError("Something went wrong")
    assert str(error) == "Something went wrong"


def test_tool_execution_error_with_info() -> None:
    """Verify ToolExecutionError with tool info."""
    error = ToolExecutionError(
        "Invalid arguments",
        tool_call_id="call_123",
        tool_name="add",
    )
    assert error.tool_call_id == "call_123"
    assert error.tool_name == "add"


def test_tool_argument_error_is_execution_error() -> None:
    """Verify ToolArgumentError inherits from ToolExecutionError."""
    error = ToolArgumentError("Bad JSON")
    assert isinstance(error, ToolExecutionError)

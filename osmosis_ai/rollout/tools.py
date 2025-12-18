"""Tool execution utilities for Osmosis remote rollout SDK.

This module provides helper functions for executing tools and creating
standardized tool result messages.

Example:
    from osmosis_ai.rollout.tools import (
        create_tool_result,
        serialize_tool_result,
        parse_tool_arguments,
        execute_tool_calls,
    )

    # Create a tool result message
    result = create_tool_result("call_123", "42")

    # Parse arguments (supports both str and dict)
    args = parse_tool_arguments('{"a": 5, "b": 3}')

    # Execute multiple tool calls in parallel
    async def my_executor(tool_call):
        name = tool_call["function"]["name"]
        args = parse_tool_arguments(tool_call["function"]["arguments"])
        result = await my_tools[name](**args)
        return create_tool_result(tool_call["id"], serialize_tool_result(result))

    results = await execute_tool_calls(tool_calls, my_executor)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Union

from osmosis_ai.rollout.core.exceptions import ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)


def create_tool_result(tool_call_id: str, content: str) -> Dict[str, str]:
    """Create a standardized tool result message.

    Creates a message dict with role="tool" that can be appended to the
    conversation history.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        content: The string content of the tool result.

    Returns:
        Dict with role, content, and tool_call_id keys.

    Example:
        result = create_tool_result("call_123", "42")
        # {"role": "tool", "content": "42", "tool_call_id": "call_123"}
    """
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }


def serialize_tool_result(result: Any) -> str:
    """Serialize a tool result to string format.

    Handles different types appropriately:
    - int/float (not bool): Simple string representation (preserves precision)
    - str: Returned as-is
    - Other (including bool, dict, list, None): JSON serialized

    Args:
        result: The result value to serialize.

    Returns:
        String representation of the result.

    Example:
        serialize_tool_result(42)      # "42"
        serialize_tool_result(3.14)    # "3.14"
        serialize_tool_result("hello") # "hello"
        serialize_tool_result({"a": 1}) # '{"a": 1}'
        serialize_tool_result(True)    # "true"
    """
    # Note: bool is a subclass of int in Python, so check bool first
    if isinstance(result, bool):
        return json.dumps(result)
    elif isinstance(result, (int, float)):
        return str(result)
    elif isinstance(result, str):
        return result
    else:
        return json.dumps(result)


def parse_tool_arguments(arguments: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse tool call arguments, handling both string and dict formats.

    LLM responses may provide arguments as either a JSON string or a dict.
    This function normalizes both formats to a dict.

    Args:
        arguments: Tool arguments as JSON string or dict.

    Returns:
        Parsed arguments as a dict.

    Raises:
        ToolArgumentError: If JSON parsing fails.

    Example:
        # Both return {"a": 5, "b": 3}
        parse_tool_arguments('{"a": 5, "b": 3}')
        parse_tool_arguments({"a": 5, "b": 3})
    """
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            raise ToolArgumentError(
                f"Expected dict from JSON, got {type(parsed).__name__}"
            )
        except json.JSONDecodeError as e:
            raise ToolArgumentError(f"Invalid JSON arguments: {e}")

    raise ToolArgumentError(
        f"Expected str or dict arguments, got {type(arguments).__name__}"
    )


def get_tool_call_info(tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    """Extract tool call ID, name, and arguments from a tool call dict.

    Args:
        tool_call: Tool call dict with id, function.name, and function.arguments.

    Returns:
        Tuple of (tool_call_id, function_name, parsed_arguments).

    Raises:
        ToolArgumentError: If the tool call is malformed or arguments are invalid.

    Example:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'}
        }
        call_id, name, args = get_tool_call_info(tool_call)
        # ("call_123", "add", {"a": 5, "b": 3})
    """
    tool_call_id = tool_call.get("id", "unknown")
    function_data = tool_call.get("function", {})
    function_name = function_data.get("name", "")
    raw_arguments = function_data.get("arguments", {})

    try:
        arguments = parse_tool_arguments(raw_arguments)
    except ToolArgumentError as e:
        raise ToolArgumentError(
            str(e),
            tool_call_id=tool_call_id,
            tool_name=function_name,
        )

    return tool_call_id, function_name, arguments


async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Execute multiple tool calls concurrently.

    Runs all tool calls in parallel using asyncio.gather. The executor
    function should return a tool result message dict.

    Args:
        tool_calls: List of tool call dicts from the LLM response.
        executor: Async function that takes a tool call and returns a tool result.

    Returns:
        List of tool result message dicts in the same order as input.

    Example:
        async def my_executor(tool_call: Dict) -> Dict[str, str]:
            call_id, name, args = get_tool_call_info(tool_call)
            result = await my_tools[name](**args)
            return create_tool_result(call_id, serialize_tool_result(result))

        results = await execute_tool_calls(tool_calls, my_executor)
        messages.extend(results)
    """
    if not tool_calls:
        return []

    tasks = [executor(tool_call) for tool_call in tool_calls]
    return await asyncio.gather(*tasks)


def create_tool_error_result(
    tool_call_id: str,
    error_message: str,
) -> Dict[str, str]:
    """Create an error result message for a failed tool call.

    Use this to return error information to the LLM so it can potentially
    recover or provide a helpful message to the user.

    Args:
        tool_call_id: The ID of the tool call that failed.
        error_message: Description of the error.

    Returns:
        Tool result message with error content.

    Example:
        try:
            result = await execute_tool(tool_call)
        except Exception as e:
            result = create_tool_error_result(tool_call["id"], str(e))
    """
    return create_tool_result(tool_call_id, f"Error: {error_message}")


__all__ = [
    "create_tool_result",
    "serialize_tool_result",
    "parse_tool_arguments",
    "get_tool_call_info",
    "execute_tool_calls",
    "create_tool_error_result",
]

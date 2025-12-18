"""Message and request utilities for Osmosis remote rollout SDK.

This module provides helper functions for working with messages and
request parameters.

Example:
    from osmosis_ai.rollout.utils import parse_tool_calls, normalize_stop

    # Safely extract tool calls from assistant message
    tool_calls = parse_tool_calls(assistant_message)

    # Normalize stop parameter
    stop = normalize_stop(params.get("stop"))  # Always List[str] or None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def parse_tool_calls(assistant_message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Safely extract tool_calls from an assistant message.

    Handles edge cases where tool_calls may be None, missing, or not a list.

    Args:
        assistant_message: The assistant message dict.

    Returns:
        List of tool call dicts, or empty list if none.

    Example:
        tool_calls = parse_tool_calls(result.message)
        if tool_calls:
            for tool_call in tool_calls:
                # Process tool call
    """
    tool_calls = assistant_message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def normalize_stop(stop: Any) -> Optional[List[str]]:
    """Normalize stop parameter to consistent format.

    Handles various input formats and normalizes to List[str] or None.

    Args:
        stop: Stop parameter in various formats (None, str, List).

    Returns:
        List of stop sequences, or None.

    Example:
        normalize_stop(None)        # None
        normalize_stop("stop")      # ["stop"]
        normalize_stop(["a", "b"])  # ["a", "b"]
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(s) for s in stop]
    return None


def get_message_content(message: Dict[str, Any]) -> str:
    """Extract text content from a message.

    Handles messages where content may be None or missing.

    Args:
        message: A message dict.

    Returns:
        The content string, or empty string if not present.

    Example:
        content = get_message_content(message)
    """
    content = message.get("content")
    return content if isinstance(content, str) else ""


def get_message_role(message: Dict[str, Any]) -> str:
    """Extract role from a message.

    Args:
        message: A message dict.

    Returns:
        The role string, or "unknown" if not present.

    Example:
        role = get_message_role(message)
    """
    role = message.get("role")
    return role if isinstance(role, str) else "unknown"


def is_assistant_message(message: Dict[str, Any]) -> bool:
    """Check if a message is from the assistant.

    Args:
        message: A message dict.

    Returns:
        True if the message role is "assistant".
    """
    return get_message_role(message) == "assistant"


def is_tool_message(message: Dict[str, Any]) -> bool:
    """Check if a message is a tool result.

    Args:
        message: A message dict.

    Returns:
        True if the message role is "tool".
    """
    return get_message_role(message) == "tool"


def is_user_message(message: Dict[str, Any]) -> bool:
    """Check if a message is from the user.

    Args:
        message: A message dict.

    Returns:
        True if the message role is "user".
    """
    return get_message_role(message) == "user"


def count_messages_by_role(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count messages by role.

    Args:
        messages: List of message dicts.

    Returns:
        Dict mapping role to count.

    Example:
        counts = count_messages_by_role(messages)
        # {"user": 3, "assistant": 2, "tool": 1}
    """
    counts: Dict[str, int] = {}
    for message in messages:
        role = get_message_role(message)
        counts[role] = counts.get(role, 0) + 1
    return counts


__all__ = [
    "parse_tool_calls",
    "normalize_stop",
    "get_message_content",
    "get_message_role",
    "is_assistant_message",
    "is_tool_message",
    "is_user_message",
    "count_messages_by_role",
]

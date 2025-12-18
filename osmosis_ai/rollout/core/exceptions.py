"""Exceptions for Osmosis remote rollout protocol.

This module defines a hierarchy of exceptions for handling errors
during rollout execution, LLM communication, and tool execution.

Exception Hierarchy:
    OsmosisRolloutError (base)
    ├── OsmosisTransportError (network errors)
    ├── OsmosisServerError (5xx, retryable)
    ├── OsmosisValidationError (4xx, not retryable)
    ├── OsmosisTimeoutError (timeouts)
    └── AgentLoopNotFoundError (registry lookup failed)

Example:
    try:
        result = await client.chat_completions(messages)
    except OsmosisServerError as e:
        if e.status_code >= 500:
            # Retry the request
            pass
    except OsmosisValidationError as e:
        # Don't retry, request is invalid
        raise
    except OsmosisRolloutError as e:
        # Catch all other rollout errors
        logger.error(f"Rollout failed: {e}")
"""

from __future__ import annotations

from typing import List


class OsmosisRolloutError(Exception):
    """Base exception for all Osmosis rollout errors.

    All rollout-related exceptions inherit from this class,
    allowing you to catch all rollout errors with a single except clause.
    """

    pass


class OsmosisTransportError(OsmosisRolloutError):
    """Network/transport level error.

    Raised when the HTTP request fails due to network issues,
    connection refused, DNS resolution failure, etc.

    These errors may be transient and retrying could succeed.
    """

    pass


class OsmosisServerError(OsmosisRolloutError):
    """Server returned 5xx error (retryable).

    Indicates a server-side error that may be transient.
    Retrying the request with exponential backoff is recommended.

    Attributes:
        status_code: HTTP status code from the server (500-599).
    """

    def __init__(self, message: str, status_code: int):
        """Initialize with message and status code.

        Args:
            message: Error description.
            status_code: HTTP status code (should be 5xx).
        """
        super().__init__(message)
        self.status_code = status_code


class OsmosisValidationError(OsmosisRolloutError):
    """Server returned 4xx error (not retryable).

    Indicates a client-side error such as invalid request format,
    authentication failure, or resource not found.

    Retrying will not help; the request must be corrected.

    Attributes:
        status_code: HTTP status code from the server (400-499).
    """

    def __init__(self, message: str, status_code: int):
        """Initialize with message and status code.

        Args:
            message: Error description.
            status_code: HTTP status code (should be 4xx).
        """
        super().__init__(message)
        self.status_code = status_code


class OsmosisTimeoutError(OsmosisRolloutError):
    """Operation timed out.

    Raised when an operation exceeds its configured timeout.
    This could be due to slow network, overloaded server,
    or long-running LLM generation.

    Consider increasing timeout or implementing retry logic.
    """

    pass


class AgentLoopNotFoundError(OsmosisRolloutError):
    """Agent loop not found in registry.

    Raised when attempting to retrieve an agent loop by name
    but no loop with that name has been registered.

    Attributes:
        name: The name that was looked up.
        available: List of registered agent loop names.
    """

    def __init__(self, name: str, available: List[str]):
        """Initialize with lookup name and available names.

        Args:
            name: The agent loop name that was not found.
            available: List of currently registered names.
        """
        self.name = name
        self.available = available
        super().__init__(
            f"Agent loop '{name}' not found. Available: {available}"
        )


class ToolExecutionError(OsmosisRolloutError):
    """Error during tool execution.

    Attributes:
        tool_call_id: The ID of the failed tool call.
        tool_name: The name of the tool that failed.
    """

    def __init__(
        self,
        message: str,
        tool_call_id: str = "",
        tool_name: str = "",
    ):
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


class ToolArgumentError(ToolExecutionError):
    """Error parsing tool arguments."""

    pass

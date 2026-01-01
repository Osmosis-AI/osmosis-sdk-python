"""Validation utilities for RolloutAgentLoop implementations.

This module provides validation functions to check if a RolloutAgentLoop
implementation conforms to the Osmosis remote rollout protocol requirements.

Example:
    from osmosis_ai.rollout import RolloutAgentLoop
    from osmosis_ai.rollout.validator import validate_agent_loop

    class MyAgent(RolloutAgentLoop):
        name = "my_agent"
        # ...

    # Validate before serving
    errors = validate_agent_loop(MyAgent())
    if errors:
        for error in errors:
            print(f"Validation error: {error}")
    else:
        print("Agent loop is valid!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionToolSchema,
    RolloutRequest,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation error.

    Attributes:
        code: Error code for programmatic handling.
        message: Human-readable error message.
        field: Optional field name that caused the error.
        details: Optional additional details.
    """

    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.field:
            return f"[{self.code}] {self.field}: {self.message}"
        return f"[{self.code}] {self.message}"


@dataclass
class ValidationResult:
    """Result of agent loop validation.

    Attributes:
        valid: True if validation passed with no errors.
        errors: List of validation errors found.
        warnings: List of validation warnings (non-fatal issues).
        agent_name: Name of the validated agent loop.
        tool_count: Number of tools returned by get_tools().
    """

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    agent_name: Optional[str] = None
    tool_count: int = 0

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.valid

    def raise_if_invalid(self) -> None:
        """Raise AgentLoopValidationError if validation failed.

        Raises:
            AgentLoopValidationError: If there are validation errors.
        """
        if not self.valid:
            error_messages = [str(e) for e in self.errors]
            raise AgentLoopValidationError(
                f"Agent loop validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(f"  - {msg}" for msg in error_messages),
                errors=self.errors,
            )


class AgentLoopValidationError(Exception):
    """Raised when agent loop validation fails.

    Attributes:
        errors: List of validation errors that caused the failure.
    """

    def __init__(self, message: str, errors: Optional[List[ValidationError]] = None):
        super().__init__(message)
        self.errors = errors or []


def _create_mock_request() -> RolloutRequest:
    """Create a mock RolloutRequest for validation purposes."""
    return RolloutRequest(
        rollout_id="validation-test-rollout",
        server_url="http://localhost:8080",
        messages=[{"role": "user", "content": "Hello"}],
        completion_params={"temperature": 1.0, "max_tokens": 512},
        metadata={"_validation": True},
    )


def _validate_name(agent_loop: RolloutAgentLoop) -> List[ValidationError]:
    """Validate agent loop name attribute."""
    errors: List[ValidationError] = []

    name = getattr(agent_loop, "name", None)
    if name is None:
        errors.append(
            ValidationError(
                code="MISSING_NAME",
                message="Agent loop must have a 'name' attribute",
                field="name",
            )
        )
    elif not isinstance(name, str):
        errors.append(
            ValidationError(
                code="INVALID_NAME_TYPE",
                message=f"'name' must be a string, got {type(name).__name__}",
                field="name",
            )
        )
    elif not name.strip():
        errors.append(
            ValidationError(
                code="EMPTY_NAME",
                message="'name' cannot be empty or whitespace only",
                field="name",
            )
        )

    return errors


def _validate_tool_schema(
    tool: Any, index: int
) -> tuple[List[ValidationError], List[ValidationError]]:
    """Validate a single tool schema.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    field_prefix = f"tools[{index}]"

    # Check if it's a valid OpenAIFunctionToolSchema or dict
    if isinstance(tool, OpenAIFunctionToolSchema):
        # Already validated by Pydantic
        return errors, warnings

    if isinstance(tool, dict):
        # Validate dict structure
        if "type" not in tool:
            errors.append(
                ValidationError(
                    code="MISSING_TOOL_TYPE",
                    message="Tool must have a 'type' field",
                    field=field_prefix,
                )
            )
        elif tool["type"] != "function":
            warnings.append(
                ValidationError(
                    code="UNKNOWN_TOOL_TYPE",
                    message=f"Tool type '{tool['type']}' may not be supported, expected 'function'",
                    field=f"{field_prefix}.type",
                )
            )

        if "function" not in tool:
            errors.append(
                ValidationError(
                    code="MISSING_FUNCTION",
                    message="Tool must have a 'function' field",
                    field=field_prefix,
                )
            )
        else:
            func = tool["function"]
            if not isinstance(func, dict):
                errors.append(
                    ValidationError(
                        code="INVALID_FUNCTION_TYPE",
                        message="'function' must be a dict",
                        field=f"{field_prefix}.function",
                    )
                )
            else:
                if "name" not in func:
                    errors.append(
                        ValidationError(
                            code="MISSING_FUNCTION_NAME",
                            message="Function must have a 'name' field",
                            field=f"{field_prefix}.function.name",
                        )
                    )
                elif not isinstance(func["name"], str) or not func["name"].strip():
                    errors.append(
                        ValidationError(
                            code="INVALID_FUNCTION_NAME",
                            message="Function name must be a non-empty string",
                            field=f"{field_prefix}.function.name",
                        )
                    )

                if "description" not in func:
                    warnings.append(
                        ValidationError(
                            code="MISSING_FUNCTION_DESCRIPTION",
                            message="Function should have a 'description' for better LLM understanding",
                            field=f"{field_prefix}.function.description",
                        )
                    )

                if "parameters" in func:
                    params = func["parameters"]
                    if not isinstance(params, dict):
                        errors.append(
                            ValidationError(
                                code="INVALID_PARAMETERS_TYPE",
                                message="'parameters' must be a dict",
                                field=f"{field_prefix}.function.parameters",
                            )
                        )
                    elif params.get("type") != "object":
                        warnings.append(
                            ValidationError(
                                code="PARAMETERS_TYPE_NOT_OBJECT",
                                message="'parameters.type' should be 'object'",
                                field=f"{field_prefix}.function.parameters.type",
                            )
                        )
    else:
        errors.append(
            ValidationError(
                code="INVALID_TOOL_TYPE",
                message=f"Tool must be OpenAIFunctionToolSchema or dict, got {type(tool).__name__}",
                field=field_prefix,
            )
        )

    return errors, warnings


def _validate_get_tools(
    agent_loop: RolloutAgentLoop, request: RolloutRequest
) -> tuple[List[ValidationError], List[ValidationError], int]:
    """Validate get_tools() method.

    Returns:
        Tuple of (errors, warnings, tool_count).
    """
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    tool_count = 0

    try:
        tools = agent_loop.get_tools(request)
    except Exception as e:
        errors.append(
            ValidationError(
                code="GET_TOOLS_EXCEPTION",
                message=f"get_tools() raised an exception: {type(e).__name__}: {e}",
                field="get_tools",
                details={"exception_type": type(e).__name__, "exception_message": str(e)},
            )
        )
        return errors, warnings, 0

    if tools is None:
        errors.append(
            ValidationError(
                code="GET_TOOLS_RETURNS_NONE",
                message="get_tools() must return a list, got None",
                field="get_tools",
            )
        )
        return errors, warnings, 0

    if not isinstance(tools, list):
        errors.append(
            ValidationError(
                code="GET_TOOLS_INVALID_TYPE",
                message=f"get_tools() must return a list, got {type(tools).__name__}",
                field="get_tools",
            )
        )
        return errors, warnings, 0

    tool_count = len(tools)

    # Validate each tool
    for i, tool in enumerate(tools):
        tool_errors, tool_warnings = _validate_tool_schema(tool, i)
        errors.extend(tool_errors)
        warnings.extend(tool_warnings)

    return errors, warnings, tool_count


def _validate_run_method(agent_loop: RolloutAgentLoop) -> List[ValidationError]:
    """Validate that run() method exists and is async."""
    errors: List[ValidationError] = []

    run_method = getattr(agent_loop, "run", None)
    if run_method is None:
        errors.append(
            ValidationError(
                code="MISSING_RUN_METHOD",
                message="Agent loop must implement 'run' method",
                field="run",
            )
        )
    elif not callable(run_method):
        errors.append(
            ValidationError(
                code="RUN_NOT_CALLABLE",
                message="'run' must be a callable method",
                field="run",
            )
        )
    else:
        import asyncio

        if not asyncio.iscoroutinefunction(run_method):
            errors.append(
                ValidationError(
                    code="RUN_NOT_ASYNC",
                    message="'run' method must be an async function (async def)",
                    field="run",
                )
            )

    return errors


def validate_agent_loop(
    agent_loop: RolloutAgentLoop,
    *,
    request: Optional[RolloutRequest] = None,
) -> ValidationResult:
    """Validate a RolloutAgentLoop implementation.

    This function checks that the agent loop:
    1. Has a valid 'name' attribute
    2. Implements get_tools() correctly
    3. Returns valid OpenAI-compatible tool schemas
    4. Has a valid async run() method

    Args:
        agent_loop: The RolloutAgentLoop instance to validate.
        request: Optional custom RolloutRequest for testing get_tools().
                 If not provided, a mock request is used.

    Returns:
        ValidationResult with validation status and any errors/warnings.

    Example:
        from osmosis_ai.rollout.validator import validate_agent_loop

        result = validate_agent_loop(MyAgentLoop())
        if not result.valid:
            for error in result.errors:
                print(f"Error: {error}")
        else:
            print(f"Agent '{result.agent_name}' is valid with {result.tool_count} tools")

        # Or raise exception if invalid
        result.raise_if_invalid()
    """
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []

    # Validate name
    name_errors = _validate_name(agent_loop)
    errors.extend(name_errors)

    agent_name = getattr(agent_loop, "name", None)

    # Validate get_tools
    if request is None:
        request = _create_mock_request()

    tools_errors, tools_warnings, tool_count = _validate_get_tools(agent_loop, request)
    errors.extend(tools_errors)
    warnings.extend(tools_warnings)

    # Validate run method
    run_errors = _validate_run_method(agent_loop)
    errors.extend(run_errors)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        agent_name=agent_name if isinstance(agent_name, str) else None,
        tool_count=tool_count,
    )


__all__ = [
    "ValidationError",
    "ValidationResult",
    "AgentLoopValidationError",
    "validate_agent_loop",
]

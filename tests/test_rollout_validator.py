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

"""Tests for osmosis_ai.rollout.validator."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
)
from osmosis_ai.rollout.validator import (
    AgentLoopValidationError,
    ValidationError,
    ValidationResult,
    validate_agent_loop,
)


# =============================================================================
# Test Agent Loop Implementations
# =============================================================================


class ValidAgentLoop(RolloutAgentLoop):
    """A valid agent loop for testing."""

    name = "valid_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return [
            OpenAIFunctionToolSchema(
                type="function",
                function=OpenAIFunctionSchema(
                    name="add",
                    description="Add two numbers",
                    parameters=OpenAIFunctionParametersSchema(
                        type="object",
                        properties={
                            "a": OpenAIFunctionPropertySchema(type="number"),
                            "b": OpenAIFunctionPropertySchema(type="number"),
                        },
                        required=["a", "b"],
                    ),
                ),
            )
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete(list(ctx.request.messages))


class NoToolsAgentLoop(RolloutAgentLoop):
    """An agent loop with no tools."""

    name = "no_tools"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete(list(ctx.request.messages))


class DictToolsAgentLoop(RolloutAgentLoop):
    """An agent loop that returns dict tools instead of Pydantic models."""

    name = "dict_tools"

    def get_tools(self, request: RolloutRequest) -> List[Any]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["x", "y"],
                    },
                },
            }
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete(list(ctx.request.messages))


class InvalidToolsAgentLoop(RolloutAgentLoop):
    """An agent loop with invalid tool format."""

    name = "invalid_tools"

    def get_tools(self, request: RolloutRequest) -> List[Any]:
        return [
            {"invalid": "tool"},  # Missing type and function
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


class GetToolsRaisesAgentLoop(RolloutAgentLoop):
    """An agent loop where get_tools raises an exception."""

    name = "get_tools_raises"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        raise ValueError("Test exception in get_tools")

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


class GetToolsReturnsNoneAgentLoop(RolloutAgentLoop):
    """An agent loop where get_tools returns None."""

    name = "get_tools_none"

    def get_tools(self, request: RolloutRequest) -> Any:
        return None

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


class MissingFunctionNameAgentLoop(RolloutAgentLoop):
    """An agent loop with tool missing function name."""

    name = "missing_function_name"

    def get_tools(self, request: RolloutRequest) -> List[Any]:
        return [
            {
                "type": "function",
                "function": {
                    "description": "A tool without a name",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


class MissingDescriptionAgentLoop(RolloutAgentLoop):
    """An agent loop with tool missing description (warning only)."""

    name = "missing_description"

    def get_tools(self, request: RolloutRequest) -> List[Any]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


# =============================================================================
# ValidationResult Tests
# =============================================================================


def test_validation_result_valid() -> None:
    """Verify valid ValidationResult."""
    result = ValidationResult(valid=True, agent_name="test", tool_count=2)
    assert result.valid
    assert bool(result) is True
    assert result.agent_name == "test"
    assert result.tool_count == 2
    assert len(result.errors) == 0


def test_validation_result_invalid() -> None:
    """Verify invalid ValidationResult."""
    errors = [ValidationError(code="TEST", message="Test error")]
    result = ValidationResult(valid=False, errors=errors)
    assert not result.valid
    assert bool(result) is False
    assert len(result.errors) == 1


def test_validation_result_raise_if_invalid() -> None:
    """Verify raise_if_invalid raises on invalid result."""
    errors = [ValidationError(code="TEST", message="Test error")]
    result = ValidationResult(valid=False, errors=errors)

    with pytest.raises(AgentLoopValidationError) as exc_info:
        result.raise_if_invalid()

    assert "1 error(s)" in str(exc_info.value)
    assert len(exc_info.value.errors) == 1


def test_validation_result_raise_if_invalid_passes_when_valid() -> None:
    """Verify raise_if_invalid does nothing when valid."""
    result = ValidationResult(valid=True)
    result.raise_if_invalid()  # Should not raise


# =============================================================================
# ValidationError Tests
# =============================================================================


def test_validation_error_str_without_field() -> None:
    """Verify ValidationError string representation without field."""
    error = ValidationError(code="TEST", message="Test error")
    assert str(error) == "[TEST] Test error"


def test_validation_error_str_with_field() -> None:
    """Verify ValidationError string representation with field."""
    error = ValidationError(code="TEST", message="Test error", field="test_field")
    assert str(error) == "[TEST] test_field: Test error"


# =============================================================================
# validate_agent_loop Tests - Valid Cases
# =============================================================================


def test_validate_valid_agent_loop() -> None:
    """Verify valid agent loop passes validation."""
    result = validate_agent_loop(ValidAgentLoop())

    assert result.valid
    assert len(result.errors) == 0
    assert result.agent_name == "valid_agent"
    assert result.tool_count == 1


def test_validate_no_tools_agent_loop() -> None:
    """Verify agent loop with no tools passes validation."""
    result = validate_agent_loop(NoToolsAgentLoop())

    assert result.valid
    assert len(result.errors) == 0
    assert result.tool_count == 0


def test_validate_dict_tools_agent_loop() -> None:
    """Verify agent loop with dict tools passes validation."""
    result = validate_agent_loop(DictToolsAgentLoop())

    assert result.valid
    assert len(result.errors) == 0
    assert result.tool_count == 1


# =============================================================================
# validate_agent_loop Tests - Error Cases
# =============================================================================


def test_validate_invalid_tools_format() -> None:
    """Verify invalid tool format is detected."""
    result = validate_agent_loop(InvalidToolsAgentLoop())

    assert not result.valid
    assert len(result.errors) >= 1

    # Should have errors for missing type and function
    error_codes = [e.code for e in result.errors]
    assert "MISSING_TOOL_TYPE" in error_codes or "MISSING_FUNCTION" in error_codes


def test_validate_get_tools_raises() -> None:
    """Verify exception in get_tools is caught and reported."""
    result = validate_agent_loop(GetToolsRaisesAgentLoop())

    assert not result.valid
    assert len(result.errors) == 1
    assert result.errors[0].code == "GET_TOOLS_EXCEPTION"
    assert "Test exception" in result.errors[0].message


def test_validate_get_tools_returns_none() -> None:
    """Verify get_tools returning None is detected."""
    result = validate_agent_loop(GetToolsReturnsNoneAgentLoop())

    assert not result.valid
    assert len(result.errors) == 1
    assert result.errors[0].code == "GET_TOOLS_RETURNS_NONE"


def test_validate_missing_function_name() -> None:
    """Verify missing function name is detected."""
    result = validate_agent_loop(MissingFunctionNameAgentLoop())

    assert not result.valid
    error_codes = [e.code for e in result.errors]
    assert "MISSING_FUNCTION_NAME" in error_codes


# =============================================================================
# validate_agent_loop Tests - Warning Cases
# =============================================================================


def test_validate_missing_description_is_warning() -> None:
    """Verify missing description produces warning, not error."""
    result = validate_agent_loop(MissingDescriptionAgentLoop())

    # Should still be valid (just warnings)
    assert result.valid
    assert len(result.errors) == 0
    assert len(result.warnings) >= 1

    warning_codes = [w.code for w in result.warnings]
    assert "MISSING_FUNCTION_DESCRIPTION" in warning_codes


# =============================================================================
# validate_agent_loop Tests - Custom Request
# =============================================================================


def test_validate_with_custom_request() -> None:
    """Verify validation with custom RolloutRequest."""
    custom_request = RolloutRequest(
        rollout_id="custom-test",
        server_url="http://custom:9000",
        messages=[{"role": "user", "content": "Custom message"}],
        completion_params={"temperature": 0.5},
        metadata={"custom": True},
    )

    result = validate_agent_loop(ValidAgentLoop(), request=custom_request)

    assert result.valid
    assert result.agent_name == "valid_agent"


# =============================================================================
# Integration Tests
# =============================================================================


def test_validate_mock_agent_loop_from_conftest(mock_agent_loop) -> None:
    """Verify validation works with mock agent loop from conftest."""
    result = validate_agent_loop(mock_agent_loop)

    assert result.valid
    assert result.agent_name == "mock_agent"

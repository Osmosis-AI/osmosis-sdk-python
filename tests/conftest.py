"""Shared pytest fixtures for osmosis_ai tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    RolloutAgentLoop,
    RolloutContext,
    RolloutMetrics,
    RolloutRequest,
    RolloutResult,
)


class MockAgentLoop(RolloutAgentLoop):
    """Mock agent loop for testing."""

    name = "mock_agent"

    def __init__(
        self,
        tools: list[OpenAIFunctionToolSchema] | None = None,
        run_result: RolloutResult | None = None,
        run_error: Exception | None = None,
    ):
        self._tools = tools or []
        self._run_result = run_result
        self._run_error = run_error

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return self._tools

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        if self._run_error:
            raise self._run_error
        if self._run_result:
            return self._run_result
        return ctx.complete(list(ctx.request.messages))


@pytest.fixture
def mock_agent_loop() -> MockAgentLoop:
    """Create a mock agent loop for testing."""
    return MockAgentLoop()


@pytest.fixture
def sample_rollout_request() -> RolloutRequest:
    """Create a sample RolloutRequest for testing."""
    return RolloutRequest(
        rollout_id="test-rollout-123",
        server_url="http://localhost:8080",
        messages=[{"role": "user", "content": "Hello"}],
        completion_params={"temperature": 0.7, "max_tokens": 512},
    )


@pytest.fixture
def sample_tool_schema() -> OpenAIFunctionToolSchema:
    """Create a sample OpenAIFunctionToolSchema for testing."""
    return OpenAIFunctionToolSchema(
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


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]


@pytest.fixture
def sample_assistant_message() -> dict[str, Any]:
    """Create a sample assistant message for testing."""
    return {"role": "assistant", "content": "The answer is 4."}


@pytest.fixture
def sample_assistant_message_with_tool_calls() -> dict[str, Any]:
    """Create a sample assistant message with tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": '{"a": 2, "b": 2}',
                },
            }
        ],
    }


@pytest.fixture
def sample_tool_message() -> dict[str, Any]:
    """Create a sample tool result message."""
    return {
        "role": "tool",
        "content": "4",
        "tool_call_id": "call_123",
    }


# =============================================================================
# Shared agent loop stubs (used by multiple test modules)
# =============================================================================


class SimpleAgentLoop(RolloutAgentLoop):
    """Simple agent loop that completes immediately."""

    name = "simple_agent"

    def __init__(self, tools: list[OpenAIFunctionToolSchema] | None = None):
        self._tools = tools or []

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return self._tools

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete(list(ctx.request.messages))


class FailingAgentLoop(RolloutAgentLoop):
    """Agent loop that raises an error."""

    name = "failing_agent"

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        raise RuntimeError("Test error")


class SlowAgentLoop(RolloutAgentLoop):
    """Agent loop that sleeps for a configurable duration before completing."""

    name = "slow_agent"

    def __init__(self, delay: float = 0.5):
        self._delay = delay

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        await asyncio.sleep(self._delay)
        return ctx.complete(list(ctx.request.messages))


# =============================================================================
# Shared helper functions (used by multiple test modules)
# =============================================================================


def make_rollout_payload(
    rollout_id: str = "test-123",
    idempotency_key: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal valid RolloutRequest JSON payload."""
    payload: dict[str, Any] = {
        "rollout_id": rollout_id,
        "server_url": "http://localhost:8080",
        "messages": [{"role": "user", "content": "Hello"}],
        "completion_params": {"temperature": 0.7},
    }
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    payload.update(overrides)
    return payload


def mock_llm_client() -> MagicMock:
    """Create a mocked OsmosisLLMClient usable as an async context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.complete_rollout = AsyncMock()
    mock.get_metrics = MagicMock(return_value=RolloutMetrics())
    return mock

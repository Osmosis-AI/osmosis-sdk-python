"""Shared pytest fixtures for osmosis_ai tests."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
)


class MockAgentLoop(RolloutAgentLoop):
    """Mock agent loop for testing."""

    name = "mock_agent"

    def __init__(
        self,
        tools: List[OpenAIFunctionToolSchema] | None = None,
        run_result: RolloutResult | None = None,
        run_error: Exception | None = None,
    ):
        self._tools = tools or []
        self._run_result = run_result
        self._run_error = run_error

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
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
def sample_messages() -> List[Dict[str, Any]]:
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]


@pytest.fixture
def sample_assistant_message() -> Dict[str, Any]:
    """Create a sample assistant message for testing."""
    return {"role": "assistant", "content": "The answer is 4."}


@pytest.fixture
def sample_assistant_message_with_tool_calls() -> Dict[str, Any]:
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
def sample_tool_message() -> Dict[str, Any]:
    """Create a sample tool result message."""
    return {
        "role": "tool",
        "content": "4",
        "tool_call_id": "call_123",
    }

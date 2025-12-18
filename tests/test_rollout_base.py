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

"""Tests for osmosis_ai.rollout.base."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutMetrics,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
)


# =============================================================================
# RolloutResult Tests
# =============================================================================


def test_rollout_result_completed_status() -> None:
    """Verify COMPLETED status RolloutResult creation."""
    result = RolloutResult(
        status="COMPLETED",
        final_messages=[{"role": "assistant", "content": "Done"}],
        finish_reason="stop",
    )
    assert result.status == "COMPLETED"
    assert result.finish_reason == "stop"
    assert result.error_message is None
    assert result.metrics is None


def test_rollout_result_error_status() -> None:
    """Verify ERROR status RolloutResult creation."""
    result = RolloutResult(
        status="ERROR",
        final_messages=[],
        finish_reason="error",
        error_message="Something went wrong",
    )
    assert result.status == "ERROR"
    assert result.finish_reason == "error"
    assert result.error_message == "Something went wrong"


def test_rollout_result_with_metrics() -> None:
    """Verify RolloutResult with metrics."""
    metrics = RolloutMetrics(
        total_latency_ms=1000.0,
        num_llm_calls=5,
    )
    result = RolloutResult(
        status="COMPLETED",
        final_messages=[],
        finish_reason="stop",
        metrics=metrics,
    )
    assert result.metrics is not None
    assert result.metrics.total_latency_ms == 1000.0
    assert result.metrics.num_llm_calls == 5


def test_rollout_result_final_messages() -> None:
    """Verify RolloutResult stores final messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = RolloutResult(
        status="COMPLETED",
        final_messages=messages,
        finish_reason="stop",
    )
    assert len(result.final_messages) == 2
    assert result.final_messages[1]["content"] == "Hi there!"


# =============================================================================
# RolloutContext Tests
# =============================================================================


def create_mock_llm_client() -> MagicMock:
    """Create a mock LLM client for testing."""
    mock_llm = MagicMock()
    mock_llm.get_metrics.return_value = RolloutMetrics(
        llm_latency_ms=100.0,
        num_llm_calls=2,
        prompt_tokens=50,
        response_tokens=25,
    )
    mock_llm.chat_completions = AsyncMock()
    return mock_llm


def test_rollout_context_complete_returns_completed_result(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify ctx.complete() returns COMPLETED result."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    result = ctx.complete(
        final_messages=[{"role": "assistant", "content": "Done"}],
        finish_reason="stop",
    )

    assert result.status == "COMPLETED"
    assert result.finish_reason == "stop"
    assert len(result.final_messages) == 1


def test_rollout_context_complete_default_finish_reason(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify ctx.complete() defaults finish_reason to 'stop'."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    result = ctx.complete(final_messages=[])

    assert result.finish_reason == "stop"


def test_rollout_context_error_returns_error_result(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify ctx.error() returns ERROR result."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    result = ctx.error(error_message="Test error")

    assert result.status == "ERROR"
    assert result.finish_reason == "error"
    assert result.error_message == "Test error"
    assert result.final_messages == []


def test_rollout_context_error_with_final_messages(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify ctx.error() can include final_messages."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    messages = [{"role": "assistant", "content": "Partial response"}]
    result = ctx.error(error_message="Failed", final_messages=messages)

    assert result.status == "ERROR"
    assert len(result.final_messages) == 1
    assert result.final_messages[0]["content"] == "Partial response"


def test_rollout_context_record_tool_call_increments_count(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify record_tool_call increments counter."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    assert ctx._num_tool_calls == 0

    ctx.record_tool_call()
    assert ctx._num_tool_calls == 1

    ctx.record_tool_call()
    assert ctx._num_tool_calls == 2


def test_rollout_context_record_tool_call_accumulates_latency(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify record_tool_call accumulates latency."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    ctx.record_tool_call(latency_ms=100.0)
    ctx.record_tool_call(latency_ms=50.0)

    assert ctx._tool_latency_ms == 150.0


def test_rollout_context_build_metrics_combines_all(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify _build_metrics combines LLM and tool metrics."""
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
        _start_time=0,  # Skip total latency calc
    )
    ctx._tool_latency_ms = 200.0
    ctx._num_tool_calls = 3

    metrics = ctx._build_metrics()

    assert metrics.llm_latency_ms == 100.0  # from mock
    assert metrics.num_llm_calls == 2  # from mock
    assert metrics.tool_latency_ms == 200.0
    assert metrics.num_tool_calls == 3
    assert metrics.prompt_tokens == 50
    assert metrics.response_tokens == 25


@pytest.mark.asyncio
async def test_rollout_context_chat_calls_llm(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify ctx.chat() delegates to llm.chat_completions()."""
    mock_llm = create_mock_llm_client()
    mock_result = MagicMock()
    mock_llm.chat_completions.return_value = mock_result

    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    messages = [{"role": "user", "content": "Hello"}]
    result = await ctx.chat(messages, temperature=0.5)

    mock_llm.chat_completions.assert_called_once_with(messages, temperature=0.5)
    assert result is mock_result


# =============================================================================
# RolloutAgentLoop Tests
# =============================================================================


def test_agent_loop_subclass_requires_name() -> None:
    """Verify subclass without name raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'name' class attribute"):

        class NoNameLoop(RolloutAgentLoop):
            def get_tools(self, request):
                return []

            async def run(self, ctx):
                return ctx.complete([])


def test_agent_loop_subclass_with_name_succeeds() -> None:
    """Verify subclass with name is created successfully."""

    class ValidLoop(RolloutAgentLoop):
        name = "valid_loop"

        def get_tools(self, request):
            return []

        async def run(self, ctx):
            return ctx.complete([])

    loop = ValidLoop()
    assert loop.name == "valid_loop"


def test_agent_loop_subclass_empty_name_rejected() -> None:
    """Verify empty string name raises TypeError."""
    # The __init_subclass__ check is `if not getattr(cls, 'name', None)`
    # So empty string is falsy and rejected at class level
    with pytest.raises(TypeError, match="must define a 'name' class attribute"):

        class EmptyNameLoop(RolloutAgentLoop):
            name = ""

            def get_tools(self, request):
                return []

            async def run(self, ctx):
                return ctx.complete([])


def test_agent_loop_abstract_methods_must_be_implemented() -> None:
    """Verify abstract methods must be implemented."""

    class PartialLoop(RolloutAgentLoop):
        name = "partial"

        def get_tools(self, request):
            return []

        # Missing run() method

    with pytest.raises(TypeError, match="abstract"):
        PartialLoop()


@pytest.mark.asyncio
async def test_agent_loop_run_can_return_complete_result(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify agent loop can return complete result."""

    class SimpleLoop(RolloutAgentLoop):
        name = "simple"

        def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
            return []

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            return ctx.complete(
                [{"role": "assistant", "content": "Hello!"}],
                finish_reason="stop",
            )

    loop = SimpleLoop()
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    result = await loop.run(ctx)

    assert result.status == "COMPLETED"
    assert result.final_messages[0]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_agent_loop_run_can_return_error_result(
    sample_rollout_request: RolloutRequest,
) -> None:
    """Verify agent loop can return error result."""

    class FailingLoop(RolloutAgentLoop):
        name = "failing"

        def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
            return []

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            return ctx.error("Something failed")

    loop = FailingLoop()
    mock_llm = create_mock_llm_client()
    ctx = RolloutContext(
        request=sample_rollout_request,
        tools=[],
        llm=mock_llm,
    )

    result = await loop.run(ctx)

    assert result.status == "ERROR"
    assert result.error_message == "Something failed"

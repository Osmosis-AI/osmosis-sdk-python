"""Reusable mock agent loop implementations for tests."""

from __future__ import annotations

import asyncio

from osmosis_ai.rollout import (
    OpenAIFunctionToolSchema,
    RolloutAgentLoop,
    RolloutContext,
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

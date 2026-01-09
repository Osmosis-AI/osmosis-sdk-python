"""Tests for test_mode runner."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
)
from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import RolloutMetrics
from osmosis_ai.rollout.test_mode.dataset import DatasetRow
from osmosis_ai.rollout.test_mode.exceptions import ToolValidationError
from osmosis_ai.rollout.test_mode.runner import (
    LocalTestBatchResult,
    LocalTestRunResult,
    LocalTestRunner,
)


class MockTestLLMClient:
    """Mock LLM client for testing.

    Implements the same interface as ExternalLLMClient for testing purposes.
    """

    def __init__(self) -> None:
        self._tools: List[Dict[str, Any]] | None = None
        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0
        self.completions_calls: List[Dict[str, Any]] = []
        self.mock_response = CompletionsResult(
            message={"role": "assistant", "content": "Test response"},
            token_ids=[],
            logprobs=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )

    def set_tools(self, tools: List[Any]) -> None:
        if tools:
            self._tools = [
                t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
                for t in tools
            ]
        else:
            self._tools = None

    def clear_tools(self) -> None:
        self._tools = None

    def reset_metrics(self) -> None:
        self._llm_latency_ms = 0.0
        self._num_llm_calls = 0
        self._prompt_tokens = 0
        self._response_tokens = 0

    def get_metrics(self) -> RolloutMetrics:
        return RolloutMetrics(
            llm_latency_ms=self._llm_latency_ms,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    def _record_usage(
        self, latency_ms: float, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self._llm_latency_ms += latency_ms
        self._num_llm_calls += 1
        self._prompt_tokens += prompt_tokens
        self._response_tokens += completion_tokens

    async def chat_completions(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> CompletionsResult:
        # Auto-inject tools if set
        if self._tools is not None and "tools" not in kwargs:
            kwargs["tools"] = self._tools
        self.completions_calls.append({"messages": messages, "kwargs": kwargs})
        self._record_usage(latency_ms=50.0, prompt_tokens=10, completion_tokens=5)
        return self.mock_response


class MockAgentLoop(RolloutAgentLoop):
    """Mock agent loop for testing."""

    name = "mock_test_agent"

    def __init__(
        self,
        tools: List[OpenAIFunctionToolSchema] | None = None,
        run_result: RolloutResult | None = None,
        run_error: Exception | None = None,
        call_llm: bool = False,
    ):
        self._tools = tools or []
        self._run_result = run_result
        self._run_error = run_error
        self._call_llm = call_llm
        self.get_tools_calls: List[RolloutRequest] = []
        self.run_calls: List[RolloutContext] = []

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        self.get_tools_calls.append(request)
        return self._tools

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        self.run_calls.append(ctx)
        if self._run_error:
            raise self._run_error
        if self._run_result:
            return self._run_result
        # Optionally make an LLM call to trigger metrics recording
        if self._call_llm:
            messages = list(ctx.request.messages)
            result = await ctx.chat(messages)
            messages.append(result.message)
            return ctx.complete(messages)
        return ctx.complete(list(ctx.request.messages))


def create_sample_tool() -> OpenAIFunctionToolSchema:
    """Create a sample tool schema for testing."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="test_tool",
            description="A test tool",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={},
                required=[],
            ),
        ),
    )


def create_sample_row(index: int = 0) -> DatasetRow:
    """Create a sample dataset row."""
    return {  # type: ignore[return-value]
        "user_prompt": f"Question {index}",
        "system_prompt": "You are a test assistant.",
        "ground_truth": f"Answer {index}",
    }


class TestLocalTestRunner:
    """Tests for LocalTestRunner class."""

    @pytest.mark.asyncio
    async def test_run_single_success(self) -> None:
        """Test successful single row execution."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is True
        assert result.row_index == 0
        assert result.result is not None
        assert result.result.status == "COMPLETED"
        assert result.error is None
        assert result.duration_ms > 0
        assert result.token_usage["prompt_tokens"] == 10
        assert result.token_usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_run_single_calls_get_tools(self) -> None:
        """Test that run_single calls agent.get_tools()."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(row, row_index=0)

        assert len(agent.get_tools_calls) == 1
        assert agent.get_tools_calls[0].rollout_id == "test-0"

    @pytest.mark.asyncio
    async def test_run_single_sets_tools_on_client(self) -> None:
        """Test that run_single injects tools into the client."""
        client = MockTestLLMClient()
        tool = create_sample_tool()
        agent = MockAgentLoop(tools=[tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(row, row_index=0)

        # Tools should have been set during execution
        # After execution, they should be cleared
        assert client._tools is None  # cleared after completion

    @pytest.mark.asyncio
    async def test_run_single_clears_tools_on_error(self) -> None:
        """Test that tools are cleared even on error."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()], run_error=RuntimeError("Test error")
        )
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is False
        assert "Test error" in result.error  # type: ignore[operator]
        assert client._tools is None  # should be cleared

    @pytest.mark.asyncio
    async def test_run_single_resets_metrics(self) -> None:
        """Test that metrics are reset before each run."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        # Simulate previous metrics
        client._record_usage(latency_ms=1000.0, prompt_tokens=100, completion_tokens=50)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        # Metrics should be from this run only, not accumulated
        assert result.token_usage["prompt_tokens"] == 10
        assert result.token_usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_run_single_with_max_turns(self) -> None:
        """Test that max_turns is passed to request."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(row, row_index=0, max_turns=20)

        assert len(agent.run_calls) == 1
        assert agent.run_calls[0].request.max_turns == 20

    @pytest.mark.asyncio
    async def test_run_single_with_completion_params(self) -> None:
        """Test that completion_params are passed through."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(
            row, row_index=0, completion_params={"temperature": 0.5}
        )

        assert len(agent.run_calls) == 1
        assert agent.run_calls[0].request.completion_params["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_run_batch(self) -> None:
        """Test running multiple rows."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_batch(rows)

        assert isinstance(result, LocalTestBatchResult)
        assert result.total == 3
        assert result.passed == 3
        assert result.failed == 0
        assert len(result.results) == 3
        assert result.total_tokens == 45  # 15 per row * 3
        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_batch_with_failures(self) -> None:
        """Test batch with some failures."""
        client = MockTestLLMClient()

        # Agent that fails on row 1
        class FailingAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                if ctx.request.metadata.get("row_index") == 1:
                    raise RuntimeError("Intentional failure")
                return ctx.complete(list(ctx.request.messages))

        agent = FailingAgent(tools=[create_sample_tool()])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_batch(rows)

        assert result.total == 3
        assert result.passed == 2
        assert result.failed == 1
        assert result.results[1].success is False
        assert "Intentional failure" in result.results[1].error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_run_batch_progress_callback(self) -> None:
        """Test that progress callback is called."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        progress_calls: List[tuple] = []

        def on_progress(current: int, total: int, result: LocalTestRunResult) -> None:
            progress_calls.append((current, total, result))

        rows = [create_sample_row(i) for i in range(3)]
        await runner.run_batch(rows, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1  # current
        assert progress_calls[0][1] == 3  # total
        assert progress_calls[2][0] == 3

    @pytest.mark.asyncio
    async def test_run_batch_with_start_index(self) -> None:
        """Test that start_index offsets row indices correctly."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_batch(rows, start_index=50)

        # Results should have offset row indices
        assert result.results[0].row_index == 50
        assert result.results[1].row_index == 51
        assert result.results[2].row_index == 52

        # Agent should have received requests with offset rollout IDs
        assert len(agent.run_calls) == 3
        assert agent.run_calls[0].request.rollout_id == "test-50"
        assert agent.run_calls[1].request.rollout_id == "test-51"
        assert agent.run_calls[2].request.rollout_id == "test-52"

        # Metadata should also have correct row_index
        assert agent.run_calls[0].request.metadata["row_index"] == 50
        assert agent.run_calls[1].request.metadata["row_index"] == 51
        assert agent.run_calls[2].request.metadata["row_index"] == 52

    @pytest.mark.asyncio
    async def test_run_batch_with_start_index_and_progress(self) -> None:
        """Test that progress callback receives correct offset row indices."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        progress_calls: List[tuple] = []

        def on_progress(current: int, total: int, result: LocalTestRunResult) -> None:
            progress_calls.append((current, total, result.row_index))

        rows = [create_sample_row(i) for i in range(3)]
        await runner.run_batch(rows, on_progress=on_progress, start_index=100)

        # Progress should report offset row indices
        assert progress_calls[0][2] == 100  # row_index for first row
        assert progress_calls[1][2] == 101
        assert progress_calls[2][2] == 102


class TestToolValidation:
    """Tests for tool schema validation."""

    @pytest.mark.asyncio
    async def test_valid_tool_passes(self) -> None:
        """Test that valid tool schema passes validation."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_tool_without_function_fails(self) -> None:
        """Test that tool without function field fails."""
        client = MockTestLLMClient()

        # Create invalid tool
        invalid_tool = MagicMock()
        invalid_tool.function = None

        agent = MockAgentLoop(tools=[invalid_tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is False
        assert "Tool validation error" in result.error  # type: ignore[operator]
        assert "missing 'function'" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_tool_without_name_fails(self) -> None:
        """Test that tool without function name fails."""
        client = MockTestLLMClient()

        # Create invalid tool
        invalid_tool = MagicMock()
        invalid_tool.function = MagicMock()
        invalid_tool.function.name = None

        agent = MockAgentLoop(tools=[invalid_tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is False
        assert "Tool validation error" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_tool_with_invalid_name_format_fails(self) -> None:
        """Test that tool with invalid name format fails."""
        client = MockTestLLMClient()

        # Create tool with invalid name (starts with number)
        invalid_tool = MagicMock()
        invalid_tool.function = MagicMock()
        invalid_tool.function.name = "123invalid"
        invalid_tool.function.parameters = None

        agent = MockAgentLoop(tools=[invalid_tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is False
        assert "Tool validation error" in result.error  # type: ignore[operator]
        assert "must start with letter/underscore" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_tool_with_invalid_parameters_type_fails(self) -> None:
        """Test that tool with invalid parameters type fails."""
        client = MockTestLLMClient()

        # Create tool with invalid parameters type
        invalid_tool = MagicMock()
        invalid_tool.function = MagicMock()
        invalid_tool.function.name = "valid_name"
        invalid_tool.function.parameters = MagicMock()
        invalid_tool.function.parameters.type = "array"  # should be "object"

        agent = MockAgentLoop(tools=[invalid_tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        result = await runner.run_single(row, row_index=0)

        assert result.success is False
        assert "parameters.type must be 'object'" in result.error  # type: ignore[operator]


class TestContextCreation:
    """Tests for RolloutContext creation in runner."""

    @pytest.mark.asyncio
    async def test_context_has_correct_request(self) -> None:
        """Test that context has correct request data."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row: Dict[str, Any] = {
            "user_prompt": "Test question",
            "system_prompt": "Test system",
            "ground_truth": "Test answer",
        }
        await runner.run_single(row, row_index=5)  # type: ignore[arg-type]

        assert len(agent.run_calls) == 1
        ctx = agent.run_calls[0]

        assert ctx.request.rollout_id == "test-5"
        assert ctx.request.messages[0]["role"] == "system"
        assert ctx.request.messages[0]["content"] == "Test system"
        assert ctx.request.messages[1]["role"] == "user"
        assert ctx.request.messages[1]["content"] == "Test question"
        assert ctx.request.metadata["ground_truth"] == "Test answer"
        assert ctx.request.metadata["test_mode"] is True

    @pytest.mark.asyncio
    async def test_context_has_tools(self) -> None:
        """Test that context has tools from agent."""
        client = MockTestLLMClient()
        tool = create_sample_tool()
        agent = MockAgentLoop(tools=[tool])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(row, row_index=0)

        assert len(agent.run_calls) == 1
        ctx = agent.run_calls[0]
        assert len(ctx.tools) == 1
        assert ctx.tools[0].function.name == "test_tool"

    @pytest.mark.asyncio
    async def test_context_has_llm_client(self) -> None:
        """Test that context has the LLM client."""
        client = MockTestLLMClient()
        agent = MockAgentLoop(tools=[])
        runner = LocalTestRunner(agent_loop=agent, llm_client=client)

        row = create_sample_row(0)
        await runner.run_single(row, row_index=0)

        assert len(agent.run_calls) == 1
        ctx = agent.run_calls[0]
        assert ctx.llm is client

"""Tests for Phase 2 runner adaptations: messages, row_index, and run_batch()."""

from __future__ import annotations

from typing import Any

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
from osmosis_ai.rollout.eval.common.dataset import DatasetRow
from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.rollout.eval.evaluation.runner import EvalRunner, EvalRunResult

# ---------------------------------------------------------------------------
# Helpers (mirroring test_runner.py patterns)
# ---------------------------------------------------------------------------


class MockLLMClient:
    def __init__(self, model: str = "mock-model") -> None:
        self.model = model
        self.display_name = model
        self._api_key: str | None = None
        self._api_base: str | None = None
        self._tools: list[dict[str, Any]] | None = None
        self._prompt_tokens = 0
        self._response_tokens = 0
        self._num_llm_calls = 0
        self.mock_response = CompletionsResult(
            message={"role": "assistant", "content": "eval response"},
            token_ids=[],
            logprobs=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )

    def set_tools(self, tools: list[Any]) -> None:
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
        self._prompt_tokens = 0
        self._response_tokens = 0
        self._num_llm_calls = 0

    def get_metrics(self) -> RolloutMetrics:
        return RolloutMetrics(
            llm_latency_ms=0.0,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    async def chat_completions(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> CompletionsResult:
        if self._tools is not None and "tools" not in kwargs:
            kwargs["tools"] = self._tools
        self._prompt_tokens += 10
        self._response_tokens += 5
        self._num_llm_calls += 1
        return self.mock_response


class MockAgentLoop(RolloutAgentLoop):
    name = "adaptation_test_agent"

    def __init__(
        self,
        tools: list[OpenAIFunctionToolSchema] | None = None,
        run_error: Exception | None = None,
        call_llm: bool = False,
    ) -> None:
        self._tools = tools or []
        self._run_error = run_error
        self._call_llm = call_llm

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return self._tools

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        if self._run_error:
            raise self._run_error
        messages = list(ctx.request.messages)
        if self._call_llm:
            completion = await ctx.chat(messages)
            messages.append(completion.message)
        return ctx.complete(messages)


def create_sample_tool() -> OpenAIFunctionToolSchema:
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
    return {  # type: ignore[return-value]
        "user_prompt": f"Question {index}",
        "system_prompt": "You are a test assistant.",
        "ground_truth": f"Answer {index}",
    }


def _make_simple_eval() -> EvalFnWrapper:
    def simple_eval(
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, Any],
    ) -> float:
        return 1.0 if "response" in solution_str else 0.0

    return EvalFnWrapper(simple_eval, "simple_eval")


# ---------------------------------------------------------------------------
# Tests: EvalRunResult.messages populated on success
# ---------------------------------------------------------------------------


class TestEvalRunResultMessages:
    @pytest.mark.asyncio
    async def test_messages_populated_on_success(self) -> None:
        """Successful run_single should populate messages with final_messages."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
        )

        assert result.success is True
        assert result.messages is not None
        assert isinstance(result.messages, list)
        assert len(result.messages) > 0
        # The agent appends the LLM response, so last message should be assistant
        assert result.messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_messages_none_on_failure(self) -> None:
        """Failed run_single should set messages to None."""
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("agent failure"),
        )

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
        )

        assert result.success is False
        assert result.messages is None


# ---------------------------------------------------------------------------
# Tests: EvalRunResult.row_index propagated
# ---------------------------------------------------------------------------


class TestEvalRunResultRowIndex:
    @pytest.mark.asyncio
    async def test_row_index_propagated_on_success(self) -> None:
        """Successful run should propagate the row_index."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=42,
            run_index=0,
        )

        assert result.success is True
        assert result.row_index == 42

    @pytest.mark.asyncio
    async def test_row_index_propagated_on_failure(self) -> None:
        """Failed run should also propagate the row_index."""
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("agent failure"),
        )

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=99,
            run_index=0,
        )

        assert result.success is False
        assert result.row_index == 99


# ---------------------------------------------------------------------------
# Tests: Default values backward compatibility
# ---------------------------------------------------------------------------


class TestEvalRunResultDefaults:
    def test_default_messages_is_none(self) -> None:
        """Creating EvalRunResult without messages should default to None."""
        result = EvalRunResult(run_index=0, success=True)
        assert result.messages is None

    def test_default_row_index_is_zero(self) -> None:
        """Creating EvalRunResult without row_index should default to 0."""
        result = EvalRunResult(run_index=0, success=True)
        assert result.row_index == 0

    def test_old_style_kwargs_still_work(self) -> None:
        """Creating EvalRunResult with only pre-existing fields should work."""
        result = EvalRunResult(
            run_index=1,
            success=True,
            scores={"eval": 0.5},
            duration_ms=100.0,
            tokens=10,
            error=None,
            model_tag="primary",
        )
        assert result.run_index == 1
        assert result.success is True
        assert result.scores == {"eval": 0.5}
        assert result.duration_ms == 100.0
        assert result.tokens == 10
        assert result.model_tag == "primary"
        # New fields should have defaults
        assert result.messages is None
        assert result.row_index == 0


# ---------------------------------------------------------------------------
# Tests: run_batch() basic functionality
# ---------------------------------------------------------------------------


class TestRunBatch:
    @pytest.mark.asyncio
    async def test_run_batch_basic(self) -> None:
        """Multiple work items should return correct ordered results."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 0, 0, None),
            (create_sample_row(1), 1, 0, None),
            (create_sample_row(2), 2, 0, None),
        ]

        batch_results, systemic_error = await runner.run_batch(work_items)

        assert systemic_error is None
        assert len(batch_results) == 3
        for i, result in enumerate(batch_results):
            assert result is not None
            assert result.success is True
            assert result.row_index == i
            assert result.run_index == 0
            assert result.messages is not None

    @pytest.mark.asyncio
    async def test_run_batch_preserves_order(self) -> None:
        """Results should be index-aligned with work_items."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 10, 0, None),
            (create_sample_row(1), 20, 1, None),
            (create_sample_row(2), 30, 2, None),
        ]

        batch_results, _ = await runner.run_batch(work_items)

        assert batch_results[0] is not None
        assert batch_results[0].row_index == 10
        assert batch_results[0].run_index == 0

        assert batch_results[1] is not None
        assert batch_results[1].row_index == 20
        assert batch_results[1].run_index == 1

        assert batch_results[2] is not None
        assert batch_results[2].row_index == 30
        assert batch_results[2].run_index == 2

    @pytest.mark.asyncio
    async def test_run_batch_empty_work_items(self) -> None:
        """Empty work_items should return empty list and None."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()])

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        batch_results, systemic_error = await runner.run_batch([])

        assert batch_results == []
        assert systemic_error is None

    @pytest.mark.asyncio
    async def test_run_batch_handles_systemic_error(self) -> None:
        """SystemicProviderError should be captured; other items may complete."""

        class SystemicAgent(MockAgentLoop):
            def __init__(self) -> None:
                super().__init__(tools=[create_sample_tool()])
                self._call_count = 0

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                self._call_count += 1
                if self._call_count == 1:
                    raise SystemicProviderError("Auth failed")
                messages = list(ctx.request.messages)
                return ctx.complete(messages)

        client = MockLLMClient()
        agent = SystemicAgent()

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 0, 0, None),
            (create_sample_row(1), 1, 0, None),
        ]

        batch_results, systemic_error = await runner.run_batch(work_items)

        assert systemic_error is not None
        assert "Auth failed" in systemic_error
        assert len(batch_results) == 2
        # The failing item should have a result with success=False
        failed = [r for r in batch_results if r is not None and not r.success]
        assert len(failed) >= 1
        assert failed[0].messages is None

    @pytest.mark.asyncio
    async def test_run_batch_normal_failures_no_systemic_error(self) -> None:
        """Non-systemic failures should be recorded but systemic_error stays None."""
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("normal failure"),
        )

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 0, 0, None),
            (create_sample_row(1), 1, 0, None),
        ]

        batch_results, systemic_error = await runner.run_batch(work_items)

        assert systemic_error is None
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert result.success is False
            assert result.messages is None

    @pytest.mark.asyncio
    async def test_run_batch_systemic_error_result_has_row_index(self) -> None:
        """Systemic error results should have correct row_index."""

        class SystemicAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                raise SystemicProviderError("Budget exceeded")

        client = MockLLMClient()
        agent = SystemicAgent(tools=[create_sample_tool()])

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 5, 0, None),
            (create_sample_row(1), 7, 1, None),
        ]

        batch_results, systemic_error = await runner.run_batch(work_items)

        assert systemic_error is not None
        for i, result in enumerate(batch_results):
            assert result is not None
            assert result.row_index == work_items[i][1]
            assert result.messages is None

    @pytest.mark.asyncio
    async def test_run_batch_with_model_tags(self) -> None:
        """run_batch should correctly handle primary/baseline model tags."""
        primary_client = MockLLMClient(model="primary")
        baseline_client = MockLLMClient(model="baseline")
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=primary_client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
            baseline_llm_client=baseline_client,  # type: ignore[arg-type]
            baseline_llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (create_sample_row(0), 0, 0, "primary"),
            (create_sample_row(0), 0, 0, "baseline"),
        ]

        batch_results, systemic_error = await runner.run_batch(work_items)

        assert systemic_error is None
        assert len(batch_results) == 2
        assert batch_results[0] is not None
        assert batch_results[0].model_tag == "primary"
        assert batch_results[1] is not None
        assert batch_results[1].model_tag == "baseline"


# ---------------------------------------------------------------------------
# Tests: run_eval() backward compatibility
# ---------------------------------------------------------------------------


class TestRunEvalBackwardCompat:
    @pytest.mark.asyncio
    async def test_run_eval_still_works(self) -> None:
        """run_eval() should still work correctly (no regression)."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1)],
            n_runs=2,
            pass_threshold=0.5,
        )

        assert eval_result.total_rows == 2
        assert eval_result.total_runs == 4
        assert eval_result.stopped_early is False
        assert "simple_eval" in eval_result.eval_summaries

        # Verify new fields are present in run results
        for row_result in eval_result.rows:
            for run in row_result.runs:
                if run.success:
                    assert run.messages is not None
                else:
                    assert run.messages is None
                # row_index should match the row
                assert run.row_index == row_result.row_index

    @pytest.mark.asyncio
    async def test_run_eval_concurrent_still_works(self) -> None:
        """run_eval() with batch_size > 1 should still work correctly."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(i) for i in range(3)],
            n_runs=1,
            batch_size=2,
        )

        assert eval_result.total_rows == 3
        assert eval_result.total_runs == 3
        assert eval_result.stopped_early is False

    @pytest.mark.asyncio
    async def test_run_eval_messages_match_final_messages(self) -> None:
        """run_eval() results should contain messages matching the conversation."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[_make_simple_eval()],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0)],
            n_runs=1,
        )

        run = eval_result.rows[0].runs[0]
        assert run.success is True
        assert run.messages is not None
        # Messages should include the system prompt and user prompt from the row,
        # plus the assistant response
        roles = [m["role"] for m in run.messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

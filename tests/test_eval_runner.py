"""Tests for eval runner."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

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
from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.rollout.eval.evaluation.runner import EvalRunner
from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import RolloutMetrics
from osmosis_ai.rollout.eval.common.dataset import DatasetRow


class MockLLMClient:
    def __init__(self, model: str = "mock-model") -> None:
        self.model = model
        self.display_name = model
        self._api_key: str | None = None
        self._api_base: str | None = None
        self._tools: List[Dict[str, Any]] | None = None
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
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> CompletionsResult:
        if self._tools is not None and "tools" not in kwargs:
            kwargs["tools"] = self._tools
        self._prompt_tokens += 10
        self._response_tokens += 5
        self._num_llm_calls += 1
        return self.mock_response


class MockAgentLoop(RolloutAgentLoop):
    name = "eval_test_agent"

    def __init__(
        self,
        tools: List[OpenAIFunctionToolSchema] | None = None,
        run_error: Exception | None = None,
        call_llm: bool = False,
    ) -> None:
        self._tools = tools or []
        self._run_error = run_error
        self._call_llm = call_llm

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
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


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_single_applies_eval_functions(self) -> None:
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        async def full_eval(
            messages: List[Dict[str, Any]],
            ground_truth: str,
            metadata: Dict[str, Any],
        ) -> float:
            assert ground_truth.startswith("Answer")
            assert "user_prompt" in metadata
            return 0.8 if messages[-1]["role"] == "assistant" else 0.0

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0 if "response" in solution_str else 0.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[
                EvalFnWrapper(full_eval, "full_eval"),
                EvalFnWrapper(simple_eval, "simple_eval"),
            ],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
        )

        assert result.success is True
        assert result.tokens == 15
        assert result.scores["full_eval"] == 0.8
        assert result.scores["simple_eval"] == 1.0

    @pytest.mark.asyncio
    async def test_run_single_propagates_agent_failure(self) -> None:
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("eval failure"),
        )

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
        )

        assert result.success is False
        assert result.error is not None
        assert "eval failure" in result.error
        assert result.scores == {}

    @pytest.mark.asyncio
    async def test_run_eval_computes_pass_at_k(self) -> None:
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        call_counter = {"n": 0}

        def alternating_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            call_counter["n"] += 1
            return 1.0 if call_counter["n"] % 2 == 1 else 0.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(alternating_eval, "alternating_eval")],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1)],
            n_runs=2,
            pass_threshold=0.5,
        )

        summary = eval_result.eval_summaries["alternating_eval"]
        assert eval_result.total_runs == 4
        assert summary.mean == 0.5
        assert 1 in summary.pass_at_k
        assert 2 not in summary.pass_at_k

    @pytest.mark.asyncio
    async def test_run_eval_concurrent(self) -> None:
        """batch_size > 1 should produce the same results as sequential."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0 if "response" in solution_str else 0.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(i) for i in range(4)]

        result = await runner.run_eval(
            rows=rows,
            n_runs=2,
            batch_size=3,
        )

        assert result.total_rows == 4
        assert result.total_runs == 8
        assert len(result.rows) == 4
        # Each row should have exactly 2 runs, ordered by run_index
        for row_result in result.rows:
            assert len(row_result.runs) == 2
            assert row_result.runs[0].run_index == 0
            assert row_result.runs[1].run_index == 1
        # Eval scores should still be computed
        assert "simple_eval" in result.eval_summaries

    @pytest.mark.asyncio
    async def test_run_eval_batch_size_gt_one_defaults_to_concurrent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """batch_size > 1 should use concurrent mode by default."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0 if "response" in solution_str else 0.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        concurrent_called = {"value": False}
        original_concurrent = EvalRunner._run_eval_concurrent

        async def wrapped_concurrent(self: EvalRunner, *args: Any, **kwargs: Any):
            concurrent_called["value"] = True
            return await original_concurrent(self, *args, **kwargs)

        monkeypatch.setattr(EvalRunner, "_run_eval_concurrent", wrapped_concurrent)

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_eval(
            rows=rows,
            n_runs=1,
            batch_size=2,
        )

        assert concurrent_called["value"] is True
        assert result.total_rows == 3
        assert result.total_runs == 3

    @pytest.mark.asyncio
    async def test_run_eval_concurrent_stops_early_on_systemic_error(self) -> None:
        """Concurrent eval should cancel pending runs on systemic provider errors."""
        from osmosis_ai.rollout.eval.common.errors import SystemicProviderError

        class SystemicAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                await asyncio.sleep(0.02)
                raise SystemicProviderError("Authentication failed")

        client = MockLLMClient()
        agent = SystemicAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(i) for i in range(20)]
        result = await runner.run_eval(
            rows=rows,
            n_runs=2,
            batch_size=3,
        )

        assert result.stopped_early is True
        assert "Authentication failed" in (result.stop_reason or "")
        assert result.total_runs == 3
        assert result.total_rows <= len(rows)

    @pytest.mark.asyncio
    async def test_run_eval_concurrent_systemic_error_without_pending_not_early_stop(
        self,
    ) -> None:
        """Systemic errors should not imply early-stop when no tasks were canceled."""
        from osmosis_ai.rollout.eval.common.errors import SystemicProviderError

        class SystemicAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                raise SystemicProviderError("Authentication failed")

        client = MockLLMClient()
        agent = SystemicAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        result = await runner.run_eval(
            rows=[create_sample_row(0)],
            n_runs=1,
            batch_size=3,
        )

        assert result.total_rows == 1
        assert result.total_runs == 1
        assert result.rows[0].runs[0].success is False
        assert "Authentication failed" in (result.rows[0].runs[0].error or "")
        assert result.stopped_early is False
        assert result.stop_reason is None

    @pytest.mark.asyncio
    async def test_run_eval_concurrent_normal_failures_continue(self) -> None:
        """Non-systemic concurrent failures should be recorded without early stopping."""
        class FailingAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                raise RuntimeError("eval failure")

        client = MockLLMClient()
        agent = FailingAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(i) for i in range(4)]
        result = await runner.run_eval(
            rows=rows,
            n_runs=2,
            batch_size=3,
        )

        assert result.stopped_early is False
        assert result.total_runs == 8
        assert result.total_rows == 4
        assert all(not run.success for row in result.rows for run in row.runs)

    @pytest.mark.asyncio
    async def test_run_eval_counts_failed_runs_as_zero_scores(self) -> None:
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("eval failure"),
        )

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0)],
            n_runs=2,
            pass_threshold=0.5,
            batch_size=2,
        )

        summary = eval_result.eval_summaries["simple_eval"]
        assert eval_result.total_runs == 2
        assert summary.mean == 0.0
        assert summary.min == 0.0
        assert summary.max == 0.0
        assert summary.pass_at_k.get(1) == 0.0

    @pytest.mark.asyncio
    async def test_run_eval_stops_early_on_systemic_error(self) -> None:
        """SystemicProviderError should stop eval early."""
        from osmosis_ai.rollout.eval.common.errors import SystemicProviderError

        call_count = {"n": 0}

        class SystemicAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                call_count["n"] += 1
                raise SystemicProviderError("Authentication failed")

        client = MockLLMClient()
        agent = SystemicAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1), create_sample_row(2)],
            n_runs=2,
            pass_threshold=0.5,
        )

        assert eval_result.stopped_early is True
        assert eval_result.stop_reason is not None
        assert "Authentication failed" in eval_result.stop_reason
        assert eval_result.total_rows == 1  # only first row attempted
        assert call_count["n"] == 1  # only one agent run attempted

    @pytest.mark.asyncio
    async def test_run_eval_systemic_error_preserves_duration_and_tokens(self) -> None:
        """Systemic failures should keep per-run duration/token stats."""
        from osmosis_ai.rollout.eval.common.errors import SystemicProviderError

        class SystemicAfterLLMAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                messages = list(ctx.request.messages)
                _ = await ctx.chat(messages)
                await asyncio.sleep(0.01)
                raise SystemicProviderError("Authentication failed")

        client = MockLLMClient()
        agent = SystemicAfterLLMAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1)],
            n_runs=1,
        )

        first_run = eval_result.rows[0].runs[0]
        assert first_run.success is False
        assert first_run.duration_ms > 0
        assert first_run.tokens == 15

    @pytest.mark.asyncio
    async def test_run_eval_concurrent_systemic_error_preserves_duration_and_tokens(self) -> None:
        """Concurrent systemic failures should keep per-run duration/token stats."""
        from osmosis_ai.rollout.eval.common.errors import SystemicProviderError

        class SystemicAfterLLMAgent(MockAgentLoop):
            async def run(self, ctx: RolloutContext) -> RolloutResult:
                messages = list(ctx.request.messages)
                _ = await ctx.chat(messages)
                await asyncio.sleep(0.01)
                raise SystemicProviderError("Authentication failed")

        client = MockLLMClient()
        agent = SystemicAfterLLMAgent(tools=[create_sample_tool()])

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1), create_sample_row(2)],
            n_runs=1,
            batch_size=2,
        )

        failed_runs = [run for row in eval_result.rows for run in row.runs if not run.success]
        assert failed_runs
        assert all(run.duration_ms > 0 for run in failed_runs)
        assert all(run.tokens == 15 for run in failed_runs)

    @pytest.mark.asyncio
    async def test_run_eval_continues_on_non_systemic_failure(self) -> None:
        """Non-systemic failures should be recorded without early stopping."""
        client = MockLLMClient()
        agent = MockAgentLoop(
            tools=[create_sample_tool()],
            run_error=RuntimeError("eval failure"),
        )

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0), create_sample_row(1)],
            n_runs=2,
            pass_threshold=0.5,
        )

        assert eval_result.stopped_early is False
        assert eval_result.total_rows == 2
        assert eval_result.total_runs == 4
        assert all(not run.success for row in eval_result.rows for run in row.runs)


class TestEvalRunnerBaseline:
    """Tests for baseline model comparison support."""

    @pytest.mark.asyncio
    async def test_run_single_with_model_tag(self) -> None:
        """model_tag should be propagated through to EvalRunResult."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
        )

        result = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
            model_tag="primary",
        )
        assert result.model_tag == "primary"
        assert result.success is True

        result_none = await runner.run_single(
            row=create_sample_row(0),
            row_index=0,
            run_index=0,
        )
        assert result_none.model_tag is None

    @pytest.mark.asyncio
    async def test_run_eval_baseline_sequential(self) -> None:
        """Sequential mode with baseline should run both models per row."""
        primary_client = MockLLMClient()
        baseline_client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=primary_client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            baseline_llm_client=baseline_client,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(0), create_sample_row(1)]
        result = await runner.run_eval(
            rows=rows,
            n_runs=1,
            batch_size=1,
        )

        # 2 rows * 1 run * 2 models = 4 total runs
        assert result.total_runs == 4
        assert result.total_rows == 2

        # Each row should have 2 runs (primary + baseline)
        for row_result in result.rows:
            tags = [r.model_tag for r in row_result.runs]
            assert "primary" in tags
            assert "baseline" in tags

        # model_summaries should be populated
        assert result.model_summaries is not None
        assert len(result.model_summaries) == 2
        assert result.model_summaries[0].model_tag == "primary"
        assert result.model_summaries[1].model_tag == "baseline"

    @pytest.mark.asyncio
    async def test_run_eval_baseline_concurrent(self) -> None:
        """Concurrent mode with baseline should also produce comparison results."""
        primary_client = MockLLMClient()
        baseline_client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=primary_client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
            baseline_llm_client=baseline_client,  # type: ignore[arg-type]
            baseline_llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_eval(
            rows=rows,
            n_runs=1,
            batch_size=4,
        )

        # 3 rows * 1 run * 2 models = 6 total runs
        assert result.total_runs == 6
        assert result.total_rows == 3

        for row_result in result.rows:
            tags = [r.model_tag for r in row_result.runs]
            assert "primary" in tags
            assert "baseline" in tags

        assert result.model_summaries is not None

    @pytest.mark.asyncio
    async def test_run_eval_no_baseline_backward_compat(self) -> None:
        """Without baseline, model_tag should be None and no comparison data."""
        client = MockLLMClient()
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def simple_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(simple_eval, "simple_eval")],
        )

        rows = [create_sample_row(0)]
        result = await runner.run_eval(rows=rows, n_runs=1)

        assert result.total_runs == 1
        assert result.model_summaries is None
        for row_result in result.rows:
            for run in row_result.runs:
                assert run.model_tag is None

    @pytest.mark.asyncio
    async def test_eval_summaries_not_polluted_by_baseline_sequential(self) -> None:
        """Top-level eval_summaries should only reflect primary model scores."""
        primary_client = MockLLMClient()
        # Baseline returns different content so eval scores differ
        baseline_client = MockLLMClient()
        baseline_client.mock_response = CompletionsResult(
            message={"role": "assistant", "content": "baseline wrong answer"},
            token_ids=[],
            logprobs=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def score_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            # Primary gets "eval response" -> 1.0, baseline gets "baseline wrong answer" -> 0.0
            return 1.0 if "eval response" in solution_str else 0.0

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=primary_client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(score_eval, "score_eval")],
            baseline_llm_client=baseline_client,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(0), create_sample_row(1)]
        result = await runner.run_eval(rows=rows, n_runs=1, batch_size=1)

        # Top-level eval_summaries should only contain primary scores (1.0)
        assert result.eval_summaries["score_eval"].mean == 1.0
        # If polluted by baseline, mean would be 0.5

        # model_summaries should have correct per-model data
        assert result.model_summaries is not None
        primary_summary = result.model_summaries[0]
        baseline_summary = result.model_summaries[1]
        assert primary_summary.model_tag == "primary"
        assert primary_summary.eval_summaries["score_eval"].mean == 1.0
        assert baseline_summary.model_tag == "baseline"
        assert baseline_summary.eval_summaries["score_eval"].mean == 0.0

    @pytest.mark.asyncio
    async def test_eval_summaries_not_polluted_by_baseline_concurrent(self) -> None:
        """Concurrent path: top-level eval_summaries should only reflect primary."""
        primary_client = MockLLMClient()
        baseline_client = MockLLMClient()
        baseline_client.mock_response = CompletionsResult(
            message={"role": "assistant", "content": "baseline wrong answer"},
            token_ids=[],
            logprobs=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )
        agent = MockAgentLoop(tools=[create_sample_tool()], call_llm=True)

        def score_eval(
            solution_str: str,
            ground_truth: str,
            extra_info: Dict[str, Any],
        ) -> float:
            return 1.0 if "eval response" in solution_str else 0.0

        def make_baseline() -> MockLLMClient:
            c = MockLLMClient()
            c.mock_response = CompletionsResult(
                message={"role": "assistant", "content": "baseline wrong answer"},
                token_ids=[],
                logprobs=[],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                finish_reason="stop",
            )
            return c

        runner = EvalRunner(
            agent_loop=agent,
            llm_client=primary_client,  # type: ignore[arg-type]
            eval_fns=[EvalFnWrapper(score_eval, "score_eval")],
            llm_client_factory=MockLLMClient,  # type: ignore[arg-type]
            baseline_llm_client=baseline_client,  # type: ignore[arg-type]
            baseline_llm_client_factory=make_baseline,  # type: ignore[arg-type]
        )

        rows = [create_sample_row(i) for i in range(3)]
        result = await runner.run_eval(rows=rows, n_runs=1, batch_size=4)

        # Top-level should be primary-only
        assert result.eval_summaries["score_eval"].mean == 1.0

        # Per-model should be separated
        assert result.model_summaries is not None
        primary_summary = next(
            ms for ms in result.model_summaries if ms.model_tag == "primary"
        )
        baseline_summary = next(
            ms for ms in result.model_summaries if ms.model_tag == "baseline"
        )
        assert primary_summary.eval_summaries["score_eval"].mean == 1.0
        assert baseline_summary.eval_summaries["score_eval"].mean == 0.0

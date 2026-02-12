"""Tests for eval runner."""

from __future__ import annotations

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
    def __init__(self) -> None:
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
        )

        eval_result = await runner.run_eval(
            rows=[create_sample_row(0)],
            n_runs=2,
            pass_threshold=0.5,
        )

        summary = eval_result.eval_summaries["simple_eval"]
        assert eval_result.total_runs == 2
        assert summary.mean == 0.0
        assert summary.min == 0.0
        assert summary.max == 0.0
        assert summary.pass_at_k.get(1) == 0.0

"""Tests for osmosis_ai.rollout.backend.local.backend."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local.backend import LocalBackend
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    SampleSource,
)
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    AgentWorkflowOutput,
    ConcurrencyConfig,
    ExecutionRequest,
    GraderConfig,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.errors import categorize_exception

# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class StaticSampleSource(SampleSource):
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    async def get_sample(self) -> RolloutSample:
        return RolloutSample(messages=self.messages)


class StubWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        from osmosis_ai.rollout.context import get_rollout_context

        rollout_ctx = get_rollout_context()
        if rollout_ctx:
            rollout_ctx.set_sample_source(
                StaticSampleSource([{"role": "assistant", "content": "done"}]),
            )


class FailingWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        raise ValueError("workflow error")


class StubGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        ctx.set_reward(1.0)


class FailingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        raise RuntimeError("grading failed")


# ---------------------------------------------------------------------------
# categorize_exception
# ---------------------------------------------------------------------------


class TestCategorizeException:
    def test_timeout(self):
        assert categorize_exception(TimeoutError()) == RolloutErrorCategory.TIMEOUT

    def test_value_error(self):
        assert (
            categorize_exception(ValueError("bad"))
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_type_error(self):
        assert (
            categorize_exception(TypeError("bad"))
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_assertion_error(self):
        assert (
            categorize_exception(AssertionError())
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_generic(self):
        assert (
            categorize_exception(RuntimeError("boom"))
            == RolloutErrorCategory.AGENT_ERROR
        )


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class TestLocalBackend:
    def _make_backend(self, *, grader=None, grader_config=None):
        return LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=grader,
            grader_config=grader_config,
        )

    def test_health(self):
        backend = self._make_backend()
        h = backend.health()
        assert h["status"] == "ok"
        assert "concurrency" in h

    async def test_rollout_artifacts_land_under_rollout_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))  # redirect ~/.osmosis

        class ArtifactWorkflow(StubWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                assert ctx.artifacts_dir is not None and ctx.artifacts_dir.is_dir()
                (ctx.artifacts_dir / "trace.txt").write_text("trace")
                await super().run(ctx)

        class WritingGrader(StubGrader):
            async def grade(self, ctx: GraderContext) -> Any:
                assert ctx.artifacts_dir is not None
                (ctx.artifacts_dir / "grade_debug.json").write_text("{}")
                await super().grade(ctx)

        backend = LocalBackend(
            workflow=ArtifactWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=WritingGrader,
        )

        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], label="x"
        )
        await backend.execute(request, AsyncMock(), AsyncMock())

        root = tmp_path / ".osmosis"
        artifacts_dir = root / "r1" / "artifacts"
        assert (artifacts_dir / "trace.txt").read_text() == "trace"
        assert (artifacts_dir / "grade_debug.json").read_text() == "{}"

    async def test_rollout_degrades_when_artifacts_dir_unavailable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "osmosis_ai.rollout.utils.file_artifacts.CREATE_BACKOFF_SECONDS", 0
        )

        def _boom(self, *_args, **_kwargs):
            raise OSError("read-only filesystem")

        monkeypatch.setattr("pathlib.Path.mkdir", _boom)

        class NoArtifactsWorkflow(StubWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                assert ctx.artifacts_dir is None
                await super().run(ctx)

        class NoArtifactsGrader(StubGrader):
            async def grade(self, ctx: GraderContext) -> Any:
                assert ctx.artifacts_dir is None
                await super().grade(ctx)

        backend = LocalBackend(
            workflow=NoArtifactsWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=NoArtifactsGrader,
        )

        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], label="x"
        )
        on_complete = AsyncMock()
        await backend.execute(request, on_complete, AsyncMock())

        # mkdir failure degrades to "artifacts unavailable"; rollout still succeeds.
        on_complete.assert_awaited_once()
        assert on_complete.call_args[0][0].status == RolloutStatus.SUCCESS

    async def test_execute_success_calls_callback(self):
        backend = self._make_backend()
        on_complete = AsyncMock()

        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])
        await backend.execute(request, on_workflow_complete=on_complete)

        on_complete.assert_awaited_once()
        result = on_complete.call_args[0][0]
        assert result.status == RolloutStatus.SUCCESS

    async def test_explicit_output_is_primary_sample_source(self):
        class ReturningWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> AgentWorkflowOutput:
                from osmosis_ai.rollout.context import get_rollout_context

                rollout_ctx = get_rollout_context()
                assert rollout_ctx is not None
                rollout_ctx.set_sample_source(
                    StaticSampleSource([{"role": "assistant", "content": "ambient"}]),
                )
                return AgentWorkflowOutput(
                    messages=[{"role": "assistant", "content": "returned"}],
                    metrics={"quality": 0.75},
                )

        backend = LocalBackend(
            workflow=ReturningWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )

        result = await backend.run_workflow(
            ExecutionRequest(
                id="r1",
                prompt=[{"role": "user", "content": "hi"}],
                label="expected",
            )
        )

        assert result.status == RolloutStatus.SUCCESS
        assert result.sample is not None
        assert result.sample.messages == [{"role": "assistant", "content": "returned"}]
        assert result.sample.label == "expected"
        assert result.sample.metrics == {"quality": 0.75}

    async def test_bare_messages_return_is_accepted(self):
        class ReturningWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> list[dict[str, Any]]:
                return [{"role": "assistant", "content": "returned"}]

        backend = LocalBackend(
            workflow=ReturningWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )

        result = await backend.run_workflow(
            ExecutionRequest(
                id="r1",
                prompt=[{"role": "user", "content": "hi"}],
            )
        )

        assert result.status == RolloutStatus.SUCCESS
        assert result.sample is not None
        assert result.sample.messages == [{"role": "assistant", "content": "returned"}]

    async def test_mutated_non_finite_metrics_fail_validation(self):
        class ReturningWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> AgentWorkflowOutput:
                output = AgentWorkflowOutput(metrics={"score": 1.0})
                output.metrics["score"] = float("nan")
                return output

        backend = LocalBackend(
            workflow=ReturningWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )

        result = await backend.run_workflow(
            ExecutionRequest(
                id="r1",
                prompt=[{"role": "user", "content": "hi"}],
            )
        )

        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR
        assert result.err_message is not None
        assert "finite" in result.err_message

    async def test_execute_failure_calls_callback_with_error(self):
        backend = LocalBackend(
            workflow=FailingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        on_complete = AsyncMock()

        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])
        await backend.execute(request, on_workflow_complete=on_complete)

        on_complete.assert_awaited_once()
        result = on_complete.call_args[0][0]
        assert result.status == RolloutStatus.FAILURE
        assert "workflow error" in result.err_message
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR

    async def test_execute_with_grader(self):
        captured: dict[str, Any] = {}

        class ReturningWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> AgentWorkflowOutput:
                return AgentWorkflowOutput(
                    messages=[{"role": "assistant", "content": "returned"}],
                    metrics={"quality": 0.75},
                    info={"workflow_only": True},
                )

        class CapturingGrader(Grader):
            async def grade(self, ctx: GraderContext) -> Any:
                assert ctx.sample is not None
                captured["messages"] = ctx.sample.messages
                captured["metrics"] = ctx.sample.metrics
                captured["extra_fields"] = ctx.sample.extra_fields
                captured["metadata"] = ctx.metadata
                ctx.set_reward(1.0)

        backend = LocalBackend(
            workflow=ReturningWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=CapturingGrader,
            grader_config=GraderConfig(name="test-grader"),
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
            metadata={"input_only": True},
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )

        on_complete.assert_awaited_once()
        on_grader.assert_awaited_once()
        grader_result = on_grader.call_args[0][0]
        assert grader_result.status == RolloutStatus.SUCCESS
        assert grader_result.sample is not None
        assert grader_result.sample.reward == 1.0
        assert captured == {
            "messages": [{"role": "assistant", "content": "returned"}],
            "metrics": {"quality": 0.75},
            "extra_fields": {},
            "metadata": {"input_only": True},
        }

    async def test_grader_callback_reports_failure_without_label(self):
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=StubGrader,
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label=None,  # no label → grader should be skipped
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )
        on_grader.assert_awaited_once()
        grader_result = on_grader.call_args[0][0]
        assert grader_result.status == RolloutStatus.FAILURE

    async def test_grader_runs_with_metadata_only(self):
        """A metadata-only row (no label) still triggers grading."""
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=StubGrader,
            grader_config=GraderConfig(name="test-grader"),
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label=None,
            metadata={"tools": ["search"]},
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )

        on_grader.assert_awaited_once()
        grader_result = on_grader.call_args[0][0]
        assert grader_result.status == RolloutStatus.SUCCESS
        assert grader_result.sample is not None
        assert grader_result.sample.reward == 1.0

    async def test_grader_runs_with_label_only(self):
        """A label-only row (no metadata) still triggers grading."""
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=StubGrader,
            grader_config=GraderConfig(name="test-grader"),
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
            metadata=None,
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )

        on_grader.assert_awaited_once()
        grader_result = on_grader.call_args[0][0]
        assert grader_result.status == RolloutStatus.SUCCESS

    async def test_metadata_reaches_both_contexts(self):
        """Both the workflow ctx and grader ctx receive the request metadata."""
        captured: dict[str, Any] = {}
        metadata = {"tools": ["search"], "difficulty": 3}

        class CapturingWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                from osmosis_ai.rollout.context import get_rollout_context

                captured["workflow_metadata"] = ctx.metadata
                rollout_ctx = get_rollout_context()
                if rollout_ctx:
                    rollout_ctx.set_sample_source(
                        StaticSampleSource([{"role": "assistant", "content": "done"}]),
                    )

        class CapturingGrader(Grader):
            async def grade(self, ctx: GraderContext) -> Any:
                captured["grader_metadata"] = ctx.metadata
                ctx.set_reward(1.0)

        backend = LocalBackend(
            workflow=CapturingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=CapturingGrader,
            grader_config=GraderConfig(name="test-grader"),
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            metadata=metadata,
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )

        assert captured["workflow_metadata"] == metadata
        assert captured["grader_metadata"] == metadata

    async def test_grader_failure_returns_error_result(self):
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=FailingGrader,
            grader_config=GraderConfig(name="test-grader"),
        )
        on_complete = AsyncMock()
        on_grader = AsyncMock()

        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
        )
        await backend.execute(
            request,
            on_workflow_complete=on_complete,
            on_grader_complete=on_grader,
        )

        on_grader.assert_awaited_once()
        grader_result = on_grader.call_args[0][0]
        assert grader_result.status == RolloutStatus.FAILURE
        assert "grading failed" in grader_result.err_message

    def test_init_with_string_reference(self):
        backend = LocalBackend(
            workflow=f"{StubWorkflow.__module__}:{StubWorkflow.__qualname__}",
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        assert backend.workflow_cls is StubWorkflow

    async def test_prompt_passes_through_unchanged(self):
        captured: dict = {}

        class CapturingWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                captured["prompt"] = ctx.prompt

        backend = LocalBackend(
            workflow=CapturingWorkflow,
            workflow_config=AgentWorkflowConfig(name="passthrough"),
        )
        on_complete = AsyncMock()

        original_prompt = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        request = ExecutionRequest(id="r1", prompt=original_prompt)
        await backend.execute(request, on_workflow_complete=on_complete)

        # Byte-for-byte identical: no content-block conversion, no copy-and-mutate.
        assert captured["prompt"] == original_prompt
        assert captured["prompt"][0]["content"] == "you are helpful"


# ---------------------------------------------------------------------------
# Deadlines (agent_timeout_sec / grader_timeout_sec)
# ---------------------------------------------------------------------------


class HangingWorkflow(AgentWorkflow):
    """Runs far longer than any deadline the controller would send."""

    cancelled = False

    async def run(self, ctx: AgentWorkflowContext) -> Any:
        import asyncio

        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            type(self).cancelled = True
            raise


class HangingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        import asyncio

        await asyncio.sleep(30)


class TestDeadlines:
    async def test_workflow_deadline_returns_timeout_result(self):
        HangingWorkflow.cancelled = False
        backend = LocalBackend(
            workflow=HangingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=0.05
        )

        result = await backend.run_workflow(request)

        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.TIMEOUT
        assert "0.05s deadline" in (result.err_message or "")
        # A cooperatively-cancellable workflow actually stops. One that
        # swallows CancelledError or blocks the loop still reports a timeout,
        # but keeps running — that limitation is inherent to asyncio.timeout.
        assert HangingWorkflow.cancelled

    async def test_workflow_deadline_reports_through_the_callback(self):
        backend = LocalBackend(
            workflow=HangingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        on_complete = AsyncMock()
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=0.05
        )

        await backend.execute(request, on_workflow_complete=on_complete)

        # A deadline is a terminal result, not an exception the server has to
        # rescue: the callback still fires and the slot is released.
        on_complete.assert_awaited_once()
        assert on_complete.await_args.args[0].err_category == (
            RolloutErrorCategory.TIMEOUT
        )
        assert backend.limiter.snapshot()["running"] == 0

    async def test_grader_deadline_is_independent_of_the_agent_deadline(self):
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=HangingGrader,
        )
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="x",
            agent_timeout_sec=5.0,
            grader_timeout_sec=0.05,
        )

        workflow_result = await backend.run_workflow(request)
        graded = await backend.run_grader(request, workflow_result)

        assert workflow_result.status == RolloutStatus.SUCCESS
        assert graded.status == RolloutStatus.FAILURE
        assert graded.err_category == RolloutErrorCategory.TIMEOUT
        assert "0.05s deadline" in (graded.err_message or "")

    async def test_no_deadline_runs_unbounded(self):
        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])

        result = await backend.run_workflow(request)

        assert result.status == RolloutStatus.SUCCESS

    async def test_completion_just_inside_the_deadline_succeeds(self):
        class BriefWorkflow(StubWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                import asyncio

                await asyncio.sleep(0.01)
                await super().run(ctx)

        backend = LocalBackend(
            workflow=BriefWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=5.0
        )

        result = await backend.run_workflow(request)

        assert result.status == RolloutStatus.SUCCESS

    async def test_user_timeout_error_keeps_its_own_message(self):
        class ImpatientWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                raise TimeoutError("upstream API gave up")

        backend = LocalBackend(
            workflow=ImpatientWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=30.0
        )

        result = await backend.run_workflow(request)

        # Same wire category, but the deadline did not fire — do not claim it did.
        assert result.err_category == RolloutErrorCategory.TIMEOUT
        assert result.err_message == "upstream API gave up"

    async def test_swallowed_cancellation_is_still_a_timeout(self):
        class CancelSwallowingWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                import asyncio
                import contextlib

                # User code eats the deadline's cancellation and returns anyway.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(30)
                return AgentWorkflowOutput(
                    messages=[{"role": "assistant", "content": "late"}]
                )

        backend = LocalBackend(
            workflow=CancelSwallowingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=0.05
        )

        result = await backend.run_workflow(request)

        # The controller stopped waiting at the deadline; success after it
        # would report a rollout nobody received.
        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.TIMEOUT
        assert "0.05s deadline" in (result.err_message or "")

    async def test_sync_blocking_past_the_deadline_is_still_a_timeout(self):
        class LoopBlockingWorkflow(AgentWorkflow):
            async def run(self, ctx: AgentWorkflowContext) -> Any:
                import time

                # Blocks the event loop, so the timeout callback never runs.
                time.sleep(0.15)
                return AgentWorkflowOutput(
                    messages=[{"role": "assistant", "content": "late"}]
                )

        backend = LocalBackend(
            workflow=LoopBlockingWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=0.05
        )

        result = await backend.run_workflow(request)

        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.TIMEOUT

    async def test_queue_time_consumes_the_workflow_budget(self):
        import asyncio

        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(
                name="test", concurrency=ConcurrencyConfig(max_concurrent=1)
            ),
        )

        async def hog_the_only_slot():
            async with backend.limiter.acquire():
                await asyncio.sleep(0.25)

        hog = asyncio.create_task(hog_the_only_slot())
        await asyncio.sleep(0.05)  # ensure the hog holds the slot first

        on_complete = AsyncMock()
        request = ExecutionRequest(
            id="r1", prompt=[{"role": "user", "content": "hi"}], agent_timeout_sec=0.1
        )
        await backend.execute(request, on_workflow_complete=on_complete)
        await hog

        # The controller's clock covered the ~0.2s queue wait, which exceeded
        # the whole 0.1s budget; reporting success after it would be a lie.
        result = on_complete.await_args.args[0]
        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.TIMEOUT
        assert "queued" in (result.err_message or "")

    async def test_grader_swallowed_cancellation_is_still_a_timeout(self):
        class CancelSwallowingGrader(Grader):
            async def grade(self, ctx: GraderContext) -> Any:
                import asyncio
                import contextlib

                ctx.set_reward(0.8)
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(30)

        backend = LocalBackend(
            workflow=StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
            grader=CancelSwallowingGrader,
        )
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="x",
            grader_timeout_sec=0.05,
        )

        workflow_result = await backend.run_workflow(request)
        graded = await backend.run_grader(request, workflow_result)

        assert graded.status == RolloutStatus.FAILURE
        assert graded.err_category == RolloutErrorCategory.TIMEOUT

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_timeouts_are_rejected_at_validation(self, value):
        # NaN crashed the 3.13 event loop selector outright and +/-inf
        # silently disables enforcement; neither may reach asyncio.timeout.
        prompt = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValidationError):
            ExecutionRequest(id="r1", prompt=prompt, agent_timeout_sec=value)
        with pytest.raises(ValidationError):
            ExecutionRequest(id="r1", prompt=prompt, grader_timeout_sec=value)

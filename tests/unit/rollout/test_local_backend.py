"""Tests for osmosis_ai.rollout.backend.local.backend."""

from typing import Any
from unittest.mock import AsyncMock

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local.backend import (
    LocalBackend,
    _categorize_exception,
)
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
)
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    ExecutionRequest,
    GraderConfig,
    RolloutErrorCategory,
    RolloutStatus,
)

# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class StubWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        # Register a fake agent with the rollout context
        from unittest.mock import MagicMock

        from osmosis_ai.rollout.context import get_rollout_context

        rollout_ctx = get_rollout_context()
        if rollout_ctx:
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "done"}]
            rollout_ctx.register_agent("sample-1", agent)


class FailingWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        raise ValueError("workflow error")


class StubGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        for sample_id in ctx.get_samples():
            ctx.set_sample_reward(sample_id, 1.0)


class FailingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        raise RuntimeError("grading failed")


# ---------------------------------------------------------------------------
# _categorize_exception
# ---------------------------------------------------------------------------


class TestCategorizeException:
    def test_timeout(self):
        assert _categorize_exception(TimeoutError()) == RolloutErrorCategory.TIMEOUT

    def test_value_error(self):
        assert (
            _categorize_exception(ValueError("bad"))
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_type_error(self):
        assert (
            _categorize_exception(TypeError("bad"))
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_assertion_error(self):
        assert (
            _categorize_exception(AssertionError())
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    def test_generic(self):
        assert (
            _categorize_exception(RuntimeError("boom"))
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

    async def test_execute_success_calls_callback(self):
        backend = self._make_backend()
        on_complete = AsyncMock()

        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])
        await backend.execute(request, on_workflow_complete=on_complete)

        on_complete.assert_awaited_once()
        result = on_complete.call_args[0][0]
        assert result.status == RolloutStatus.SUCCESS

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
        # Grader should have assigned reward=1.0
        for sample in grader_result.samples.values():
            assert sample.reward == 1.0

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

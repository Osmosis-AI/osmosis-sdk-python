import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import EnvironmentConfig as HarborEnvironmentConfig
from pydantic import Field

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.harbor.agent_adapter import OsmosisHostedAgent
from osmosis_ai.rollout.backend.harbor.backend import (
    HOSTED_AGENT_IMPORT_PATH,
    INSTALLED_AGENT_IMPORT_PATH,
    HarborBackend,
    PendingTrial,
)
from osmosis_ai.rollout.backend.harbor.workflow_runner import run_workflow
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    HarborAgentWorkflowContext,
    RolloutContext,
    SampleSource,
    get_rollout_context,
)
from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    ExecutionRequest,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)


class StaticSampleSource(SampleSource):
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    async def get_sample(self, name: str) -> RolloutSample:
        return RolloutSample(id=name, messages=self.messages)


class CapturingHarborWorkflow(AgentWorkflow):
    captured_context_type: type[AgentWorkflowContext] | None = None
    captured_environment: Any = None

    async def run(self, ctx: AgentWorkflowContext) -> None:
        CapturingHarborWorkflow.captured_context_type = type(ctx)
        CapturingHarborWorkflow.captured_environment = getattr(ctx, "environment", None)
        rollout_ctx = get_rollout_context()
        assert rollout_ctx is not None
        rollout_ctx.register_sample_source("sample-1", StaticSampleSource(ctx.prompt))


class MutableWorkflowConfig(AgentWorkflowConfig):
    seen: list[str] = Field(default_factory=list)


mutable_workflow_config = MutableWorkflowConfig(name="mutable-workflow")


class MutatingConfigWorkflow(AgentWorkflow):
    seen_snapshots: list[list[str]] = []

    async def run(self, ctx: AgentWorkflowContext) -> None:
        assert ctx.config is not None
        ctx.config.seen.append(ctx.prompt[0]["content"])
        MutatingConfigWorkflow.seen_snapshots.append(list(ctx.config.seen))


class FakeTrialQueue:
    def __init__(self) -> None:
        self.hooks: list[tuple[Any, Any]] = []

    def add_hook(self, event: Any, hook: Any) -> None:
        self.hooks.append((event, hook))


class TestHarborBackend:
    def _backend_for_trial_config(self, tmp_path: Path) -> HarborBackend:
        backend = HarborBackend.__new__(HarborBackend)
        backend.workflow_path = f"{__name__}:CapturingHarborWorkflow"
        backend.workflow_config_path = None
        backend.grader_path = None
        backend.grader_config_path = None
        backend.grading = False
        backend.trials_dir = tmp_path / "trials"
        backend.environment_config = HarborEnvironmentConfig()
        return backend

    def test_build_trial_config_uses_installed_agent_by_default(self, tmp_path: Path):
        task_dir = tmp_path / "task"
        user_code_dir = tmp_path / "user-code"
        task_dir.mkdir()
        user_code_dir.mkdir()
        backend = HarborBackend(
            orchestrator=FakeTrialQueue(),  # type: ignore[arg-type]
            task_dir=task_dir,
            user_code_dir=user_code_dir,
            workflow=CapturingHarborWorkflow,
            prebuild_local_image=False,
            symlink_environment=False,
            trials_dir=tmp_path / "trials",
        )

        config = backend.build_trial_config(
            tmp_path,
            ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}]),
        )

        assert backend.workflow_execution_mode == "installed"
        assert config.agent.import_path == INSTALLED_AGENT_IMPORT_PATH
        assert config.agent.kwargs == {
            "rollout_config_path": str(tmp_path / "rollout_config.json")
        }

    def test_build_trial_config_can_use_hosted_agent(self, tmp_path: Path):
        backend = self._backend_for_trial_config(tmp_path)
        backend.workflow_execution_mode = "hosted"

        config = backend.build_trial_config(
            tmp_path,
            ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}]),
        )

        assert config.agent.import_path == HOSTED_AGENT_IMPORT_PATH

    async def test_run_workflow_deepcopies_workflow_config_between_runs(self):
        mutable_workflow_config.seen.clear()
        MutatingConfigWorkflow.seen_snapshots.clear()
        config = {
            "workflow": f"{__name__}:MutatingConfigWorkflow",
            "workflow_config": f"{__name__}:mutable_workflow_config",
        }

        first = await run_workflow(
            config,
            [{"role": "user", "content": "first"}],
        )
        second = await run_workflow(
            config,
            [{"role": "user", "content": "second"}],
        )

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert mutable_workflow_config.seen == []
        assert MutatingConfigWorkflow.seen_snapshots == [["first"], ["second"]]

    def test_hosted_rollout_config_keeps_local_controller_url(
        self, tmp_path: Path, monkeypatch
    ):
        backend = self._backend_for_trial_config(tmp_path)
        backend.workflow_execution_mode = "hosted"
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Darwin",
        )

        with RolloutContext(
            chat_completions_url="http://127.0.0.1:8000",
            api_key="key",
            rollout_id="rollout-1",
        ):
            config = backend.build_rollout_config(
                ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])
            )

        assert config["chat_completions_url"] == "http://127.0.0.1:8000"

    def test_installed_rollout_config_rewrites_local_controller_url_for_docker(
        self, tmp_path: Path, monkeypatch
    ):
        backend = self._backend_for_trial_config(tmp_path)
        backend.workflow_execution_mode = "installed"
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Darwin",
        )

        with RolloutContext(
            chat_completions_url="http://127.0.0.1:8000",
            api_key="key",
            rollout_id="rollout-1",
        ):
            config = backend.build_rollout_config(
                ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "hi"}])
            )

        assert config["chat_completions_url"] == "http://host.docker.internal:8000"

    async def test_hosted_agent_runs_workflow_on_host_with_harbor_environment(
        self, tmp_path: Path
    ):
        config_path = tmp_path / "rollout_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "id": "r1",
                    "workflow": f"{__name__}:CapturingHarborWorkflow",
                }
            )
        )
        environment = object()
        context = AgentContext()
        agent = OsmosisHostedAgent(
            logs_dir=tmp_path,
            rollout_config_path=str(config_path),
        )

        await agent.run(
            json.dumps([{"role": "user", "content": "hi"}]),
            environment,
            context,
        )

        assert (
            CapturingHarborWorkflow.captured_context_type is HarborAgentWorkflowContext
        )
        assert CapturingHarborWorkflow.captured_environment is environment
        assert context.metadata is not None
        assert context.metadata["status"] == "success"
        assert set(context.metadata["samples"]) == {"sample-1"}
        assert (tmp_path / "samples.json").exists()

    async def test_empty_verifier_rewards_logs_and_returns_validation_failure(
        self, caplog
    ):
        backend = HarborBackend.__new__(HarborBackend)
        backend.pending = {}
        backend.cleanup_successful_trials = False

        on_workflow = AsyncMock()
        on_grader = AsyncMock()
        pending = PendingTrial(on_workflow, on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        event = SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-r1"),
            result=SimpleNamespace(
                agent_result=SimpleNamespace(
                    metadata={
                        "status": "success",
                        "samples": {
                            "sample-1": RolloutSample(
                                id="sample-1", messages=[]
                            ).model_dump()
                        },
                    }
                ),
                verifier_result=SimpleNamespace(rewards={}),
                exception_info=None,
            ),
        )

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(event)

        on_grader.assert_awaited_once()
        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR
        assert "Harbor verifier returned empty rewards for rollout r1" in caplog.text

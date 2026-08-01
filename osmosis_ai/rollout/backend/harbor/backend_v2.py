"""Harbor execution backend v2. Host-side only.

``agent=`` selects the track: a registered native agent name ("terminus-2",
"mini-swe-agent") runs Harbor's own agent with the rollout endpoint injected;
an AgentWorkflow class (or "module:Class" path) is packaged into a wheel and
installed in the task container at trial start. ``grader=None`` makes the
task's own tests the reward source; a Grader class is delivered as the
verifier instead. Task images stay pure task environments either way; with
the pinned harbor 0.20.0 each trial builds its own compose-named tag (fast
via docker's layer cache, removed at trial teardown), and newer harbor
releases share one content-addressed hb__ image across trials.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
)
from harbor.models.trial.config import (
    EnvironmentConfig as HarborEnvironmentConfig,
)
from harbor.models.trial.config import (
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.backend.harbor.artifacts import (
    merge_grader_artifacts,
    relocate_trial_artifacts,
)
from osmosis_ai.rollout.backend.harbor.backend import (
    TRIAL_NAME_PREFIX,
    PendingTrial,
    apply_managed_skypilot_placement,
    ensure_import_path,
    get_agent_metadata,
    log_trial_exception,
    parse_rollout_id,
    rewrite_url_for_docker,
    uses_local_docker_runtime,
)
from osmosis_ai.rollout.backend.harbor.bundling import resolve_backend_bundle
from osmosis_ai.rollout.backend.harbor.native_agents import (
    native_agent_config,
    native_binding,
)
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask, TaskMode
from osmosis_ai.rollout.utils.timing import PhaseTimer
from osmosis_ai.rollout.container.files import ContainerInput, ContainerResult
from osmosis_ai.rollout.container.trajectories import messages_from_trajectory
from osmosis_ai.rollout.context import get_rollout_context
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.file_artifacts import default_artifact_root
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward

logger: logging.Logger = logging.getLogger(__name__)

HARNESS_AGENT_IMPORT_PATH = (
    "osmosis_ai.rollout.backend.harbor.harness_agent:OsmosisHarnessInstalledAgent"
)


class HarborBackendV2(ExecutionBackend):
    def __init__(
        self,
        *,
        orchestrator: TrialQueue,
        tasks_dir: Path,
        agent: str | type | None = None,
        task_mode: TaskMode | str = TaskMode.TEMPLATE,
        model_name: str = "openai/osmosis-rollout",
        grader: type | str | None = None,
        workflow_config: Any = None,
        grader_config: Any = None,
        code_dir: Path | None = None,
        bundle: Path | None = None,
        environment_config: HarborEnvironmentConfig | None = None,
        trials_dir: Path | None = None,
        cleanup_successful_trials: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        self.tasks_dir = Path(tasks_dir)
        self.task_mode = TaskMode(task_mode)
        self.model_name = model_name
        self.agent = agent
        self.native = native_binding(agent)
        self.bundle = resolve_backend_bundle(
            agent=agent,
            grader=grader,
            workflow_config=workflow_config,
            grader_config=grader_config,
            code_dir=code_dir,
            bundle=bundle,
            native=self.native is not None,
        )
        self.environment_config = apply_managed_skypilot_placement(
            environment_config or HarborEnvironmentConfig()
        )

        root = Path(f"/tmp/osmosis-harbor-{self.tasks_dir.name}")
        self.rollouts_dir = root / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = trials_dir or root / "trials"
        self.artifact_root = default_artifact_root()
        self.cleanup_successful_trials = cleanup_successful_trials
        self.pending: dict[str, PendingTrial] = {}
        self.running = 0
        self.timer = PhaseTimer()

        orchestrator.add_hook(TrialEvent.START, self.on_trial_started)
        orchestrator.add_hook(TrialEvent.ENVIRONMENT_START, self.mark_phase("environment"))
        orchestrator.add_hook(TrialEvent.AGENT_START, self.mark_phase("agent"))
        orchestrator.add_hook(TrialEvent.VERIFICATION_START, self.on_verification_start)
        orchestrator.add_hook(TrialEvent.END, self.on_trial_end)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "harbor-v2",
            "agent": self.agent
            if isinstance(self.agent, str)
            else ensure_import_path(self.agent),
            "in_flight": len(self.pending),
            "running": self.running,
            "queued": max(0, len(self.pending) - self.running),
        }

    def select_task(self, request: ExecutionRequest) -> HarborTask:
        if self.task_mode == TaskMode.TEMPLATE:
            return HarborTask(self.tasks_dir)
        task_id = (request.metadata or {}).get("harbor_task_id")
        if not task_id:
            raise ValueError("dataset mode requires metadata['harbor_task_id']")
        return HarborTask.from_dataset(self.tasks_dir, task_id)

    def build_input(self, request: ExecutionRequest) -> ContainerInput:
        container_input = ContainerInput(
            rollout_id=request.id,
            # Dataset tasks carry their own instruction.md; row prompts only
            # drive template mode.
            prompt=(request.prompt or []) if self.task_mode == TaskMode.TEMPLATE else [],
            label=request.label,
            metadata=request.metadata,
        )
        ctx = get_rollout_context()
        if ctx:
            url = ctx.chat_completions_url or ""
            if url and uses_local_docker_runtime(self.environment_config):
                url = rewrite_url_for_docker(url)
            container_input.chat_completions_url = url
            container_input.api_key = ctx.api_key
        return container_input

    def build_agent_config(
        self, task_dir: Path, container_input: ContainerInput
    ) -> HarborAgentConfig:
        if self.native is not None:
            return native_agent_config(
                self.agent,
                self.native,
                self.model_name,
                container_input.chat_completions_url,
                container_input.api_key or "dummy",
            )
        return HarborAgentConfig(
            import_path=HARNESS_AGENT_IMPORT_PATH,
            kwargs={
                "bundle_path": str(self.bundle.wheel),
                "agent_script": self.bundle.agent_script,
                "input_path": str(task_dir / "container_input.json"),
            },
        )

    def build_trial_config(
        self, task_dir: Path, request: ExecutionRequest, container_input: ContainerInput
    ) -> TrialConfig:
        agent_config = self.build_agent_config(task_dir, container_input)
        if request.agent_timeout_sec is not None:
            agent_config.override_timeout_sec = request.agent_timeout_sec

        verifier_config = VerifierConfig(disable=not (task_dir / "tests").is_dir())
        if request.grader_timeout_sec is not None:
            verifier_config.override_timeout_sec = request.grader_timeout_sec

        return TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name=f"{TRIAL_NAME_PREFIX}{request.id}",
            trials_dir=self.trials_dir,
            agent=agent_config,
            environment=self.environment_config.model_copy(deep=True),
            verifier=verifier_config,
        )

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        pending = PendingTrial(on_workflow_complete, on_grader_complete)
        self.pending[request.id] = pending
        try:
            container_input = self.build_input(request)
            task_dir = self.select_task(request).materialize(
                self.rollouts_dir / request.id,
                container_input,
                grader_script=self.bundle.grader_script if self.bundle else None,
                grader_wheel=self.bundle.wheel if self.bundle and self.native else None,
            )
            await self.orchestrator.submit(
                self.build_trial_config(task_dir, request, container_input)
            )
            await pending.done
        except Exception as e:
            self.pending.pop(request.id, None)
            logger.error("Failed trial %s: %s", request.id, e)
            await on_workflow_complete(
                ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    err_message=str(e),
                    err_category=RolloutErrorCategory.AGENT_ERROR,
                )
            )

    async def on_trial_started(self, event: TrialHookEvent) -> None:
        self.running += 1
        self.timer.start(parse_rollout_id(event))

    def mark_phase(self, phase: str):
        async def hook(event: TrialHookEvent) -> None:
            self.timer.mark(parse_rollout_id(event), phase)

        return hook

    def container_result(self, event: TrialHookEvent) -> ContainerResult | None:
        metadata = get_agent_metadata(event)
        if not metadata:
            return None
        try:
            return ContainerResult.model_validate(metadata)
        except ValueError:
            return None

    def native_sample(self, rollout_id: str) -> RolloutSample | None:
        trajectory_path = (
            self.trials_dir
            / f"{TRIAL_NAME_PREFIX}{rollout_id}"
            / "agent"
            / "trajectory.json"
        )
        if not trajectory_path.exists():
            return None
        try:
            messages = messages_from_trajectory(json.loads(trajectory_path.read_text()))
        except (ValueError, OSError):
            return None
        if not messages:
            return None
        return RolloutSample(id="default", messages=messages)

    def primary_sample(
        self, event: TrialHookEvent, rollout_id: str
    ) -> RolloutSample | None:
        if self.native is not None:
            return self.native_sample(rollout_id)
        result = self.container_result(event)
        output = result.output if result else None
        messages = output.primary_messages() if output else None
        if messages is None:
            return None
        return RolloutSample(id="default", messages=messages, metrics=output.metrics)

    def agent_succeeded(self, event: TrialHookEvent) -> tuple[bool, str | None]:
        if event.result and event.result.exception_info:
            return False, event.result.exception_info.exception_message
        if self.native is not None:
            return True, None
        result = self.container_result(event)
        if result is not None and result.status == RolloutStatus.SUCCESS:
            return True, None
        return False, result.err_message if result else "Unknown error"

    async def on_verification_start(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        self.timer.mark(rollout_id, "verification")
        pending = self.pending.get(rollout_id)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        succeeded, err_message = self.agent_succeeded(event)
        if succeeded:
            outcome = ExecutionResult(
                status=RolloutStatus.SUCCESS,
                sample=self.primary_sample(event, rollout_id),
            )
        else:
            outcome = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=err_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )

        pending.workflow_complete_called = True
        await pending.on_workflow_complete(outcome)

    def grader_outcome(self, event: TrialHookEvent, rollout_id: str) -> ExecutionResult:
        sample = self.primary_sample(event, rollout_id)

        if event.result and event.result.verifier_result:
            rewards = event.result.verifier_result.rewards or {}
            reward = rewards.get("reward")
            if sample is not None and reward is not None:
                sample.reward = float(reward)
            try:
                validate_sample_has_reward(sample)
            except ValueError as e:
                logger.warning(
                    "Verifier rewards for rollout %s missing 'reward': %s",
                    rollout_id,
                    e,
                )
                return ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    sample=sample,
                    err_message=str(e),
                    err_category=RolloutErrorCategory.VALIDATION_ERROR,
                )
            return ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample)

        if event.result and event.result.exception_info:
            err = event.result.exception_info
            log_trial_exception(rollout_id, err, phase="during grading")
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                err_message=err.exception_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )
        return ExecutionResult(status=RolloutStatus.FAILURE, sample=sample)

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        self.running = max(0, self.running - 1)
        pending = self.pending.pop(rollout_id, None)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        try:
            merge_grader_artifacts(self.trials_dir, rollout_id)
            delete_trial = bool(
                self.cleanup_successful_trials
                and event.result
                and not event.result.exception_info
            )
            grader_result = (
                self.grader_outcome(event, rollout_id)
                if pending.on_grader_complete
                else None
            )
            relocated = relocate_trial_artifacts(
                self.trials_dir, self.artifact_root, rollout_id, move=delete_trial
            )

            if not pending.workflow_complete_called:
                if event.result and event.result.exception_info:
                    err = event.result.exception_info
                    log_trial_exception(
                        rollout_id, err, phase="before the agent completed"
                    )
                    message = err.exception_message
                else:
                    message = "Trial ended before agent completed"
                await pending.on_workflow_complete(
                    ExecutionResult(
                        status=RolloutStatus.FAILURE,
                        err_message=message,
                        err_category=RolloutErrorCategory.AGENT_ERROR,
                    )
                )

            if pending.on_grader_complete and grader_result is not None:
                await pending.on_grader_complete(grader_result)

            if self.cleanup_successful_trials:
                shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
            if delete_trial and relocated:
                trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
                shutil.rmtree(trial_dir, ignore_errors=True)
        finally:
            timings = self.timer.finish(rollout_id)
            if timings:
                logger.info("rollout %s phase timings: %s", rollout_id, timings)
            if not pending.done.done():
                pending.done.set_result(None)

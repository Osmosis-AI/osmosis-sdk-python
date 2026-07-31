"""Execution backend that runs bundled workflows inside Harbor containers.

Code reaches the container as a wheel installed at trial setup (OsmosisHarnessInstalledAgent),
so task images stay pure task environments and Harbor's content-addressed
image cache applies unchanged. The host <-> container contract lives in
contract.py; task shaping lives in tasks.py.
"""

from __future__ import annotations

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

from osmosis_ai.packaging import build_bundle, inspect_bundle, project_dir_for
from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
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
from osmosis_ai.rollout.utils.imports import resolve_object
from osmosis_ai.rollout.container.files import ContainerInput, ContainerResult
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask, TaskMode
from osmosis_ai.rollout.context import get_rollout_context
from osmosis_ai.rollout.types import (
    RolloutSample,
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
    copy_artifact_tree,
    default_artifact_root,
)
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward

logger: logging.Logger = logging.getLogger(__name__)

BUNDLE_AGENT_IMPORT_PATH = (
    "osmosis_ai.rollout.backend.harbor.harness_agent:OsmosisHarnessInstalledAgent"
)


class HarborBackendV2(ExecutionBackend):
    def __init__(
        self,
        *,
        orchestrator: TrialQueue,
        tasks_dir: Path,
        workflow: type | str | None = None,
        grader: type | str | None = None,
        workflow_config: Any = None,
        grader_config: Any = None,
        code_dir: Path | None = None,
        bundle: Path | None = None,
        task_mode: TaskMode | str = TaskMode.TEMPLATE,
        environment_config: HarborEnvironmentConfig | None = None,
        trials_dir: Path | None = None,
        cleanup_successful_trials: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        if bundle is None:
            if workflow is None:
                raise ValueError("pass workflow (or a prebuilt bundle)")
            resolved = resolve_object(workflow)
            bundle = build_bundle(
                code_dir or project_dir_for(resolved),
                workflow=ensure_import_path(workflow),
                grader=ensure_import_path(grader) if grader else None,
                workflow_config=(
                    ensure_import_path(workflow_config) if workflow_config else None
                ),
                grader_config=(
                    ensure_import_path(grader_config) if grader_config else None
                ),
            )
        self.bundle = inspect_bundle(Path(bundle))
        self.environment_config = apply_managed_skypilot_placement(
            environment_config or HarborEnvironmentConfig()
        )

        root = Path(f"/tmp/osmosis-harbor-{Path(tasks_dir).name}")
        self.rollouts_dir = root / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = trials_dir or root / "trials"
        self.artifact_root = default_artifact_root()
        self.tasks_dir = Path(tasks_dir)
        self.task_mode = TaskMode(task_mode)
        self.cleanup_successful_trials = cleanup_successful_trials
        self.pending: dict[str, PendingTrial] = {}

        orchestrator.add_hook(TrialEvent.VERIFICATION_START, self.on_verification_start)
        orchestrator.add_hook(TrialEvent.END, self.on_trial_end)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "harbor-v2",
            "pending_trials": len(self.pending),
        }

    def build_input(self, request: ExecutionRequest) -> ContainerInput:
        container_input = ContainerInput(
            rollout_id=request.id,
            prompt=request.prompt or [],
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

    def build_trial_config(
        self, task_dir: Path, request: ExecutionRequest
    ) -> TrialConfig:
        agent_config = HarborAgentConfig(
            import_path=BUNDLE_AGENT_IMPORT_PATH,
            kwargs={
                "bundle_path": str(self.bundle.wheel),
                "agent_script": self.bundle.agent_script,
                "input_path": str(task_dir / "container_input.json"),
            },
        )
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
            environment=self.environment_config,
            verifier=verifier_config,
        )

    def select_task(self, request: ExecutionRequest) -> HarborTask:
        if self.task_mode == TaskMode.TEMPLATE:
            return HarborTask(self.tasks_dir)
        task_id = (request.metadata or {}).get("harbor_task_id")
        if not task_id:
            raise ValueError("dataset mode requires metadata['harbor_task_id']")
        return HarborTask.from_dataset(self.tasks_dir, task_id)

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        pending = PendingTrial(on_workflow_complete, on_grader_complete)
        self.pending[request.id] = pending
        try:
            task_dir = self.select_task(request).materialize(
                self.rollouts_dir / request.id,
                self.build_input(request),
                self.bundle.grader_script,
            )
            await self.orchestrator.submit(self.build_trial_config(task_dir, request))
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

    def container_result(self, event: TrialHookEvent) -> ContainerResult | None:
        metadata = get_agent_metadata(event)
        if not metadata:
            return None
        try:
            return ContainerResult.model_validate(metadata)
        except ValueError:
            return None

    def primary_sample(self, result: ContainerResult | None) -> RolloutSample | None:
        output = result.output if result else None
        messages = output.primary_messages() if output else None
        if messages is None:
            return None
        return RolloutSample(id="default", messages=messages, metrics=output.metrics)

    async def on_verification_start(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        pending = self.pending.get(rollout_id)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        result = self.container_result(event)
        if result is not None and result.status == RolloutStatus.SUCCESS:
            outcome = ExecutionResult(
                status=RolloutStatus.SUCCESS, sample=self.primary_sample(result)
            )
        elif event.result and event.result.exception_info:
            err = event.result.exception_info
            outcome = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=err.exception_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )
        else:
            outcome = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=result.err_message if result else "Unknown error",
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )

        pending.workflow_complete_called = True
        await pending.on_workflow_complete(outcome)

    def grader_outcome(self, event: TrialHookEvent, rollout_id: str) -> ExecutionResult:
        sample = self.primary_sample(self.container_result(event))

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
        pending = self.pending.pop(rollout_id, None)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        # pending.done must resolve no matter what fails in here, or the
        # rollout's caller waits forever.
        try:
            self.merge_grader_artifacts(rollout_id)
            delete_trial = bool(
                self.cleanup_successful_trials
                and event.result
                and not event.result.exception_info
            )
            relocated = self.relocate_trial_artifacts(rollout_id, move=delete_trial)

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

            if pending.on_grader_complete:
                await pending.on_grader_complete(self.grader_outcome(event, rollout_id))

            if self.cleanup_successful_trials:
                shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
            if delete_trial and relocated:
                trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
                shutil.rmtree(trial_dir, ignore_errors=True)
        finally:
            if not pending.done.done():
                pending.done.set_result(None)

    def relocate_trial_artifacts(self, rollout_id: str, *, move: bool) -> bool:
        source_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}" / "artifacts"
        if not source_dir.is_dir():
            return True
        try:
            copy_artifact_tree(
                source_dir,
                self.artifact_root / rollout_id / "artifacts",
                destination_root=self.artifact_root,
                replace_destination=True,
            )
            if move:
                shutil.rmtree(source_dir)
        except Exception:
            logger.warning(
                "Failed to relocate trial artifacts for rollout %s (best-effort)",
                rollout_id,
                exc_info=True,
            )
            return False
        return True

    def merge_grader_artifacts(self, rollout_id: str) -> None:
        trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
        source_dir = trial_dir / "verifier" / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        if not source_dir.is_dir():
            return
        try:
            copy_artifact_tree(
                source_dir,
                trial_dir / "artifacts" / HARBOR_ARTIFACTS_DIR.relative_to("/"),
                destination_root=self.trials_dir,
            )
        except Exception:
            logger.warning(
                "Failed to merge grader artifacts for rollout %s (best-effort)",
                rollout_id,
                exc_info=True,
            )

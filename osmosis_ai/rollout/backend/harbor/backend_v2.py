"""Harbor execution backend v2. Host-side only.

``agent=`` selects the track: a registered native agent name ("terminus-2",
"mini-swe-agent", "oracle") runs Harbor's own agent with the rollout endpoint
injected; an AgentWorkflow class (or "module:Class" path) is packaged into a
wheel and installed in the task container at trial start. ``grader=None``
makes the task's own tests the reward source; a Grader class is delivered as
the verifier instead.

Tasks come from ``tasks_dir`` (template or dataset mode), or per rollout via
``metadata["harbor_task"]``: a local path, a registry package "org/name[@ref]",
or a git checkout (with ``metadata["git_url"]`` and, ideally, a pinned
``metadata["git_commit_id"]``). ``metadata["harbor_model"]`` overrides the
model per rollout.

Task images stay pure task environments either way; with the pinned harbor
0.20.0 each trial builds its own compose-named tag (fast via docker's layer
cache, removed at trial teardown), and newer harbor releases share one
content-addressed hb__ image across trials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import traceback
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from harbor.models.trajectories import Trajectory
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
from harbor.tasks.client import TaskClient
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
from osmosis_ai.rollout.backend.harbor.diagnostics import (
    diagnostic_payload,
    failure_phase,
    redact_secrets,
    trial_metrics,
    trial_timings,
)
from osmosis_ai.rollout.backend.harbor.native_agents import (
    native_agent_config,
    native_binding,
)
from osmosis_ai.rollout.backend.harbor.tasks import (
    HarborTask,
    TaskMode,
    parse_task_ref,
)
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
PREWARM_PREFIX = "prewarm-"


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
        patch_dockerfile_with_sdk: bool | None = None,
        agent_setup_timeout_sec: float | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.tasks_dir = Path(tasks_dir)
        self.task_mode = TaskMode(task_mode)
        self.model_name = model_name
        self.agent = agent
        self.agent_setup_timeout_sec = agent_setup_timeout_sec
        self.native = native_binding(agent)
        if self.native and not self.native.trainable:
            logger.warning(
                "native agent %r emits no model trajectory; use it to validate "
                "datasets and verifiers, never for training",
                agent,
            )
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
        if patch_dockerfile_with_sdk and self.bundle is None:
            raise ValueError("patch_dockerfile_with_sdk requires a bundle")
        if patch_dockerfile_with_sdk is None:
            patch_dockerfile_with_sdk = self.bundle is not None
        # The bundle's declared dependencies (stable) are pre-installed into the
        # task image; the bundle itself (volatile user code) installs per trial.
        self.sdk_requirements = (
            self.bundle.requirements if patch_dockerfile_with_sdk else None
        )

        root = Path(f"/tmp/osmosis-harbor-{self.tasks_dir.name}")
        self.rollouts_dir = root / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = trials_dir or root / "trials"
        self.artifact_root = default_artifact_root()
        self.cleanup_successful_trials = cleanup_successful_trials
        self.pending: dict[str, PendingTrial] = {}
        self.fetch_locks: dict[str, asyncio.Lock] = {}
        self.running = 0

        orchestrator.add_hook(TrialEvent.START, self.on_trial_started)
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

    async def resolve_task(self, request: ExecutionRequest) -> HarborTask:
        metadata = request.metadata or {}
        if ref := metadata.get("harbor_task"):
            return await self.fetch_task(str(ref), metadata)
        return self.select_task(request)

    def select_task(self, request: ExecutionRequest) -> HarborTask:
        if self.task_mode == TaskMode.TEMPLATE:
            return HarborTask(self.tasks_dir)
        task_id = (request.metadata or {}).get("harbor_task_id")
        if not task_id:
            raise ValueError(
                "dataset mode requires metadata['harbor_task_id'] or "
                "metadata['harbor_task']"
            )
        return HarborTask.from_dataset(self.tasks_dir, task_id)

    async def fetch_task(self, ref: str, metadata: dict[str, Any]) -> HarborTask:
        """Download (or reuse from cache) a git/package/local task ref."""
        lock = self.fetch_locks.setdefault(ref, asyncio.Lock())
        async with lock:
            batch = await TaskClient().download_tasks([parse_task_ref(ref, metadata)])
        return HarborTask(batch.paths[0])

    def build_input(self, request: ExecutionRequest) -> ContainerInput:
        container_input = ContainerInput(
            rollout_id=request.id,
            # Dataset tasks carry their own instruction.md; row prompts only
            # drive template mode.
            prompt=(request.prompt or [])
            if self.task_mode == TaskMode.TEMPLATE
            else [],
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
            model = (container_input.metadata or {}).get(
                "harbor_model"
            ) or self.model_name
            return native_agent_config(
                self.agent,
                self.native,
                model,
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
        if self.agent_setup_timeout_sec is not None:
            agent_config.override_setup_timeout_sec = self.agent_setup_timeout_sec

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

    def materialize_task(
        self, task: HarborTask, rollout_id: str, container_input: ContainerInput
    ) -> Path:
        return task.materialize(
            self.rollouts_dir / rollout_id,
            container_input,
            grader_script=self.bundle.grader_script if self.bundle else None,
            grader_wheel=self.bundle.wheel if self.bundle and self.native else None,
            sdk_requirements=self.sdk_requirements,
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
            pending.api_key = container_input.api_key
            task = await self.resolve_task(request)
            task_dir = self.materialize_task(task, request.id, container_input)
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
                    extra_fields=diagnostic_payload(
                        phase="setup",
                        category=RolloutErrorCategory.AGENT_ERROR,
                        exception_type=type(e).__name__,
                        timings={},
                    ),
                )
            )

    async def prewarm(self, task_ids: Sequence[str] | None = None) -> None:
        """Build every task image and run agent setup before serving rollouts.

        Prewarm trials are install-only, carry no rollout credentials, and
        report all failing tasks together.
        """
        if self.task_mode == TaskMode.DATASET:
            if not task_ids:
                raise ValueError("dataset mode prewarm requires task ids")
            tasks = [HarborTask.from_dataset(self.tasks_dir, t) for t in task_ids]
        else:
            tasks = [HarborTask(self.tasks_dir)]

        configs = [self.prewarm_trial_config(task) for task in tasks]
        logger.info("Prewarming %d harbor task(s)", len(configs))
        outcomes = await asyncio.gather(
            *(self.orchestrator.submit(config) for config in configs),
            return_exceptions=True,
        )

        failures = []
        for config, outcome in zip(configs, outcomes, strict=True):
            label = config.task.path.name
            if isinstance(outcome, BaseException):
                failures.append(f"{label}: {type(outcome).__name__}: {outcome}")
            elif getattr(outcome, "exception_info", None) is not None:
                failures.append(f"{label}: {outcome.exception_info.exception_type}")
        if failures:
            raise RuntimeError(
                f"prewarm failed for {len(failures)} of {len(configs)} task(s):\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )
        logger.info("Prewarmed %d harbor task(s)", len(configs))

    def prewarm_trial_config(self, task: HarborTask) -> TrialConfig:
        rollout_id = f"{PREWARM_PREFIX}{uuid.uuid4().hex[:8]}"
        container_input = ContainerInput(
            rollout_id=rollout_id,
            prompt=[{"role": "user", "content": "prewarm"}]
            if self.task_mode == TaskMode.TEMPLATE
            else [],
        )
        task_dir = self.materialize_task(task, rollout_id, container_input)
        config = self.build_trial_config(
            task_dir, ExecutionRequest(id=rollout_id, prompt=[]), container_input
        )
        config.install_only = True
        config.verifier.disable = True
        return config

    def prewarm_lifespan(
        self, task_ids: Sequence[str] | None = None
    ) -> Callable[[object], AbstractAsyncContextManager[None]]:
        """An ASGI lifespan that prewarms before the server accepts traffic."""

        @asynccontextmanager
        async def lifespan(app: object) -> AsyncIterator[None]:
            await self.prewarm(task_ids)
            yield

        return lifespan

    async def try_callback(
        self,
        callback: ResultCallback,
        result: ExecutionResult,
        rollout_id: str,
        label: str,
    ) -> bool:
        """Callback delivery failures must never abort trial archival."""
        try:
            await callback(result)
            return True
        except Exception:
            logger.error(
                "%s callback for rollout %s failed: %s",
                label,
                rollout_id,
                traceback.format_exc(),
            )
            return False

    async def on_trial_started(self, event: TrialHookEvent) -> None:
        self.running += 1

    def event_diagnostics(
        self, event: TrialHookEvent, category: RolloutErrorCategory | None = None
    ) -> dict[str, Any]:
        err = event.result.exception_info if event.result else None
        return diagnostic_payload(
            phase=failure_phase(event.result),
            category=category,
            exception_type=err.exception_type if err else None,
            timings=trial_timings(event.result),
        )

    def container_result(self, event: TrialHookEvent) -> ContainerResult | None:
        metadata = get_agent_metadata(event)
        if not metadata:
            return None
        try:
            return ContainerResult.model_validate(metadata)
        except ValueError:
            return None

    def native_sample(
        self, event: TrialHookEvent, rollout_id: str, pending: PendingTrial
    ) -> RolloutSample | None:
        trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
        paths = [p for p in (trial_dir / "agent" / "trajectory.json",) if p.is_file()]
        paths += sorted(trial_dir.glob("steps/*/agent/trajectory.json"))
        if not paths:
            return None
        if len(paths) > 1:
            logger.warning(
                "rollout %s emitted %d trajectory documents; preserving the "
                "trial instead of fabricating a merged trajectory",
                rollout_id,
                len(paths),
            )
            pending.preserve_trial = True
            return None
        try:
            trajectory = Trajectory.model_validate(json.loads(paths[0].read_text()))
        except Exception:
            logger.warning(
                "rollout %s emitted an invalid ATIF trajectory; preserving the trial",
                rollout_id,
                exc_info=True,
            )
            pending.preserve_trial = True
            return None
        if trajectory.agent.extra:
            trajectory.agent.extra = redact_secrets(
                trajectory.agent.extra, pending.api_key
            )
        messages = messages_from_trajectory(trajectory.to_json_dict(exclude_none=True))
        if not messages:
            return None
        return RolloutSample(messages=messages, metrics=trial_metrics(event.result))

    def primary_sample(
        self, event: TrialHookEvent, rollout_id: str, pending: PendingTrial
    ) -> RolloutSample | None:
        if self.native is not None:
            return self.native_sample(event, rollout_id, pending)
        result = self.container_result(event)
        output = result.output if result else None
        messages = output.primary_messages() if output else None
        if messages is None:
            return None
        return RolloutSample(
            messages=messages,
            metrics={**trial_metrics(event.result), **output.metrics},
        )

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
        pending = self.pending.get(rollout_id)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        succeeded, err_message = self.agent_succeeded(event)
        if succeeded:
            outcome = ExecutionResult(
                status=RolloutStatus.SUCCESS,
                sample=self.primary_sample(event, rollout_id, pending),
                extra_fields=self.event_diagnostics(event),
            )
        else:
            outcome = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=err_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
                extra_fields=self.event_diagnostics(
                    event, RolloutErrorCategory.AGENT_ERROR
                ),
            )

        pending.workflow_complete_called = await self.try_callback(
            pending.on_workflow_complete, outcome, rollout_id, "workflow"
        )

    def grader_outcome(
        self, event: TrialHookEvent, rollout_id: str, pending: PendingTrial
    ) -> ExecutionResult:
        sample = self.primary_sample(event, rollout_id, pending)

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
                    extra_fields=self.event_diagnostics(
                        event, RolloutErrorCategory.VALIDATION_ERROR
                    ),
                )
            return ExecutionResult(
                status=RolloutStatus.SUCCESS,
                sample=sample,
                extra_fields=self.event_diagnostics(event),
            )

        if event.result and event.result.exception_info:
            err = event.result.exception_info
            log_trial_exception(rollout_id, err, phase="during grading")
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                err_message=err.exception_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
                extra_fields=self.event_diagnostics(
                    event, RolloutErrorCategory.AGENT_ERROR
                ),
            )
        return ExecutionResult(
            status=RolloutStatus.FAILURE,
            sample=sample,
            extra_fields=self.event_diagnostics(
                event, RolloutErrorCategory.AGENT_ERROR
            ),
        )

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        self.running = max(0, self.running - 1)
        if rollout_id.startswith(PREWARM_PREFIX):
            if event.result and not event.result.exception_info:
                trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
                shutil.rmtree(trial_dir, ignore_errors=True)
                shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
            return

        pending = self.pending.pop(rollout_id, None)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        try:
            merge_grader_artifacts(self.trials_dir, rollout_id)
            grader_result = (
                self.grader_outcome(event, rollout_id, pending)
                if pending.on_grader_complete
                else None
            )
            delete_trial = bool(
                self.cleanup_successful_trials
                and event.result
                and not event.result.exception_info
                and not pending.preserve_trial
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
                pending.workflow_complete_called = await self.try_callback(
                    pending.on_workflow_complete,
                    ExecutionResult(
                        status=RolloutStatus.FAILURE,
                        err_message=message,
                        err_category=RolloutErrorCategory.AGENT_ERROR,
                        extra_fields=self.event_diagnostics(
                            event, RolloutErrorCategory.AGENT_ERROR
                        ),
                    ),
                    rollout_id,
                    "workflow",
                )

            if grader_result is not None:
                await self.try_callback(
                    pending.on_grader_complete, grader_result, rollout_id, "grader"
                )

            if self.cleanup_successful_trials and not pending.preserve_trial:
                shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
            if delete_trial and relocated and not pending.preserve_trial:
                trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
                shutil.rmtree(trial_dir, ignore_errors=True)
        finally:
            timings = trial_timings(event.result)
            if timings:
                logger.info("rollout %s phase timings: %s", rollout_id, timings)
            if not pending.done.done():
                pending.done.set_result(None)

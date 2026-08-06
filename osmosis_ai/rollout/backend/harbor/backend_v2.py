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
import inspect
import json
import logging
import math
import shutil
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
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

from osmosis_ai.packaging import BundleInfo
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
    agent_phase_failure,
    categorize_exception,
    diagnostic_payload,
    failure_phase,
    redact_secrets,
    trial_metrics,
    trial_timings,
)
from osmosis_ai.rollout.backend.harbor.native_agents import (
    NativeAgentBinding,
    native_agent_config,
    native_binding,
    native_prewarm_agent_config,
    validate_model_for_binding,
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
from osmosis_ai.rollout.utils.ttl_cache import TtlCache

logger: logging.Logger = logging.getLogger(__name__)

HARNESS_AGENT_IMPORT_PATH = (
    "osmosis_ai.rollout.backend.harbor.harness_agent:OsmosisHarnessInstalledAgent"
)
PREWARM_PREFIX = "prewarm-"
STATUS_RETENTION_SEC = 900.0


class HarborBackendV2(ExecutionBackend):
    def __init__(
        self,
        *,
        orchestrator: TrialQueue,
        tasks_dir: Path,
        agent: str | type | None = None,
        task_mode: TaskMode | str = TaskMode.TEMPLATE,
        task_resolver: Callable[[ExecutionRequest], HarborTask | Awaitable[HarborTask]]
        | None = None,
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
        max_queue_depth: int | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.tasks_dir: Path = Path(tasks_dir)
        self.task_mode: TaskMode = TaskMode(task_mode)
        self.task_resolver = task_resolver
        self.model_name = model_name
        self.agent = agent
        if agent_setup_timeout_sec is not None and not (
            math.isfinite(agent_setup_timeout_sec) and agent_setup_timeout_sec > 0
        ):
            # Harbor feeds this straight into asyncio.wait_for; 0/negative/NaN
            # break it.
            raise ValueError("agent_setup_timeout_sec must be a finite value > 0")
        self.agent_setup_timeout_sec = agent_setup_timeout_sec
        self.native: NativeAgentBinding | None = native_binding(agent)
        if self.native is not None and isinstance(agent, str):
            validate_model_for_binding(agent, self.native, model_name)
        if self.native and not self.native.trainable:
            logger.warning(
                "native agent %r emits no model trajectory; use it to validate "
                "datasets and verifiers, never for training",
                agent,
            )
        self.bundle: BundleInfo | None = resolve_backend_bundle(
            agent=agent,
            grader=grader,
            workflow_config=workflow_config,
            grader_config=grader_config,
            code_dir=code_dir,
            bundle=bundle,
            native=self.native is not None,
        )
        # The placement helper mutates in place; keep the caller's object
        # untouched.
        self.environment_config: HarborEnvironmentConfig = (
            apply_managed_skypilot_placement(
                environment_config.model_copy(deep=True)
                if environment_config is not None
                else HarborEnvironmentConfig()
            )
        )
        if patch_dockerfile_with_sdk and self.bundle is None:
            raise ValueError("patch_dockerfile_with_sdk requires a bundle")
        if patch_dockerfile_with_sdk is None:
            patch_dockerfile_with_sdk = self.bundle is not None
        # The bundle's declared dependencies (stable) are pre-installed into the
        # task image; the bundle itself (volatile user code) installs per trial.
        self.sdk_requirements: list[str] | None = (
            self.bundle.requirements
            if patch_dockerfile_with_sdk and self.bundle
            else None
        )

        root = Path(f"/tmp/osmosis-harbor-{self.tasks_dir.name}")
        self.rollouts_dir: Path = root / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir: Path = trials_dir or root / "trials"
        self.artifact_root: Path = default_artifact_root()
        self.cleanup_successful_trials = cleanup_successful_trials
        if max_queue_depth is not None and max_queue_depth < 1:
            # A depth of 0 is unenforceable here; it previously meant
            # reject-everything.
            raise ValueError("max_queue_depth must be >= 1, or None for unbounded")
        self.max_queue_depth = max_queue_depth
        self.pending: dict[str, PendingTrial] = {}
        # Hooks dispatch on membership, never on caller-controlled id patterns.
        self.prewarm_trials: set[str] = set()
        self.fetch_locks: dict[str, asyncio.Lock] = {}
        self.running: int = 0
        self.finished: TtlCache[str, dict[str, Any]] = TtlCache(STATUS_RETENTION_SEC)

        orchestrator.add_hook(TrialEvent.START, self.on_trial_started)
        orchestrator.add_hook(TrialEvent.VERIFICATION_START, self.on_verification_start)
        orchestrator.add_hook(TrialEvent.END, self.on_trial_end)

    @property
    def capture_final_result(self) -> bool:
        # Harbor verification computes the final reward backend-side; the
        # server should archive it even when no grader callback URL exists.
        return True

    def queued(self) -> int:
        return max(0, len(self.pending) - self.running)

    def has_capacity(self) -> bool:
        return self.max_queue_depth is None or self.queued() < self.max_queue_depth

    def record_outcome(
        self,
        rollout_id: str,
        status: RolloutStatus,
        reward: float | None = None,
        err_message: str | None = None,
    ) -> None:
        """Retain a terminal state so status polls answer after completion."""
        self.finished.set(
            rollout_id,
            {"status": status, "reward": reward, "err_message": err_message},
        )

    def rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        pending = self.pending.get(rollout_id)
        if pending is not None:
            if pending.grading:
                return {"status": RolloutStatus.GRADING}
            if pending.started:
                return {"status": RolloutStatus.RUNNING}
            return {"status": RolloutStatus.QUEUED}
        return self.finished.get(rollout_id)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "harbor-v2",
            "agent": self.agent
            if isinstance(self.agent, str)
            else ensure_import_path(self.agent),
            "in_flight": len(self.pending),
            "running": self.running,
            "queued": self.queued(),
            "max_queue_depth": self.max_queue_depth,
        }

    async def resolve_task(self, request: ExecutionRequest) -> HarborTask:
        # A caller-supplied resolver bypasses metadata and task-mode resolution.
        if self.task_resolver is not None:
            task = self.task_resolver(request)
            return await task if inspect.isawaitable(task) else task
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
        if self.native is not None and isinstance(self.agent, str):
            metadata = container_input.metadata or {}
            # A present-but-invalid override must fail, not fall back.
            if "harbor_model" in metadata:
                model = metadata["harbor_model"]
                if not isinstance(model, str) or not model.strip():
                    raise ValueError(
                        f"rollout {container_input.rollout_id!r}: "
                        f"metadata['harbor_model'] must be a non-empty string; "
                        f"got {model!r}"
                    )
            else:
                model = self.model_name
            validate_model_for_binding(self.agent, self.native, model)
            url = container_input.chat_completions_url
            if self.native.wiring != "none" and not url:
                # An empty api_base routes litellm (and the credential) to the
                # provider's public endpoint.
                raise ValueError(
                    f"rollout {container_input.rollout_id!r} has no "
                    f"chat_completions_url; refusing to wire {self.agent!r} "
                    "without the controller endpoint"
                )
            return native_agent_config(
                self.agent,
                self.native,
                model,
                url,
                container_input.api_key or "dummy",
            )
        return self.harness_agent_config(task_dir)

    def harness_agent_config(self, task_dir: Path) -> HarborAgentConfig:
        if self.bundle is None:
            raise ValueError("workflow agents require a bundle")
        return HarborAgentConfig(
            import_path=HARNESS_AGENT_IMPORT_PATH,
            kwargs={
                "bundle_path": str(self.bundle.wheel),
                "agent_script": self.bundle.agent_script,
                "input_path": str(task_dir / "container_input.json"),
            },
        )

    def build_trial_config(
        self,
        task_dir: Path,
        request: ExecutionRequest,
        container_input: ContainerInput,
        agent_config: HarborAgentConfig | None = None,
    ) -> TrialConfig:
        if agent_config is None:
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
            # Only the bundled harness reads the top-level input file; don't
            # stage the api_key without a consumer.
            write_input=self.bundle is not None and self.native is None,
        )

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        if request.id in self.pending:
            # A duplicate would overwrite the live rollout's pending state and
            # staged files, then strand one of the two execute() calls forever.
            raise ValueError(
                f"rollout {request.id!r} is already active; duplicate "
                "submissions are rejected"
            )
        pending = PendingTrial(on_workflow_complete, on_grader_complete)
        self.pending[request.id] = pending
        try:
            container_input = self.build_input(request)
            pending.api_key = container_input.api_key
            task = await self.resolve_task(request)
            # Dataset mode zeroes the prompt, so the task keeps its own
            # instruction.md. In template mode the prompt owns it: silently
            # for the configured template dir, with a warning elsewhere.
            if (
                container_input.prompt
                and task.path != self.tasks_dir.resolve()
                and (task.path / "instruction.md").is_file()
            ):
                logger.warning(
                    "rollout %s: template mode replaces the instruction.md that "
                    "task %s ships with the request prompt; use "
                    "task_mode='dataset' to keep the task's own instruction",
                    request.id,
                    task.path.name,
                )
            task_dir = self.materialize_task(task, request.id, container_input)
            pending.task = asyncio.create_task(
                self.orchestrator.submit(
                    self.build_trial_config(task_dir, request, container_input)
                )
            )
            trial_result = await pending.task
            await pending.done
        except asyncio.CancelledError:
            # The canceller owns the outcome; no callbacks are due.
            self.pending.pop(request.id, None)
            self.record_outcome(request.id, RolloutStatus.CANCELLED)
            self.cleanup_rollout_residue(request.id, include_trial=True)
            if not pending.cancel_requested:
                # Not a requested rollout cancellation: uvicorn shutdown or an
                # enclosing task group owns this; swallowing it would corrupt
                # the canceller's bookkeeping.
                raise
            logger.info("Rollout %s cancelled", request.id)
        except Exception as e:
            self.pending.pop(request.id, None)
            category = categorize_exception(e)
            self.record_outcome(request.id, RolloutStatus.FAILURE, err_message=str(e))
            self.cleanup_rollout_residue(request.id, include_trial=False)
            logger.error("Failed trial %s: %s", request.id, traceback.format_exc())
            failure = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=str(e),
                err_category=category,
                extra_fields=diagnostic_payload(
                    phase="setup",
                    category=category,
                    exception_type=type(e).__name__,
                    timings={},
                ),
            )
            # Both channels get a terminal outcome, so a controller waiting on
            # the grader callback does not burn its full deadline; wrapped
            # deliveries keep exceptions out of the server's fabrication path.
            if not pending.workflow_complete_called:
                await self.try_callback(
                    on_workflow_complete,
                    pending.workflow_result or failure,
                    request.id,
                    "workflow",
                )
            if on_grader_complete is not None and not pending.grader_complete_called:
                await self.try_callback(
                    on_grader_complete,
                    pending.grader_result or failure,
                    request.id,
                    "grader",
                )
        else:
            err = getattr(trial_result, "exception_info", None)
            if err is not None and err.exception_type == "CancelledError":
                self.cleanup_rollout_residue(request.id, include_trial=True)
            else:
                self.archive_trial(request.id, trial_result, pending)

    def archive_trial(
        self, rollout_id: str, trial_result: Any, pending: PendingTrial
    ) -> None:
        """Persist trial artifacts and clean per-rollout staging.

        Runs strictly after ``orchestrator.submit()`` resolved: harbor scrubs
        secrets before ``submit()`` returns, so any earlier relocation (e.g.
        from a trial hook) copies unredacted content into durable storage.
        """
        merge_grader_artifacts(self.trials_dir, rollout_id)
        delete_trial = bool(
            self.cleanup_successful_trials
            and trial_result is not None
            and getattr(trial_result, "exception_info", None) is None
            and not pending.preserve_trial
        )
        relocated = relocate_trial_artifacts(
            self.trials_dir, self.artifact_root, rollout_id, move=delete_trial
        )
        if self.cleanup_successful_trials and not pending.preserve_trial:
            shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
        if delete_trial and relocated:
            trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
            shutil.rmtree(trial_dir, ignore_errors=True)

    def cleanup_rollout_residue(self, rollout_id: str, *, include_trial: bool) -> None:
        """Remove per-rollout staging (and optionally the trial directory).

        The staging dir can hold the api_key in cleartext; failed trials keep
        their trial directory for debugging.
        """
        shutil.rmtree(self.rollouts_dir / rollout_id, ignore_errors=True)
        if include_trial:
            trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
            shutil.rmtree(trial_dir, ignore_errors=True)

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
        failures: list[str] = []
        try:
            outcomes = await asyncio.gather(
                *(self.orchestrator.submit(config) for config in configs),
                return_exceptions=True,
            )

            for config, outcome in zip(configs, outcomes, strict=True):
                label = config.task.path.name if config.task.path else config.trial_name
                if isinstance(outcome, BaseException):
                    failures.append(f"{label}: {type(outcome).__name__}: {outcome}")
                elif (err := getattr(outcome, "exception_info", None)) is not None:
                    failures.append(f"{label}: {err.exception_type}")
                elif self.cleanup_successful_trials:
                    # submit() resolved, so harbor's secret scrub has run.
                    self.cleanup_rollout_residue(
                        config.trial_name.removeprefix(TRIAL_NAME_PREFIX),
                        include_trial=True,
                    )
        finally:
            self.prewarm_trials.difference_update(
                config.trial_name for config in configs
            )
        if failures:
            raise RuntimeError(
                f"prewarm failed for {len(failures)} of {len(configs)} task(s):\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )
        logger.info("Prewarmed %d harbor task(s)", len(configs))

    def prewarm_agent_config(self, task_dir: Path) -> HarborAgentConfig:
        """Install-only trials never run the agent: no endpoint, no credentials."""
        if self.native is not None and isinstance(self.agent, str):
            return native_prewarm_agent_config(self.agent, self.native, self.model_name)
        return self.harness_agent_config(task_dir)

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
            task_dir,
            ExecutionRequest(id=rollout_id, prompt=[]),
            container_input,
            agent_config=self.prewarm_agent_config(task_dir),
        )
        config.install_only = True
        config.verifier.disable = True
        self.prewarm_trials.add(config.trial_name)
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
        if event.config.trial_name in self.prewarm_trials:
            return
        pending = self.pending.get(parse_rollout_id(event))
        if pending:
            pending.started = True

    def cancel_rollouts(
        self,
        ids: Sequence[str] | None = None,
        prefix: str | None = None,
        all: bool = False,
    ) -> dict[str, str]:
        """Cancel matching rollouts; queued ones never reach a sandbox.

        Returns a disposition per requested rollout: ``cancelled_queued``,
        ``cancelled_running``, or ``not_found`` (unknown or already finished,
        making cancellation idempotent).
        """
        if all:
            selected = list(self.pending)
        elif prefix is not None:
            selected = [rid for rid in self.pending if rid.startswith(prefix)]
        else:
            selected = list(ids or [])

        dispositions: dict[str, str] = {}
        for rollout_id in selected:
            pending = self.pending.get(rollout_id)
            if pending is None or pending.task is None or pending.task.done():
                dispositions[rollout_id] = "not_found"
                continue
            pending.cancel_requested = True
            pending.task.cancel()
            dispositions[rollout_id] = (
                "cancelled_running" if pending.started else "cancelled_queued"
            )
        return dispositions

    def event_diagnostics(
        self,
        event: TrialHookEvent,
        category: RolloutErrorCategory | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        err = event.result.exception_info if event.result else None
        return diagnostic_payload(
            phase=phase or failure_phase(event.result),
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
        if output is None:
            return None
        messages = output.primary_messages()
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
        if event.config.trial_name in self.prewarm_trials:
            return
        rollout_id = parse_rollout_id(event)
        pending = self.pending.get(rollout_id)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return
        pending.grading = True

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

        pending.workflow_result = outcome
        pending.workflow_complete_called = await self.try_callback(
            pending.on_workflow_complete, outcome, rollout_id, "workflow"
        )

    def grader_outcome(
        self, event: TrialHookEvent, rollout_id: str, pending: PendingTrial
    ) -> ExecutionResult:
        sample = self.primary_sample(event, rollout_id, pending)
        err = event.result.exception_info if event.result else None

        # Report agent-phase failures before the verifier branch buries them
        # under a secondary validation error.
        if (agent_err := agent_phase_failure(event.result)) is not None:
            log_trial_exception(rollout_id, agent_err, phase="during the agent run")
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                err_message=agent_err.exception_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
                extra_fields=self.event_diagnostics(
                    event, RolloutErrorCategory.AGENT_ERROR
                ),
            )

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
                        event, RolloutErrorCategory.VALIDATION_ERROR, phase="grading"
                    ),
                )
            return ExecutionResult(
                status=RolloutStatus.SUCCESS,
                sample=sample,
                extra_fields=self.event_diagnostics(event),
            )

        if err:
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
        # Reachable in normal operation: a task without tests/ and without a
        # grader disables the verifier, so no reward source ever runs.
        return ExecutionResult(
            status=RolloutStatus.FAILURE,
            sample=sample,
            err_message=(
                "trial ended with no verifier result and no recorded error; "
                "the task has no reward source (no tests/ and no grader)"
            ),
            err_category=RolloutErrorCategory.VALIDATION_ERROR,
            extra_fields=self.event_diagnostics(
                event, RolloutErrorCategory.VALIDATION_ERROR, phase="grading"
            ),
        )

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        self.running = max(0, self.running - 1)
        if event.config.trial_name in self.prewarm_trials:
            # Prewarm cleanup happens at the prewarm() call site, after
            # submit() resolves (post-scrub).
            return
        rollout_id = parse_rollout_id(event)

        pending = self.pending.pop(rollout_id, None)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        err = event.result.exception_info if event.result else None
        if err and err.exception_type == "CancelledError":
            self.record_outcome(rollout_id, RolloutStatus.CANCELLED)
            if not pending.done.done():
                pending.done.set_result(None)
            return

        try:
            grader_result = self.grader_outcome(event, rollout_id, pending)
            self.record_outcome(
                rollout_id,
                RolloutStatus(grader_result.status.value),
                reward=grader_result.sample.reward if grader_result.sample else None,
                err_message=grader_result.err_message,
            )

            if not pending.workflow_complete_called:
                if pending.workflow_result is not None:
                    # The semantic outcome was produced but delivery failed;
                    # resend it byte-identical instead of fabricating a
                    # failure for a trial that may have succeeded.
                    result = pending.workflow_result
                elif event.result and event.result.exception_info:
                    err = event.result.exception_info
                    log_trial_exception(
                        rollout_id, err, phase="before the agent completed"
                    )
                    result = ExecutionResult(
                        status=RolloutStatus.FAILURE,
                        err_message=err.exception_message,
                        err_category=RolloutErrorCategory.AGENT_ERROR,
                        extra_fields=self.event_diagnostics(
                            event, RolloutErrorCategory.AGENT_ERROR
                        ),
                    )
                else:
                    result = ExecutionResult(
                        status=RolloutStatus.FAILURE,
                        err_message="Trial ended before agent completed",
                        err_category=RolloutErrorCategory.AGENT_ERROR,
                        extra_fields=self.event_diagnostics(
                            event, RolloutErrorCategory.AGENT_ERROR
                        ),
                    )
                pending.workflow_result = result
                pending.workflow_complete_called = await self.try_callback(
                    pending.on_workflow_complete, result, rollout_id, "workflow"
                )

            if pending.on_grader_complete:
                pending.grader_result = grader_result
                pending.grader_complete_called = await self.try_callback(
                    pending.on_grader_complete, grader_result, rollout_id, "grader"
                )
        finally:
            timings = trial_timings(event.result)
            if timings:
                logger.info("rollout %s phase timings: %s", rollout_id, timings)
            if not pending.done.done():
                pending.done.set_result(None)

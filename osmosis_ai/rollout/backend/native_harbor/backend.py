"""Native Harbor execution backend: drive one harbor Trial per rollout and map
its verifier reward onto the rollout's single sample. The agent is fixed per
backend; only the task and model vary per rollout via metadata."""

import importlib
import json
import logging
import shutil
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from harbor.models.job.config import RetryConfig
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import (
    AgentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.config import (
    EnvironmentConfig as HarborEnvironmentConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialHookEvent
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.backend.harbor.backend import (
    apply_managed_skypilot_placement,
    log_trial_exception,
    rewrite_url_for_docker,
    uses_local_docker_runtime,
)
from osmosis_ai.rollout.context import (
    RolloutContext,
    get_rollout_context,
)
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.file_artifacts import (
    copy_artifact_tree,
    default_artifact_root,
)
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward

logger: logging.Logger = logging.getLogger(__name__)

HARBOR_TASK_KEY = "harbor_task"
HARBOR_MODEL_KEY = "harbor_model"
GIT_URL_KEY = "git_url"
GIT_TASK_PATH_KEY = "task_path"
GIT_COMMIT_KEY = "git_commit_id"

DEFAULT_AGENT_NAME = "terminus-2"
DEFAULT_MODEL_NAME = "openai/osmosis-rollout"
DEFAULT_REWARD_KEY = "reward"
DEFAULT_MAX_CONCURRENT = 8
TRIAL_NAME_PREFIX = "native-"

# terminus-2 summarizes mid-run, breaking training's append-only trajectory; default
# it off (agent_kwargs override). Default agent only -- other agents are the caller's job.
_TERMINUS_2_DEFAULT_KWARGS: dict[str, Any] = {
    "enable_summarize": False,
    "proactive_summarization_threshold": 0,
}

TaskResolver = Callable[[ExecutionRequest], TaskConfig]

_REDACTED = "[REDACTED]"
_SENSITIVE_AGENT_EXTRA_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


@dataclass
class _PendingNativeTrial:
    request: ExecutionRequest
    context: RolloutContext
    on_workflow_complete: ResultCallback
    workflow_complete_called: bool = False
    workflow_result: ExecutionResult | None = None
    preserve_trial: bool = False


def resolve_task(request: ExecutionRequest) -> TaskConfig:
    """Resolve metadata["harbor_task"] to a TaskConfig: local path, package, or git."""
    md = request.metadata or {}
    raw = md.get(HARBOR_TASK_KEY)
    if not raw:
        raise ValueError(
            f"metadata[{HARBOR_TASK_KEY!r}] is required for the native harbor backend"
        )

    if isinstance(raw, str) and raw.startswith((".", "/", "~")):
        return TaskConfig(path=Path(raw).expanduser())

    if md.get(GIT_URL_KEY):
        task_path = md.get(GIT_TASK_PATH_KEY)
        return TaskConfig(
            git_url=md[GIT_URL_KEY],
            path=Path(task_path) if task_path else None,
            git_commit_id=md.get(GIT_COMMIT_KEY),
        )

    name, _, ref = str(raw).partition("@")
    if "/" not in name:
        raise ValueError(
            f"metadata[{HARBOR_TASK_KEY!r}]={raw!r} must be a local path "
            "(./, /, ~), a git form (set git_url), or a package 'org/name[@ref]'"
        )
    return TaskConfig(name=name, ref=ref or "latest")


def _categorize_exception(exc: Exception) -> RolloutErrorCategory:
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    return RolloutErrorCategory.AGENT_ERROR


def _resolve_agent_class(
    name: str | None, import_path: str | None
) -> type[BaseAgent] | None:
    """Resolve the harbor agent class, or None to let Trial.create raise the canonical error."""
    if import_path:
        if ":" not in import_path:
            return None
        module_path, _, class_name = import_path.partition(":")
        try:
            return getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError):
            return None
    try:
        # get_agent_class imports the class; _AGENT_MAP holds path strings, not classes.
        return AgentFactory.get_agent_class(AgentName(name))
    except (KeyError, ValueError, ImportError, AttributeError):
        return None


def _is_installed_agent(cls: type[BaseAgent] | None) -> bool:
    """Whether the agent is a harbor installed CLI (env-wired) vs in-process."""
    return cls is not None and issubclass(cls, BaseInstalledAgent)


def _is_sensitive_agent_extra_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_AGENT_EXTRA_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _SENSITIVE_AGENT_EXTRA_KEYS
    )


def _redact_agent_extra(value: Any, *, api_key: str | None) -> Any:
    """Preserve agent metadata while replacing credential-bearing leaves."""
    if isinstance(value, dict):
        return {
            key: _REDACTED
            if _is_sensitive_agent_extra_key(str(key))
            else _redact_agent_extra(child, api_key=api_key)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_agent_extra(child, api_key=api_key) for child in value]
    if isinstance(value, tuple):
        return [_redact_agent_extra(child, api_key=api_key) for child in value]
    if api_key and isinstance(value, str) and api_key in value:
        return _REDACTED
    return value


class NativeHarborBackend(ExecutionBackend):
    """Drive a harbor Trial per rollout and map its verifier reward."""

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        agent_env: dict[str, str] | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        reward_key: str = DEFAULT_REWARD_KEY,
        trials_dir: Path | str = Path("native_trials"),
        task_resolver: TaskResolver | None = None,
        environment_config: HarborEnvironmentConfig | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        retry_config: RetryConfig | None = None,
        cleanup_successful_trials: bool = True,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1; the native harbor backend spawns a "
                "harbor Trial (often a container) per rollout, so unbounded "
                "concurrency would exhaust the host."
            )
        if agent_name is not None and agent_import_path is not None:
            raise ValueError("set agent_name or agent_import_path, not both")
        if agent_name is None and agent_import_path is None:
            agent_name = DEFAULT_AGENT_NAME
        self.agent_name = agent_name
        self.agent_import_path = agent_import_path
        self.agent_kwargs = agent_kwargs
        self.agent_env = agent_env
        self.model_name = model_name
        self.reward_key = reward_key
        self.trials_dir: Path = Path(trials_dir)
        self.task_resolver: TaskResolver = task_resolver or resolve_task
        self._environment_config: HarborEnvironmentConfig = (
            apply_managed_skypilot_placement(
                environment_config or HarborEnvironmentConfig()
            )
        )
        self.cleanup_successful_trials = cleanup_successful_trials
        self._max_concurrency = max_concurrent
        self.artifact_root: Path = default_artifact_root()
        self._pending: dict[str, _PendingNativeTrial] = {}
        self._retry_config: RetryConfig = retry_config or RetryConfig()
        self._queue = TrialQueue(
            n_concurrent=max_concurrent, retry_config=self._retry_config
        )
        self._queue.on_verification_started(self._on_verification_started)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "native_harbor",
            "agent": self.agent_name or self.agent_import_path,
            "max_concurrency": self._max_concurrency,
        }

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        ctx = get_rollout_context() or RolloutContext()
        trial_name = f"{TRIAL_NAME_PREFIX}{request.id}"
        pending = _PendingNativeTrial(
            request=request,
            context=ctx,
            on_workflow_complete=on_workflow_complete,
        )
        self._pending[trial_name] = pending
        trial_result: TrialResult | None = None
        setup_error: Exception | None = None
        try:
            try:
                task_cfg = self.task_resolver(request)
                agent_cfg = self._build_agent_config(request, ctx)
                trial_cfg = self._build_trial_config(
                    request, task_cfg, agent_cfg, trial_name
                )
                # submit() = Trial.create + run under the queue's semaphore (+ retry).
                trial_result = await self._queue.submit(trial_cfg)
            except Exception as exc:
                setup_error = exc
                logger.error(
                    "Native trial %s failed to run: %s",
                    request.id,
                    traceback.format_exc(),
                )

            # Single-step Harbor trials fire the workflow callback at verification
            # start. Multi-step verification is interleaved with agent work, so its
            # workflow callback intentionally waits for the final trial result.
            if not pending.workflow_complete_called:
                workflow_result = self._build_workflow_result(
                    pending,
                    trial_result,
                    setup_error=setup_error,
                )
                pending.workflow_result = workflow_result
                pending.workflow_complete_called = True
                await self._safe_callback(
                    on_workflow_complete, workflow_result, request.id, "workflow"
                )

            workflow_result = pending.workflow_result or ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message="Trial ended before the agent result was available",
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )
            if on_grader_complete is not None:
                grader_result = self._build_grader_result(
                    request, workflow_result, trial_result
                )
                await self._safe_callback(
                    on_grader_complete, grader_result, request.id, "grader"
                )

            relocated = self._relocate_trial_artifacts(request.id)
            successful = bool(
                trial_result is not None
                and getattr(trial_result, "exception_info", None) is None
            )
            if (
                successful
                and relocated
                and not pending.preserve_trial
                and self.cleanup_successful_trials
            ):
                self._cleanup_trial(trial_name)
        finally:
            self._pending.pop(trial_name, None)

    @staticmethod
    async def _safe_callback(
        callback: ResultCallback, result: ExecutionResult, rollout_id: str, label: str
    ) -> None:
        """Swallow callback errors: a propagating one re-fires both callbacks in
        app.py, breaking fire-exactly-once."""
        try:
            await callback(result)
        except Exception:
            logger.error(
                "Native %s callback for rollout %s failed: %s",
                label,
                rollout_id,
                traceback.format_exc(),
            )

    async def _on_verification_started(self, event: TrialHookEvent) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is None or pending.workflow_complete_called:
            return
        if self._retry_config.max_retries > 0:
            # The event belongs to an attempt, not necessarily the attempt that
            # TrialQueue ultimately returns. Defer to the final result so a retry
            # cannot leave the controller holding a stale trajectory.
            return
        if event.result.step_results is not None:
            # Multi-step trials verify after every agent step. Firing here would
            # incorrectly announce workflow completion after the first step.
            return

        result = self._build_workflow_result(pending, event.result)
        pending.workflow_result = result
        pending.workflow_complete_called = True
        await self._safe_callback(
            pending.on_workflow_complete,
            result,
            pending.request.id,
            "workflow",
        )

    def _build_workflow_result(
        self,
        pending: _PendingNativeTrial,
        trial_result: TrialResult | None,
        *,
        setup_error: Exception | None = None,
    ) -> ExecutionResult:
        request = pending.request
        if setup_error is not None:
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=str(setup_error),
                err_category=_categorize_exception(setup_error),
            )

        trajectory_document = self._load_native_trajectory(
            request.id, pending.context.api_key
        )
        if trajectory_document is None and self._native_trajectory_paths(request.id):
            pending.preserve_trial = True

        sample = RolloutSample(
            label=request.label,
            trajectory_messages=None,
            metrics=self._trial_metrics(trial_result),
        )
        err = (
            getattr(trial_result, "exception_info", None)
            if trial_result is not None
            else None
        )
        if err is not None:
            if getattr(err, "exception_traceback", None) is not None:
                log_trial_exception(request.id, err, phase="during native execution")
            else:
                logger.error(
                    "Native Harbor trial %s failed during execution [%s]: %s",
                    request.id,
                    getattr(err, "exception_type", "unknown"),
                    getattr(err, "exception_message", "unknown error"),
                )
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                trajectory_document=trajectory_document,
                err_message=getattr(err, "exception_message", None)
                or "Trial failed before completion",
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
            trajectory_document=trajectory_document,
        )

    @staticmethod
    def _trial_metrics(trial_result: TrialResult | None) -> dict[str, Any]:
        if trial_result is None:
            return {}
        try:
            input_tokens, cache_tokens, output_tokens, cost_usd = (
                trial_result.compute_token_cost_totals()
            )
        except AttributeError:
            # Some integrators and unit tests provide a duck-typed older result.
            return {}
        except Exception:
            logger.warning("Failed to read native Harbor token totals", exc_info=True)
            return {}
        return {
            key: value
            for key, value in {
                "input_tokens": input_tokens,
                "cached_tokens": cache_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
            }.items()
            if value is not None
        }

    def _native_trajectory_paths(self, rollout_id: str) -> list[Path]:
        trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
        primary = trial_dir / "agent" / "trajectory.json"
        paths: list[Path] = []
        if primary.is_file():
            paths.append(primary)
        paths.extend(sorted((trial_dir / "steps").glob("*/agent/trajectory.json")))
        return paths

    def _load_native_trajectory(
        self,
        rollout_id: str,
        api_key: str | None,
    ) -> dict[str, Any] | None:
        paths = self._native_trajectory_paths(rollout_id)
        if not paths:
            logger.info(
                "Native Harbor agent emitted no ATIF trajectory for rollout %s",
                rollout_id,
            )
            return None
        if len(paths) > 1:
            logger.warning(
                "Native Harbor rollout %s emitted %d independent trajectory "
                "documents; preserving the trial directory instead of fabricating "
                "a lossy merged trajectory",
                rollout_id,
                len(paths),
            )
            return None

        try:
            raw = json.loads(paths[0].read_text())
            trajectory = Trajectory.model_validate(raw)
        except Exception:
            logger.warning(
                "Native Harbor emitted an invalid ATIF trajectory for rollout %s; "
                "preserving the trial directory",
                rollout_id,
                exc_info=True,
            )
            return None

        if trajectory.agent.extra is not None:
            trajectory.agent.extra = _redact_agent_extra(
                trajectory.agent.extra, api_key=api_key
            )
        return trajectory.to_json_dict(exclude_none=True)

    def _relocate_trial_artifacts(self, rollout_id: str) -> bool:
        """Copy only artifacts Harbor already collected from the environment."""
        trial_dir = self.trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
        destinations: list[tuple[Path, Path]] = []
        source = trial_dir / "artifacts"
        if source.is_dir():
            destinations.append((source, self.artifact_root / rollout_id / "artifacts"))
        steps_dir = trial_dir / "steps"
        if steps_dir.is_dir():
            for step_dir in sorted(steps_dir.iterdir(), key=lambda path: path.name):
                step_artifacts = step_dir / "artifacts"
                if step_artifacts.is_dir():
                    destinations.append(
                        (
                            step_artifacts,
                            self.artifact_root
                            / rollout_id
                            / "artifacts"
                            / "steps"
                            / step_dir.name,
                        )
                    )

        try:
            for source_dir, destination_dir in destinations:
                copy_artifact_tree(
                    source_dir,
                    destination_dir,
                    destination_root=self.artifact_root,
                    replace_destination=True,
                )
        except Exception:
            logger.warning(
                "Failed to relocate native Harbor artifacts for rollout %s; "
                "preserving the trial directory",
                rollout_id,
                exc_info=True,
            )
            return False
        return True

    def _cleanup_trial(self, trial_name: str) -> None:
        shutil.rmtree(self.trials_dir / trial_name, ignore_errors=True)

    def _build_trial_config(
        self,
        request: ExecutionRequest,
        task_cfg: TaskConfig,
        agent_cfg: AgentConfig,
        trial_name: str,
    ) -> TrialConfig:
        verifier_cfg = VerifierConfig(disable=False)
        if request.grader_timeout_sec is not None:
            verifier_cfg.override_timeout_sec = request.grader_timeout_sec
        return TrialConfig(
            task=task_cfg,
            trial_name=trial_name,
            trials_dir=self.trials_dir,
            agent=agent_cfg,
            verifier=verifier_cfg,
            environment=self._environment_config,
        )

    def _build_agent_config(
        self, request: ExecutionRequest, ctx: RolloutContext
    ) -> AgentConfig:
        md = request.metadata or {}
        name = self.agent_name
        import_path = self.agent_import_path
        model_name = md.get(HARBOR_MODEL_KEY) or self.model_name

        agent_cls = _resolve_agent_class(name, import_path)

        endpoint = ctx.chat_completions_url
        if not endpoint:
            raise ValueError(f"rollout {request.id!r} has no chat_completions_url")
        if uses_local_docker_runtime(self._environment_config):
            endpoint = rewrite_url_for_docker(endpoint)
        api_key = ctx.api_key
        # User passthrough is the base layer; SDK-wired values below overlay it.
        kwargs: dict[str, Any] = dict(self.agent_kwargs or {})
        env: dict[str, str] = dict(self.agent_env or {})

        if _is_installed_agent(agent_cls):
            # Env-wired; SDK endpoint/key overwrite agent_env so identity can't be redirected.
            env["OPENAI_BASE_URL"] = endpoint
            if api_key:
                env["OPENAI_API_KEY"] = api_key
        else:
            # Precedence low -> high: default-agent kwargs, user kwargs, SDK wiring.
            defaults = _TERMINUS_2_DEFAULT_KWARGS if name == DEFAULT_AGENT_NAME else {}
            kwargs = {**defaults, **kwargs}
            # Identity: api_base kwarg + api_key in llm_kwargs (deep-merged).
            kwargs["api_base"] = endpoint
            llm_kwargs: dict[str, Any] = dict(kwargs.get("llm_kwargs") or {})
            if api_key:
                llm_kwargs["api_key"] = api_key
            # TODO(temporary): in-process agents need JSON; pin stream=False in extra_body
            # (setdefault, so a user stream wins). REMOVE when controllers send JSON.
            extra_body: dict[str, Any] = dict(llm_kwargs.get("extra_body") or {})
            extra_body.setdefault("stream", False)
            llm_kwargs["extra_body"] = extra_body
            kwargs["llm_kwargs"] = llm_kwargs

        agent_cfg = AgentConfig(
            name=name,
            import_path=import_path,
            model_name=model_name,
            kwargs=kwargs,
            env=env,
        )
        if request.agent_timeout_sec is not None:
            agent_cfg.override_timeout_sec = request.agent_timeout_sec
        return agent_cfg

    def _build_grader_result(
        self,
        request: ExecutionRequest,
        workflow_result: ExecutionResult,
        trial_result: TrialResult | None,
    ) -> ExecutionResult:
        """Grade the rollout's single sample from the harbor verifier's in-memory
        result. On setup failure (no trial result) the workflow error is propagated."""
        sample = (
            workflow_result.sample.model_copy(deep=True)
            if workflow_result.sample is not None
            else RolloutSample(label=request.label, trajectory_messages=None)
        )
        if trial_result is None:
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                trajectory_document=workflow_result.trajectory_document,
                err_message=workflow_result.err_message,
                err_category=workflow_result.err_category,
            )

        reward_value = self._pick_reward(self._extract_rewards(trial_result))
        if reward_value is not None:
            sample.reward = float(reward_value)

        err = getattr(trial_result, "exception_info", None)
        if sample.reward is None and err is not None:
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                trajectory_document=workflow_result.trajectory_document,
                err_message=getattr(err, "exception_message", None)
                or "Trial failed before grading completed",
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )

        try:
            validate_sample_has_reward(sample)
        except ValueError as e:
            logger.warning("Native grading incomplete: %s", e)
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                trajectory_document=workflow_result.trajectory_document,
                err_message=str(e),
                err_category=RolloutErrorCategory.VALIDATION_ERROR,
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
            trajectory_document=workflow_result.trajectory_document,
        )

    @staticmethod
    def _extract_rewards(trial_result: TrialResult) -> dict[str, float | int] | None:
        """Verifier rewards, or None. Harbor rewards are a named-channel dict (not a
        scalar); take the trial-level one if present, else the first step's."""
        top = trial_result.verifier_result
        if top is not None and top.rewards:
            return top.rewards
        for step in trial_result.step_results or []:
            step_vr = step.verifier_result
            if step_vr is not None and step_vr.rewards:
                return step_vr.rewards
        return None

    def _pick_reward(
        self, rewards: dict[str, float | int] | None
    ) -> float | int | None:
        # Collapse named-channel rewards to one float: the 'reward' key, else the sole value.
        if not rewards:
            return None
        if self.reward_key in rewards:
            return rewards[self.reward_key]
        if len(rewards) == 1:
            return next(iter(rewards.values()))
        logger.warning(
            "Native verifier returned rewards %s with no %r channel; reward unset",
            sorted(rewards),
            self.reward_key,
        )
        return None

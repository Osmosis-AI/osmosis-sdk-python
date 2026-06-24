"""Native Harbor execution backend.

Drives each rollout as a harbor ``Trial``: resolve the task from
``metadata["harbor_task"]``, wire the controller endpoint/identity into the
agent, run it, and map the task verifier's reward onto a ``RolloutSample``.
"""

import importlib
import inspect
import logging
import shutil
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from harbor.models.job.config import RetryConfig
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
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.context import RolloutContext, get_rollout_context
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.rewards import validate_samples_have_rewards

logger: logging.Logger = logging.getLogger(__name__)

HARBOR_TASK_KEY = "harbor_task"
HARBOR_AGENT_KEY = "harbor_agent"
HARBOR_AGENT_IMPORT_PATH_KEY = "harbor_agent_import_path"
HARBOR_MODEL_KEY = "harbor_model"
GIT_URL_KEY = "git_url"
GIT_TASK_PATH_KEY = "task_path"
GIT_COMMIT_KEY = "git_commit_id"

DEFAULT_AGENT_NAME = "terminus-2"
DEFAULT_MODEL_NAME = "openai/osmosis-rollout"
DEFAULT_REWARD_KEY = "reward"
# Each rollout spawns a harbor Trial (often a container); bound concurrency.
DEFAULT_MAX_CONCURRENT = 8
TRIAL_NAME_PREFIX = "native-"

# Summarization knobs forced (to the safe value) on agents that declare them.
_TRAINING_SAFE_SUMMARIZATION_KWARGS: dict[str, Any] = {
    "enable_summarize": False,
    "proactive_summarization_threshold": 0,
}

TaskResolver = Callable[[ExecutionRequest], TaskConfig]


def resolve_task(request: ExecutionRequest) -> TaskConfig:
    """Map ``metadata["harbor_task"]`` to a ``TaskConfig`` selector
    (local path | package ``org/name@ref`` | git)."""
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
            f"metadata[{HARBOR_TASK_KEY!r}]={raw!r} is not a local path "
            "(start with ./, /, or ~), a git form (set git_url), or a package "
            "'org/name[@ref]'"
        )
    return TaskConfig(name=name, ref=ref or "latest")


def _categorize_exception(exc: Exception) -> RolloutErrorCategory:
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    return RolloutErrorCategory.AGENT_ERROR


def _categorize_exception_type(exc_type: str | None) -> RolloutErrorCategory:
    """Categorize a harbor ``ExceptionInfo.exception_type`` name -- the only
    signal, since ``Trial.run()`` swallows failures into ``exception_info``."""
    if not exc_type:
        return RolloutErrorCategory.AGENT_ERROR
    if exc_type.endswith("TimeoutError"):
        return RolloutErrorCategory.TIMEOUT
    if (
        "Verifier" in exc_type
        or "RewardFile" in exc_type
        or exc_type.endswith("ParseError")
    ):
        return RolloutErrorCategory.VALIDATION_ERROR
    return RolloutErrorCategory.AGENT_ERROR


def _resolve_agent_class(
    name: str | None, import_path: str | None
) -> type[BaseAgent] | None:
    """Resolve the harbor agent *class* (wiring decisions ask the class, not the
    name), or ``None`` to defer the canonical error to ``Trial.create``."""
    if import_path:
        if ":" not in import_path:
            return None
        module_path, _, class_name = import_path.partition(":")
        try:
            return getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError):
            return None
    try:
        # 0.15.0 _AGENT_MAP holds import-path strings; get_agent_class imports
        # the class (indexing the map would yield a str and break issubclass()).
        return AgentFactory.get_agent_class(AgentName(name))
    except (KeyError, ValueError, ImportError, AttributeError):
        return None


def _is_installed_agent(cls: type[BaseAgent] | None) -> bool:
    """Whether the agent is a harbor installed CLI (env-wired) vs in-process."""
    return cls is not None and issubclass(cls, BaseInstalledAgent)


def _accepts_summarization_knobs(cls: type[BaseAgent] | None) -> bool:
    """Whether the constructor declares the summarization knobs (``**kwargs``
    does not count)."""
    if cls is None:
        return False
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return all(knob in params for knob in _TRAINING_SAFE_SUMMARIZATION_KWARGS)


def _unaccepted_kwargs(
    cls: type[BaseAgent] | None, kwargs: dict[str, Any]
) -> list[str]:
    """Names in ``kwargs`` the constructor cannot receive (empty if unresolved
    or it declares ``**kwargs``)."""
    if cls is None:
        return []
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return [name for name in kwargs if name not in params]


class NativeHarborBackend(ExecutionBackend):
    """Drive a harbor ``Trial`` per rollout and map its verifier reward."""

    def __init__(
        self,
        *,
        default_agent_name: str = DEFAULT_AGENT_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        reward_key: str = DEFAULT_REWARD_KEY,
        trials_dir: Path | str = Path("native_trials"),
        inject_identity_headers: bool = True,
        collect_rollout_details: bool = False,
        training_safe: bool = True,
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
        self.default_agent_name = default_agent_name
        self.model_name = model_name
        self.reward_key = reward_key
        self.trials_dir = Path(trials_dir)
        self.inject_identity_headers = inject_identity_headers
        self.collect_rollout_details = collect_rollout_details
        self.training_safe = training_safe
        self.task_resolver: TaskResolver = task_resolver or resolve_task
        # Trial-layer environment selects the sandbox type (docker default, or
        # daytona/e2b/...); the task only carries resources, not the type.
        self.environment_config = environment_config or HarborEnvironmentConfig()
        self.cleanup_successful_trials = cleanup_successful_trials
        self._max_concurrency = max_concurrent
        # Shared orchestrator: bounds concurrency via its semaphore and adds
        # retry. Native uses submit()'s return value, so registers no hooks.
        self._queue = TrialQueue(n_concurrent=max_concurrent, retry_config=retry_config)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "native_harbor",
            "agent": self.default_agent_name,
            "max_concurrency": self._max_concurrency,
        }

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        ctx = get_rollout_context() or RolloutContext()
        sample_id = uuid.uuid4().hex

        workflow_result, trial_result = await self._run_trial(request, ctx, sample_id)
        await self._safe_callback(
            on_workflow_complete, workflow_result, request.id, "workflow"
        )

        if on_grader_complete is None:
            return

        if trial_result is not None:
            grader_result = self._build_grader_result(
                trial_result, workflow_result.samples, sample_id
            )
        else:
            # Setup failed before any result; fire FAILURE so the grader wait resolves.
            grader_result = ExecutionResult(
                status=RolloutStatus.FAILURE,
                samples=workflow_result.samples,
                err_message=workflow_result.err_message,
                err_category=workflow_result.err_category,
            )
        await self._safe_callback(
            on_grader_complete, grader_result, request.id, "grader"
        )

    @staticmethod
    async def _safe_callback(
        callback: ResultCallback, result: ExecutionResult, rollout_id: str, label: str
    ) -> None:
        """Invoke a callback without letting it escape ``execute()``: a
        propagating error would trip app.py into re-firing both callbacks,
        breaking fire-exactly-once."""
        try:
            await callback(result)
        except Exception:
            logger.error(
                "Native %s callback for rollout %s failed: %s",
                label,
                rollout_id,
                traceback.format_exc(),
            )

    async def _run_trial(
        self, request: ExecutionRequest, ctx: RolloutContext, sample_id: str
    ) -> tuple[ExecutionResult, TrialResult | None]:
        """Build + run the trial. Never raises; returns a FAILURE result (with
        ``None`` trial when creation/setup failed)."""
        samples = {sample_id: RolloutSample(id=sample_id, label=request.label)}
        # sample_id makes the trial dir unique, so a reused/retried rollout_id
        # cannot collide on a shared dir.
        trial_name = f"{TRIAL_NAME_PREFIX}{request.id}-{sample_id}"
        try:
            task_cfg = self.task_resolver(request)
            agent_cfg = self._build_agent_config(request, ctx, sample_id)
            trial_cfg = self._build_trial_config(
                request, task_cfg, agent_cfg, trial_name
            )
            # submit() = Trial.create + run under the queue's semaphore (+ retry).
            result = await self._queue.submit(trial_cfg)
        except Exception as e:
            logger.error(
                "Native trial %s failed to run: %s",
                request.id,
                traceback.format_exc(),
            )
            return (
                ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    samples=samples,
                    err_message=str(e),
                    err_category=_categorize_exception(e),
                ),
                None,
            )

        # Harbor swallows failures into exception_info; keep the dir for
        # debugging and categorize by the recorded type (a timeout is not an
        # agent error).
        if result.exception_info is not None:
            err = result.exception_info
            return (
                ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    samples=samples,
                    err_message=getattr(err, "exception_message", None)
                    or "Trial failed before completion",
                    err_category=_categorize_exception_type(
                        getattr(err, "exception_type", None)
                    ),
                ),
                result,
            )

        # Reward came from the in-memory result, so the trial dir is disposable.
        self._cleanup_trial(trial_name)
        return ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples), result

    def _cleanup_trial(self, trial_name: str) -> None:
        if self.cleanup_successful_trials:
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
            environment=self.environment_config,
        )

    def _build_agent_config(
        self, request: ExecutionRequest, ctx: RolloutContext, sample_id: str
    ) -> AgentConfig:
        md = request.metadata or {}
        # harbor's factory ignores import_path when name is also set; reject both.
        name = md.get(HARBOR_AGENT_KEY) or None
        import_path = md.get(HARBOR_AGENT_IMPORT_PATH_KEY) or None
        if name is not None and import_path is not None:
            raise ValueError(
                f"metadata sets both {HARBOR_AGENT_KEY!r} and "
                f"{HARBOR_AGENT_IMPORT_PATH_KEY!r}; choose one"
            )
        if name is None and import_path is None:
            name = self.default_agent_name
        model_name = md.get(HARBOR_MODEL_KEY) or self.model_name

        agent_cls = _resolve_agent_class(name, import_path)

        endpoint = ctx.chat_completions_url
        api_key = ctx.api_key
        kwargs: dict[str, Any] = {}
        env: dict[str, str] = {}

        if _is_installed_agent(agent_cls):
            # Installed CLIs can't forward per-rollout identity headers; if the
            # controller requires them, fail loud instead of 400-ing every call.
            if self.inject_identity_headers:
                cls_name = agent_cls.__name__ if agent_cls is not None else name
                raise ValueError(
                    f"Installed harbor agent {cls_name!r} is not yet supported by "
                    "the native harbor backend: it cannot send the per-rollout "
                    "identity headers the policy controller requires. Use an "
                    "in-process agent, or set inject_identity_headers=False if "
                    "your controller routes identity another way."
                )
            if endpoint:
                env["OPENAI_BASE_URL"] = endpoint
            if api_key:
                env["OPENAI_API_KEY"] = api_key
        else:
            # In-process agent: endpoint/key/identity go via kwargs.
            if endpoint:
                kwargs["api_base"] = endpoint
            else:
                # Without api_base, litellm routes "openai/..." to api.openai.com.
                logger.warning(
                    "No chat_completions_url in RolloutContext for rollout %s; "
                    "the agent's LLM calls will not reach the policy server.",
                    request.id,
                )
            kwargs["collect_rollout_details"] = self.collect_rollout_details
            llm_kwargs: dict[str, Any] = {}
            if api_key:
                llm_kwargs["api_key"] = api_key
            if self.inject_identity_headers:
                if not ctx.rollout_id:
                    logger.warning(
                        "Injecting an empty x-rollout-id for rollout %s; the "
                        "policy controller will reject the call.",
                        request.id,
                    )
                llm_kwargs["extra_headers"] = {
                    "x-rollout-id": ctx.rollout_id,
                    "x-sample-id": sample_id,
                }
            if llm_kwargs:
                kwargs["llm_kwargs"] = llm_kwargs
            if self.training_safe and _accepts_summarization_knobs(agent_cls):
                kwargs.update(_TRAINING_SAFE_SUMMARIZATION_KWARGS)

            # Preflight: a clear error instead of a cryptic TypeError from harbor.
            unaccepted = _unaccepted_kwargs(agent_cls, kwargs)
            if agent_cls is not None and unaccepted:
                raise ValueError(
                    f"In-process agent {agent_cls.__name__!r} cannot receive the "
                    f"native-harbor wiring kwargs {sorted(unaccepted)}: add them "
                    f"(or **kwargs) to its __init__, or expose it as an installed "
                    f"agent wired via OPENAI_BASE_URL/OPENAI_API_KEY env vars."
                )

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
        trial_result: TrialResult,
        samples: dict[str, RolloutSample],
        sample_id: str,
    ) -> ExecutionResult:
        reward_value = self._pick_reward(self._extract_rewards(trial_result))
        if reward_value is not None:
            samples[sample_id].reward = float(reward_value)

        try:
            validate_samples_have_rewards(samples)
        except ValueError as e:
            logger.warning("Native grading incomplete for %s: %s", sample_id, e)
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                samples=samples,
                err_message=str(e),
                err_category=RolloutErrorCategory.VALIDATION_ERROR,
            )
        return ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples)

    @staticmethod
    def _extract_rewards(trial_result: TrialResult) -> dict[str, float | int] | None:
        """Read verifier rewards, tolerating single- vs multi-step trials."""
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
        # Pick the configured reward channel, else the sole value when unambiguous.
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

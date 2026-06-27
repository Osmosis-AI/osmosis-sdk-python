"""Native Harbor execution backend: drive one harbor Trial per rollout and map
its verifier reward onto the rollout's single sample. The agent is fixed per
backend; only the task and model vary per rollout via metadata."""

import importlib
import logging
import shutil
import traceback
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
        self.trials_dir = Path(trials_dir)
        self.task_resolver: TaskResolver = task_resolver or resolve_task
        self.environment_config = environment_config or HarborEnvironmentConfig()
        self.cleanup_successful_trials = cleanup_successful_trials
        self._max_concurrency = max_concurrent
        self._queue = TrialQueue(n_concurrent=max_concurrent, retry_config=retry_config)

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
        workflow_result, trial_result = await self._run_trial(request, ctx)
        await self._safe_callback(
            on_workflow_complete, workflow_result, request.id, "workflow"
        )

        if on_grader_complete is None:
            return

        grader_result = self._build_grader_result(
            request, workflow_result, trial_result
        )
        await self._safe_callback(
            on_grader_complete, grader_result, request.id, "grader"
        )

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

    async def _run_trial(
        self, request: ExecutionRequest, ctx: RolloutContext
    ) -> tuple[ExecutionResult, TrialResult | None]:
        """Build and run the trial. Never raises; returns (workflow result, harbor
        TrialResult or None when setup failed before any result)."""
        trial_name = f"{TRIAL_NAME_PREFIX}{request.id}"
        try:
            task_cfg = self.task_resolver(request)
            agent_cfg = self._build_agent_config(request, ctx)
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
                    err_message=str(e),
                    err_category=_categorize_exception(e),
                ),
                None,
            )

        # Harbor reports in-trial failures via exception_info, not by raising; keep the dir.
        if result.exception_info is not None:
            err = result.exception_info
            logger.debug(
                "Native trial %s failed inside harbor: %s: %s",
                request.id,
                getattr(err, "exception_type", None),
                getattr(err, "exception_message", None),
            )
            return (
                ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    err_message=getattr(err, "exception_message", None)
                    or "Trial failed before completion",
                    err_category=RolloutErrorCategory.AGENT_ERROR,
                ),
                result,
            )

        # Reward came from the in-memory result, so the trial dir is disposable.
        self._cleanup_trial(trial_name)
        return ExecutionResult(status=RolloutStatus.SUCCESS), result

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
        sample = RolloutSample(label=request.label)
        if trial_result is None:
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                err_message=workflow_result.err_message,
                err_category=workflow_result.err_category,
            )

        reward_value = self._pick_reward(self._extract_rewards(trial_result))
        if reward_value is not None:
            sample.reward = float(reward_value)

        try:
            validate_sample_has_reward(sample)
        except ValueError as e:
            logger.warning("Native grading incomplete: %s", e)
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                err_message=str(e),
                err_category=RolloutErrorCategory.VALIDATION_ERROR,
            )
        return ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample)

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

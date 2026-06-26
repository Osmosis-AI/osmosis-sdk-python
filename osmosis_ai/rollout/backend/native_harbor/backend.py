"""Native Harbor execution backend.

Drives each rollout as a harbor Trial: resolve the task from
metadata["harbor_task"], wire the controller endpoint into the agent
(rollout identity rides in the chat-completions URL), run it, and map the
task verifier's reward onto the rollout's single RolloutSample.

The agent is fixed per backend (a built-in by name or a user agent by import
path); only the task and optional model name vary per rollout via metadata.
"""

import importlib
import inspect
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
from osmosis_ai.rollout.context import RolloutContext, get_rollout_context
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
# Each rollout spawns a harbor Trial (often a container); bound concurrency.
DEFAULT_MAX_CONCURRENT = 8
TRIAL_NAME_PREFIX = "native-"

# Built-ins whose default summarization breaks a linear append-only
# trajectory -> kwargs forcing it off, applied up front to fail fast.
_TRAINING_SAFE_BUILTIN_AGENTS: dict[str, dict[str, Any]] = {
    "terminus-2": {
        "import_path": "harbor.agents.terminus_2:Terminus2",
        "kwargs": {"enable_summarize": False, "proactive_summarization_threshold": 0},
    },
}

TaskResolver = Callable[[ExecutionRequest], TaskConfig]


def resolve_task(request: ExecutionRequest) -> TaskConfig:
    """Map metadata["harbor_task"] to a TaskConfig selector: a local path, a
    package "org/name@ref", or git."""
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


def _categorize_exception_type(exc_type: str | None) -> RolloutErrorCategory:
    """Categorize a harbor ExceptionInfo.exception_type name -- the only failure
    signal, since Trial.run() records exceptions instead of raising."""
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
    """Resolve the harbor agent class -- downstream wiring inspects the class,
    not the name. Returns None to let Trial.create raise the canonical error."""
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


def _is_custom_agent(import_path: str | None) -> bool:
    """Whether the agent is user-implemented (import_path outside harbor.*). A
    string check, so it covers classes that cannot be imported here."""
    return import_path is not None and not import_path.startswith("harbor.")


def _training_safe_kwargs_for_builtin(
    name: str | None, cls: type[BaseAgent] | None
) -> dict[str, Any] | None:
    """Training-safe kwargs for a whitelisted built-in, else None. Matches by
    name, or by issubclass for harbor.* import-path addressing."""
    for agent_name, entry in _TRAINING_SAFE_BUILTIN_AGENTS.items():
        if name == agent_name:
            return dict(entry["kwargs"])
        if cls is not None:
            ref = _resolve_agent_class(None, entry["import_path"])
            if ref is not None and issubclass(cls, ref):
                return dict(entry["kwargs"])
    return None


def _unaccepted_kwargs(
    cls: type[BaseAgent] | None, kwargs: dict[str, Any]
) -> list[str]:
    """Names in kwargs the constructor cannot receive (empty if the class is
    unresolved or declares **kwargs)."""
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
    """Drive a harbor Trial per rollout and map its verifier reward."""

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        reward_key: str = DEFAULT_REWARD_KEY,
        trials_dir: Path | str = Path("native_trials"),
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
        # Agent fixed per backend: a built-in by name or a user agent by import
        # path ("module:Class"), not both; defaults to the built-in.
        if agent_name is not None and agent_import_path is not None:
            raise ValueError("set agent_name or agent_import_path, not both")
        if agent_name is None and agent_import_path is None:
            agent_name = DEFAULT_AGENT_NAME
        self.agent_name = agent_name
        self.agent_import_path = agent_import_path
        self.model_name = model_name
        self.reward_key = reward_key
        self.trials_dir = Path(trials_dir)
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
        """Run a callback without letting it escape: a propagating error would
        trip app.py into re-firing both callbacks, breaking fire-exactly-once."""
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
        """Build and run the trial. Never raises; returns a workflow result
        (status/error only -- the sample is a grading concern) and the harbor
        TrialResult (None when creation/setup failed before any result).
        """
        # The rollout id alone names the trial dir (mirrors HarborBackend).
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

        # Harbor swallows failures into exception_info; keep the dir for
        # debugging and categorize by the recorded type (a timeout is not an
        # agent error).
        if result.exception_info is not None:
            err = result.exception_info
            # DEBUG (temporary): harbor only logs the underlying failure at DEBUG
            # before swallowing it; surface type+message. Safe to delete later.
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
                    err_category=_categorize_exception_type(
                        getattr(err, "exception_type", None)
                    ),
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

        # Training-safe gate: built-ins must be whitelisted (kwargs applied
        # below) or fail closed; user-implemented agents are trusted, not gated.
        safe_kwargs: dict[str, Any] = {}
        if self.training_safe and not _is_custom_agent(import_path):
            resolved = _training_safe_kwargs_for_builtin(name, agent_cls)
            if resolved is None:
                label = import_path or name
                raise ValueError(
                    f"agent {label!r} is not on the training-safe whitelist "
                    f"{sorted(_TRAINING_SAFE_BUILTIN_AGENTS)}; use one of those, set "
                    f"training_safe=False, or supply a user-implemented agent via "
                    f"agent_import_path."
                )
            safe_kwargs = resolved

        endpoint = ctx.chat_completions_url
        api_key = ctx.api_key
        kwargs: dict[str, Any] = {}
        env: dict[str, str] = {}

        if _is_installed_agent(agent_cls):
            # Installed CLIs are wired purely via env vars. The rollout id is in
            # the URL path, so OPENAI_BASE_URL alone carries routing identity --
            # no per-rollout headers, which is what lets installed agents run here.
            if endpoint:
                env["OPENAI_BASE_URL"] = endpoint
            else:
                logger.warning(
                    "No chat_completions_url in RolloutContext for rollout %s; "
                    "the installed agent's LLM calls will not reach the policy "
                    "server.",
                    request.id,
                )
            if api_key:
                env["OPENAI_API_KEY"] = api_key
        else:
            # In-process agent: endpoint/key go via kwargs. Rollout identity
            # rides in the endpoint URL path, so no per-call headers are set.
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
            # TODO(temporary): osmosis controllers default to SSE when `stream`
            # is absent, but harbor's in-process agents (terminus-2) parse the
            # body as JSON -> every LLM call fails. A plain stream=False is
            # dropped from the wire body, so inject it via extra_body to send
            # "stream": false verbatim. REMOVE once controllers default to JSON.
            llm_kwargs["extra_body"] = {"stream": False}
            if api_key:
                llm_kwargs["api_key"] = api_key
            if llm_kwargs:
                kwargs["llm_kwargs"] = llm_kwargs
            kwargs.update(safe_kwargs)

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
        request: ExecutionRequest,
        workflow_result: ExecutionResult,
        trial_result: TrialResult | None,
    ) -> ExecutionResult:
        """Grade the rollout's single sample. The reward comes from the harbor
        verifier's in-memory result; on a setup failure (no trial result) the
        workflow error is propagated so the grader wait resolves."""
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
        """Read verifier rewards, or None. Harbor rewards are a named-channel
        dict (dict[str, float | int]), not a scalar, and a multi-step task
        carries one per step; take the trial-level dict if present, else the
        first step's. _pick_reward reduces it to a scalar."""
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
        # Collapse harbor's named-channel rewards to one float: prefer the
        # 'reward' key (harbor's 1D convention), else the sole value. Refuse to
        # guess among multiple channels -- a wrong pick feeds the wrong gradient
        # silently -- and warn instead.
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

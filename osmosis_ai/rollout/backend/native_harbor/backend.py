"""Native Harbor execution backend: drive one harbor Trial per rollout and map
its verifier reward onto the rollout's single sample. The agent is fixed per
backend; only the task and model vary per rollout via metadata."""

import copy
import importlib
import json
import logging
import math
import shutil
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any

from harbor.agents.factory import AgentFactory
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
from osmosis_ai.rollout.trajectory.save import _save_trajectories_with_status
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
_BACKEND_DIAGNOSTIC_NAME = "native_harbor"

# terminus-2 summarizes mid-run, breaking training's append-only trajectory; default
# it off (agent_kwargs override). Default agent only -- other agents are the caller's job.
_TERMINUS_2_DEFAULT_KWARGS: dict[str, Any] = {
    "enable_summarize": False,
    "proactive_summarization_threshold": 0,
}


class _AgentProtocol(StrEnum):
    CHAT_COMPLETIONS = "OpenAI Chat Completions"
    OPENAI_RESPONSES = "OpenAI Responses"
    ANTHROPIC_MESSAGES = "Anthropic Messages"
    NONE = "none"


class _IdentityChannel(StrEnum):
    KWARGS = "kwargs"
    OPENAI_ENV = "openai_env"
    ANTHROPIC_ENV = "anthropic_env"
    NONE = "none"


@dataclass(frozen=True)
class _AgentBinding:
    name: str
    protocol: _AgentProtocol
    identity_channel: _IdentityChannel
    eval_supported: bool
    training_supported: bool
    cli_version: str | None = None
    requires_opt_in: bool = False
    blocker: str | None = None
    warning: str | None = None
    allowed_model_providers: frozenset[str] | None = None


_CUSTOM_CHAT_BINDING = "custom-chat-completions"
_AGENT_BINDINGS: dict[str, _AgentBinding] = {
    DEFAULT_AGENT_NAME: _AgentBinding(
        name=DEFAULT_AGENT_NAME,
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.KWARGS,
        eval_supported=True,
        training_supported=True,
        allowed_model_providers=frozenset({"openai"}),
    ),
    "oracle": _AgentBinding(
        name="oracle",
        protocol=_AgentProtocol.NONE,
        identity_channel=_IdentityChannel.NONE,
        eval_supported=True,
        training_supported=False,
        warning=(
            "The native Harbor oracle binding is eval-only: it emits no model "
            "trajectory, so it is not training-safe."
        ),
    ),
    "opencode": _AgentBinding(
        name="opencode",
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.OPENAI_ENV,
        eval_supported=True,
        training_supported=False,
        cli_version="1.18.9",
        requires_opt_in=True,
        allowed_model_providers=frozenset({"openai"}),
        warning=(
            "The native Harbor opencode binding is unverified and eval-only: "
            "Harbor's baseURL injection and append-only behavior still require E2E "
            "validation. Do not use it for training."
        ),
    ),
    "codex": _AgentBinding(
        name="codex",
        protocol=_AgentProtocol.OPENAI_RESPONSES,
        identity_channel=_IdentityChannel.OPENAI_ENV,
        eval_supported=False,
        training_supported=False,
        cli_version="0.146.0",
        blocker=(
            "the rollout controllers expose only OpenAI Chat Completions; an "
            "OpenAI Responses translation gateway is required"
        ),
    ),
    "claude-code": _AgentBinding(
        name="claude-code",
        protocol=_AgentProtocol.ANTHROPIC_MESSAGES,
        identity_channel=_IdentityChannel.ANTHROPIC_ENV,
        eval_supported=False,
        training_supported=False,
        cli_version="2.1.220",
        blocker=(
            "the rollout controllers expose only OpenAI Chat Completions; an "
            "Anthropic Messages translation gateway is required"
        ),
    ),
    _CUSTOM_CHAT_BINDING: _AgentBinding(
        name=_CUSTOM_CHAT_BINDING,
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.KWARGS,
        eval_supported=True,
        training_supported=False,
        requires_opt_in=True,
        allowed_model_providers=frozenset({"openai"}),
        warning=(
            "The custom Chat Completions binding is unverified and eval-only. "
            "Only use it with an agent that accepts api_base/llm_kwargs identity "
            "wiring, and do not use it for training before an append-only E2E."
        ),
    ),
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
    phase: str = "setup"
    phase_started_at: float | None = field(default_factory=lambda: monotonic())
    phase_timings_sec: dict[str, float] = field(default_factory=dict)
    error_phase: str | None = None
    error_payload: dict[str, Any] | None = None

    def transition_phase(self, phase: str) -> None:
        now = monotonic()
        self._finish_phase(now)
        self.phase = phase
        self.phase_started_at = now

    def finish_phase(self) -> None:
        self._finish_phase(monotonic())

    def timing_snapshot(self) -> dict[str, float]:
        timings = dict(self.phase_timings_sec)
        if self.phase_started_at is not None:
            elapsed = max(0.0, monotonic() - self.phase_started_at)
            timings[self.phase] = timings.get(self.phase, 0.0) + elapsed
        return {key: round(value, 6) for key, value in timings.items()}

    def _finish_phase(self, now: float) -> None:
        if self.phase_started_at is None:
            return
        elapsed = max(0.0, now - self.phase_started_at)
        self.phase_timings_sec[self.phase] = (
            self.phase_timings_sec.get(self.phase, 0.0) + elapsed
        )
        self.phase_started_at = None


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


def _canonical_agent_name_for_import_path(import_path: str) -> str | None:
    """Return a registered Harbor name when an import path bypasses its binding."""
    if ":" not in import_path:
        return None
    module_path, _, class_name = import_path.partition(":")
    try:
        imported = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError):
        return None

    for name in _AGENT_BINDINGS:
        if name == _CUSTOM_CHAT_BINDING:
            continue
        try:
            registered = AgentFactory.get_agent_class(AgentName(name))
        except (KeyError, ValueError, ImportError, AttributeError):
            continue
        if imported is registered:
            return name
    return None


def _opencode_base_url_override(agent_kwargs: dict[str, Any]) -> bool:
    config = agent_kwargs.get("opencode_config")
    if not isinstance(config, dict):
        return False
    providers = config.get("provider")
    if not isinstance(providers, dict):
        return False
    openai = providers.get("openai")
    if not isinstance(openai, dict):
        return False
    options = openai.get("options")
    if not isinstance(options, dict):
        return False
    return any(str(key).lower().replace("_", "") == "baseurl" for key in options)


def _validate_agent_kwargs(
    binding: _AgentBinding, agent_kwargs: dict[str, Any]
) -> None:
    configured_version = agent_kwargs.get("version")
    if (
        binding.cli_version is not None
        and configured_version is not None
        and configured_version != binding.cli_version
    ):
        raise ValueError(
            f"Native Harbor binding {binding.name!r} pins CLI version "
            f"{binding.cli_version}; remove conflicting agent_kwargs"
            f"['version']={configured_version!r}"
        )
    if binding.name == "opencode" and _opencode_base_url_override(agent_kwargs):
        raise ValueError(
            "Native Harbor binding 'opencode' owns "
            "agent_kwargs['opencode_config']['provider']['openai']['options']"
            "['baseURL']; remove that override so model traffic cannot be redirected"
        )


def _identity_env_keys(binding: _AgentBinding) -> frozenset[str]:
    if binding.identity_channel == _IdentityChannel.OPENAI_ENV:
        return frozenset({"OPENAI_BASE_URL", "OPENAI_API_KEY"})
    if binding.identity_channel == _IdentityChannel.ANTHROPIC_ENV:
        return frozenset({"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"})
    return frozenset()


def _validate_agent_env(binding: _AgentBinding, agent_env: dict[str, str]) -> None:
    conflicts = sorted(_identity_env_keys(binding).intersection(agent_env))
    if conflicts:
        raise ValueError(
            f"Native Harbor binding {binding.name!r} owns agent.env identity "
            f"keys {conflicts!r}; remove them so model traffic cannot be redirected"
        )


def _validate_agent_config(agent: AgentConfig) -> None:
    if agent.name is not None and agent.import_path is not None:
        raise ValueError("agent config must set name or import_path, not both")
    if agent.n_concurrent is not None:
        raise ValueError(
            "agent.n_concurrent is unsupported; NativeHarborBackend.max_concurrent "
            "and TrialQueue own rollout concurrency"
        )
    if agent.concurrency_group is not None:
        raise ValueError(
            "agent.concurrency_group is unsupported; NativeHarborBackend.max_concurrent "
            "and TrialQueue own rollout concurrency"
        )
    if agent.resume_trajectory:
        raise ValueError(
            "agent.resume_trajectory is unsupported for single-step native rollouts"
        )


def _validate_model_for_binding(binding: _AgentBinding, model_name: str) -> None:
    allowed = binding.allowed_model_providers
    if allowed is None:
        return
    provider, separator, _ = model_name.partition("/")
    if not separator or provider not in allowed:
        raise ValueError(
            f"Native Harbor binding {binding.name!r} requires a model prefixed by "
            f"one of {sorted(allowed)!r} so it uses {binding.protocol.value}; got "
            f"{model_name!r}"
        )


def _resolve_binding(
    *,
    agent_name: str | None,
    agent_import_path: str | None,
    binding_name: str | None,
    allow_unverified_agent: bool,
) -> _AgentBinding:
    if agent_import_path is not None:
        canonical_name = _canonical_agent_name_for_import_path(agent_import_path)
        if canonical_name is not None:
            raise ValueError(
                f"agent_import_path selects Harbor's built-in {canonical_name!r}; "
                f"use agent_name={canonical_name!r} so its validated protocol "
                "binding and CLI pin cannot be bypassed"
            )
        if binding_name is None:
            raise ValueError(
                "agent_import_path requires binding='custom-chat-completions' "
                "so the agent's wire protocol is explicit"
            )
        if binding_name != _CUSTOM_CHAT_BINDING:
            raise ValueError(
                "agent_import_path only supports binding='custom-chat-completions'"
            )
    else:
        binding_name = binding_name or agent_name
        if binding_name != agent_name:
            raise ValueError(
                f"binding {binding_name!r} does not match agent_name {agent_name!r}; "
                "built-in agents must use their own validated binding"
            )

    selected = _AGENT_BINDINGS.get(binding_name or "")
    if selected is None:
        raise ValueError(
            f"no validated Native Harbor binding for {binding_name!r}; supported "
            f"bindings: {', '.join(sorted(_AGENT_BINDINGS))}"
        )
    if not selected.eval_supported:
        raise ValueError(
            f"Native Harbor binding {selected.name!r} speaks "
            f"{selected.protocol.value}, which is not reachable: {selected.blocker}."
        )
    if selected.requires_opt_in and not allow_unverified_agent:
        raise ValueError(
            f"Native Harbor binding {selected.name!r} is not E2E-verified; pass "
            "allow_unverified_agent=True to opt in for eval only"
        )
    return selected


class NativeHarborBackend(ExecutionBackend):
    """Drive a harbor Trial per rollout and map its verifier reward."""

    def __init__(
        self,
        *,
        agent: AgentConfig | None = None,
        environment: HarborEnvironmentConfig | None = None,
        verifier: VerifierConfig | None = None,
        # Compatibility shims for the original reduced constructor surface.
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        agent_env: dict[str, str] | None = None,
        agent_setup_timeout_sec: float | None = None,
        binding: str | None = None,
        allow_unverified_agent: bool = False,
        model_name: str | None = None,
        reward_key: str = DEFAULT_REWARD_KEY,
        trials_dir: Path | str = Path("native_trials"),
        task_resolver: TaskResolver | None = None,
        environment_config: HarborEnvironmentConfig | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        cleanup_successful_trials: bool = True,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1; the native harbor backend spawns a "
                "harbor Trial (often a container) per rollout, so unbounded "
                "concurrency would exhaust the host."
            )
        if agent_setup_timeout_sec is not None and (
            not math.isfinite(agent_setup_timeout_sec) or agent_setup_timeout_sec <= 0
        ):
            raise ValueError("agent_setup_timeout_sec must be > 0 and finite")

        legacy_agent_fields = {
            "agent_name": agent_name,
            "agent_import_path": agent_import_path,
            "agent_kwargs": agent_kwargs,
            "agent_env": agent_env,
        }
        if agent is not None and any(
            value is not None for value in legacy_agent_fields.values()
        ):
            supplied = sorted(
                key for key, value in legacy_agent_fields.items() if value is not None
            )
            raise ValueError(
                f"agent cannot be combined with legacy constructor fields {supplied!r}"
            )
        if environment is not None and environment_config is not None:
            raise ValueError("environment and environment_config cannot both be set")

        if agent is None:
            if agent_name is not None and agent_import_path is not None:
                raise ValueError("set agent_name or agent_import_path, not both")
            if agent_name is None and agent_import_path is None:
                agent_name = DEFAULT_AGENT_NAME
            agent_template = AgentConfig(
                name=agent_name,
                import_path=agent_import_path,
                kwargs=copy.deepcopy(agent_kwargs or {}),
                env=dict(agent_env or {}),
            )
        else:
            agent_template = agent.model_copy(deep=True)

        _validate_agent_config(agent_template)
        agent_name = agent_template.name
        agent_import_path = agent_template.import_path
        resolved_binding = _resolve_binding(
            agent_name=agent_name,
            agent_import_path=agent_import_path,
            binding_name=binding,
            allow_unverified_agent=allow_unverified_agent,
        )
        _validate_agent_kwargs(resolved_binding, agent_template.kwargs)
        _validate_agent_env(resolved_binding, agent_template.env)

        effective_model_name = (
            model_name
            if model_name is not None
            else agent_template.model_name
            if agent_template.model_name is not None
            else DEFAULT_MODEL_NAME
        )
        _validate_model_for_binding(resolved_binding, effective_model_name)
        agent_template.model_name = effective_model_name

        self._agent_name = agent_name
        self._agent_import_path = agent_import_path
        self._agent_config = agent_template
        self._binding = resolved_binding
        self.agent_setup_timeout_sec = agent_setup_timeout_sec
        if resolved_binding.warning is not None:
            warnings.warn(resolved_binding.warning, UserWarning, stacklevel=2)
        self.model_name = effective_model_name
        self.reward_key = reward_key
        self.trials_dir: Path = Path(trials_dir)
        self.task_resolver: TaskResolver = task_resolver or resolve_task
        environment_template = (
            environment
            if environment is not None
            else environment_config
            if environment_config is not None
            else HarborEnvironmentConfig()
        ).model_copy(deep=True)
        self._environment_config: HarborEnvironmentConfig = (
            apply_managed_skypilot_placement(environment_template)
        )
        verifier_template = (
            verifier.model_copy(deep=True) if verifier is not None else VerifierConfig()
        )
        verifier_template.disable = False
        self._verifier_config = verifier_template
        self.cleanup_successful_trials = cleanup_successful_trials
        self._max_concurrency = max_concurrent
        self.artifact_root: Path = default_artifact_root()
        self._pending: dict[str, _PendingNativeTrial] = {}
        self._queue = TrialQueue(
            n_concurrent=max_concurrent,
            retry_config=RetryConfig(max_retries=0),
        )
        self._queue.on_trial_started(self._on_trial_started)
        self._queue.on_environment_started(self._on_environment_started)
        self._queue.on_agent_started(self._on_agent_started)
        self._queue.on_agent_ended(self._on_agent_ended)
        self._queue.on_verification_started(self._on_verification_started)
        self._queue.on_trial_ended(self._on_trial_ended)
        self._queue.on_trial_cancelled(self._on_trial_cancelled)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def capture_final_result(self) -> bool:
        # Harbor verification always runs as part of the Trial. Capture its final
        # outcome for diagnostics/archive even when no remote grader URL was given.
        return True

    @property
    def agent_name(self) -> str | None:
        return self._agent_name

    @property
    def agent_import_path(self) -> str | None:
        return self._agent_import_path

    @property
    def binding(self) -> str:
        return self._binding.name

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
        callback_error: Exception | None = None
        try:
            try:
                task_cfg = self.task_resolver(request)
                agent_cfg = self._build_agent_config(request, ctx)
                trial_cfg = self._build_trial_config(
                    request, task_cfg, agent_cfg, trial_name
                )
                # submit() = Trial.create + run under the queue's semaphore.
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
                workflow_result = (
                    pending.workflow_result
                    or self._build_workflow_result(
                        pending, trial_result, setup_error=setup_error
                    )
                )
                pending.workflow_result = workflow_result
                callback_error = await self._try_callback(
                    on_workflow_complete, workflow_result, request.id, "workflow"
                )
                pending.workflow_complete_called = callback_error is None

            workflow_result = pending.workflow_result or ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message="Trial ended before the agent result was available",
                err_category=RolloutErrorCategory.AGENT_ERROR,
                extra_fields=self._diagnostic_payload(
                    pending,
                    trial_result,
                    category=RolloutErrorCategory.AGENT_ERROR,
                ),
            )
            final_workflow_result = (
                self._build_workflow_result(
                    pending,
                    trial_result,
                    setup_error=setup_error,
                )
                if setup_error is not None
                else workflow_result
            )
            grader_result = self._build_grader_result(
                pending,
                final_workflow_result,
                trial_result,
            )
            result_to_persist = grader_result
            if on_grader_complete is not None:
                grader_callback_error = await self._try_callback(
                    on_grader_complete, grader_result, request.id, "grader"
                )
                callback_error = callback_error or grader_callback_error

            successful = bool(
                trial_result is not None
                and getattr(trial_result, "exception_info", None) is None
            )
            if result_to_persist.extra_fields is not None:
                # Persist diagnostics for every outcome, even when the agent emitted
                # no ATIF document. The server later overlays controller metrics on
                # any trajectory and rewrites the same diagnostics sidecar.
                persisted = await _save_trajectories_with_status(
                    rollout_id=request.id,
                    result=result_to_persist,
                    request_label=request.label,
                    request_metadata=request.metadata,
                    artifact_root=self.artifact_root,
                )
                if successful and not persisted:
                    # Never delete the source trial after losing the only durable
                    # diagnostics/trajectory archive for a successful rollout.
                    pending.preserve_trial = True

            relocated = self._relocate_trial_artifacts(request.id)
            if (
                successful
                and relocated
                and not pending.preserve_trial
                and self.cleanup_successful_trials
            ):
                self._cleanup_trial(trial_name)
            if callback_error is not None:
                # The server owns the final failure-notification path. Raising only
                # after outputs are safe avoids aborting Harbor mid-trial while still
                # guaranteeing that an unacknowledged callback is not silently lost.
                raise callback_error
        finally:
            self._pending.pop(trial_name, None)

    @staticmethod
    async def _try_callback(
        callback: ResultCallback, result: ExecutionResult, rollout_id: str, label: str
    ) -> Exception | None:
        """Attempt a controller callback without aborting an in-flight Harbor trial.

        Traingate callbacks are idempotent by rollout id. A failed hook delivery can
        therefore be retried after the trial, and a final failure must escape to the
        server's fallback instead of leaving the controller waiting forever.
        """
        try:
            await callback(result)
        except Exception as exc:
            logger.error(
                "Native %s callback for rollout %s failed: %s",
                label,
                rollout_id,
                traceback.format_exc(),
            )
            return exc
        return None

    async def _on_trial_started(self, event: TrialHookEvent) -> None:
        self._transition_pending_phase(event, "trial_setup")

    async def _on_environment_started(self, event: TrialHookEvent) -> None:
        self._transition_pending_phase(event, "environment_setup")

    async def _on_agent_started(self, event: TrialHookEvent) -> None:
        self._transition_pending_phase(event, "agent")

    async def _on_agent_ended(self, event: TrialHookEvent) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is not None and pending.phase == "agent":
            pending.finish_phase()

    async def _on_trial_ended(self, event: TrialHookEvent) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is not None:
            self._capture_error_phase(pending, event.result)
            pending.finish_phase()

    async def _on_trial_cancelled(self, event: TrialHookEvent) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is not None:
            pending.transition_phase("cancelled")
            pending.finish_phase()

    def _transition_pending_phase(self, event: TrialHookEvent, phase: str) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is not None:
            pending.transition_phase(phase)

    def _capture_error_phase(
        self, pending: _PendingNativeTrial, trial_result: TrialResult
    ) -> None:
        if (
            pending.error_phase is None
            and getattr(trial_result, "exception_info", None) is not None
        ):
            pending.error_phase = self._infer_phase(pending, trial_result)

    async def _on_verification_started(self, event: TrialHookEvent) -> None:
        pending = self._pending.get(event.trial_name)
        if pending is None:
            return
        # Single-step Harbor may continue into verification after recording an
        # agent failure. Capture that failure's phase before advancing the hook state.
        self._capture_error_phase(pending, event.result)
        pending.transition_phase("verification")
        if pending.workflow_complete_called:
            return
        if event.result.step_results is not None:
            # Multi-step trials verify after every agent step. Firing here would
            # incorrectly announce workflow completion after the first step.
            return

        result = self._build_workflow_result(pending, event.result)
        pending.workflow_result = result
        callback_error = await self._try_callback(
            pending.on_workflow_complete,
            result,
            pending.request.id,
            "workflow",
        )
        pending.workflow_complete_called = callback_error is None

    def _diagnostic_payload(
        self,
        pending: _PendingNativeTrial,
        trial_result: TrialResult | None,
        *,
        category: RolloutErrorCategory | None = None,
        harbor_exception_type: str | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """Build the callback/archive diagnostics for one native result.

        Failure payloads are cached so the callback, SDK log, grader result, and
        trajectory archive all carry identical content even if later hooks advance
        the timing state.
        """
        if category is not None and pending.error_payload is not None:
            return pending.error_payload

        resolved_phase = (
            phase
            or (pending.error_phase if category is not None else None)
            or self._infer_phase(pending, trial_result)
        )
        payload: dict[str, Any] = {
            "backend": _BACKEND_DIAGNOSTIC_NAME,
            "phase": resolved_phase,
            "harbor_exception_type": harbor_exception_type,
            "category": category.value if category is not None else None,
            "timings_sec": self._phase_timings(pending, trial_result),
        }
        if category is not None:
            pending.error_payload = payload
            logger.error(
                "Native Harbor structured error for rollout %s: %s",
                pending.request.id,
                json.dumps(payload, sort_keys=True),
            )
        return payload

    @staticmethod
    def _infer_phase(
        pending: _PendingNativeTrial, trial_result: TrialResult | None
    ) -> str:
        phase = pending.phase
        if trial_result is None:
            return phase

        # Agent setup has no Harbor queue hook. Its TimingInfo is the only precise
        # signal separating it from the broader environment/setup interval.
        if (
            phase == "environment_setup"
            and getattr(trial_result, "agent_setup", None) is not None
        ):
            return "agent_setup"
        if phase != "setup":
            return phase

        # Older/duck-typed queue integrations may not invoke hooks. Recover the
        # furthest entered phase from Harbor's result timing slots.
        for attr, inferred in (
            ("verifier", "verification"),
            ("agent_execution", "agent"),
            ("agent_setup", "agent_setup"),
            ("environment_setup", "environment_setup"),
        ):
            if getattr(trial_result, attr, None) is not None:
                return inferred
        return phase

    @classmethod
    def _phase_timings(
        cls, pending: _PendingNativeTrial, trial_result: TrialResult | None
    ) -> dict[str, float]:
        timings = pending.timing_snapshot()
        if trial_result is None:
            return timings

        exact: dict[str, float | None] = {
            "trial": cls._duration_sec(
                getattr(trial_result, "started_at", None),
                getattr(trial_result, "finished_at", None),
            )
        }
        for attr, key in (
            ("environment_setup", "environment_setup"),
            ("agent_setup", "agent_setup"),
            ("agent_execution", "agent"),
            ("verifier", "verification"),
        ):
            timing = getattr(trial_result, attr, None)
            exact[key] = cls._duration_sec(
                getattr(timing, "started_at", None),
                getattr(timing, "finished_at", None),
            )
        timings.update(
            {key: round(value, 6) for key, value in exact.items() if value is not None}
        )
        return timings

    @staticmethod
    def _duration_sec(started_at: Any, finished_at: Any) -> float | None:
        if started_at is None or finished_at is None:
            return None
        try:
            duration = float((finished_at - started_at).total_seconds())
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(duration):
            return None
        return max(0.0, duration)

    def _build_workflow_result(
        self,
        pending: _PendingNativeTrial,
        trial_result: TrialResult | None,
        *,
        setup_error: Exception | None = None,
    ) -> ExecutionResult:
        request = pending.request
        if setup_error is not None:
            category = _categorize_exception(setup_error)
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=str(setup_error),
                err_category=category,
                extra_fields=self._diagnostic_payload(
                    pending,
                    trial_result,
                    category=category,
                    harbor_exception_type=type(setup_error).__name__,
                ),
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
                extra_fields=self._diagnostic_payload(
                    pending,
                    trial_result,
                    category=RolloutErrorCategory.AGENT_ERROR,
                    harbor_exception_type=getattr(err, "exception_type", None),
                ),
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
            trajectory_document=trajectory_document,
            extra_fields=self._diagnostic_payload(pending, trial_result),
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
        verifier_cfg = self._verifier_config.model_copy(deep=True)
        # The Harbor verifier is the sole reward source for native rollouts.
        verifier_cfg.disable = False
        if request.grader_timeout_sec is not None:
            verifier_cfg.override_timeout_sec = request.grader_timeout_sec
        return TrialConfig(
            task=task_cfg,
            trial_name=trial_name,
            trials_dir=self.trials_dir,
            agent=agent_cfg,
            verifier=verifier_cfg,
            environment=self._environment_config.model_copy(deep=True),
        )

    def _build_agent_config(
        self, request: ExecutionRequest, ctx: RolloutContext
    ) -> AgentConfig:
        md = request.metadata or {}
        agent_cfg = self._agent_config.model_copy(deep=True)
        _validate_agent_config(agent_cfg)
        name = agent_cfg.name
        model_name = md.get(HARBOR_MODEL_KEY, self.model_name)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"rollout {request.id!r} has a harbor_model that is not a "
                "non-empty string"
            )
        _validate_model_for_binding(self._binding, model_name)
        _validate_agent_kwargs(self._binding, agent_cfg.kwargs)
        _validate_agent_env(self._binding, agent_cfg.env)
        # User passthrough is the base layer; SDK-wired values below overlay it.
        kwargs = agent_cfg.kwargs
        env = agent_cfg.env

        endpoint: str | None = None
        api_key = ctx.api_key
        if self._binding.identity_channel != _IdentityChannel.NONE:
            endpoint = ctx.chat_completions_url
            if not endpoint:
                raise ValueError(f"rollout {request.id!r} has no chat_completions_url")
            if uses_local_docker_runtime(self._environment_config):
                endpoint = rewrite_url_for_docker(endpoint)

        if self._binding.cli_version is not None:
            # Binding-owned: installed CLIs must not float to @latest.
            kwargs["version"] = self._binding.cli_version

        if self._binding.identity_channel == _IdentityChannel.OPENAI_ENV:
            assert endpoint is not None
            env["OPENAI_BASE_URL"] = endpoint
            if api_key:
                env["OPENAI_API_KEY"] = api_key
        elif self._binding.identity_channel == _IdentityChannel.ANTHROPIC_ENV:
            assert endpoint is not None
            env["ANTHROPIC_BASE_URL"] = endpoint
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
        elif self._binding.identity_channel == _IdentityChannel.KWARGS:
            assert endpoint is not None
            # Precedence low -> high: default-agent kwargs, user kwargs, SDK wiring.
            defaults = _TERMINUS_2_DEFAULT_KWARGS if name == DEFAULT_AGENT_NAME else {}
            kwargs = {**defaults, **kwargs}
            # Identity: api_base kwarg + api_key in llm_kwargs (deep-merged).
            kwargs["api_base"] = endpoint
            llm_kwargs: dict[str, Any] = dict(kwargs.get("llm_kwargs") or {})
            if api_key:
                llm_kwargs["api_key"] = api_key
            # Controllers require JSON for this path, so stream=False is binding-owned.
            extra_body: dict[str, Any] = dict(llm_kwargs.get("extra_body") or {})
            extra_body["stream"] = False
            llm_kwargs["extra_body"] = extra_body
            kwargs["llm_kwargs"] = llm_kwargs

        agent_cfg.model_name = model_name
        agent_cfg.kwargs = kwargs
        agent_cfg.env = env
        if self.agent_setup_timeout_sec is not None:
            agent_cfg.override_setup_timeout_sec = self.agent_setup_timeout_sec
        if request.agent_timeout_sec is not None:
            agent_cfg.override_timeout_sec = request.agent_timeout_sec
        return agent_cfg

    def _build_grader_result(
        self,
        pending: _PendingNativeTrial,
        workflow_result: ExecutionResult,
        trial_result: TrialResult | None,
    ) -> ExecutionResult:
        """Grade the rollout's single sample from the harbor verifier's in-memory
        result. On setup failure (no trial result) the workflow error is propagated."""
        request = pending.request
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
                extra_fields=workflow_result.extra_fields,
            )

        err = getattr(trial_result, "exception_info", None)
        if err is not None:
            # A Harbor failure is authoritative even when verification emitted a
            # numeric reward before a later phase failed. Never let that reward
            # revive the rollout for either eval or training.
            sample.reward = None
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample,
                trajectory_document=workflow_result.trajectory_document,
                err_message=getattr(err, "exception_message", None)
                or "Trial failed before grading completed",
                err_category=RolloutErrorCategory.AGENT_ERROR,
                extra_fields=self._diagnostic_payload(
                    pending,
                    trial_result,
                    category=RolloutErrorCategory.AGENT_ERROR,
                    harbor_exception_type=getattr(err, "exception_type", None),
                ),
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
                trajectory_document=workflow_result.trajectory_document,
                err_message=str(e),
                err_category=RolloutErrorCategory.VALIDATION_ERROR,
                extra_fields=self._diagnostic_payload(
                    pending,
                    trial_result,
                    category=RolloutErrorCategory.VALIDATION_ERROR,
                    phase="grading",
                ),
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
            trajectory_document=workflow_result.trajectory_document,
            extra_fields=self._diagnostic_payload(pending, trial_result),
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

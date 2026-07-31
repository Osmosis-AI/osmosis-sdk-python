"""Native Harbor execution backend: drive one harbor Trial per rollout and map
its verifier reward onto the rollout's single sample. The agent is fixed per
backend; only the task and model vary per rollout via metadata."""

import asyncio
import copy
import importlib
import json
import logging
import math
import shutil
import traceback
import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

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
PREWARM_TRIAL_NAME_PREFIX = "native-prewarm-"
_BACKEND_DIAGNOSTIC_NAME = "native_harbor"

# terminus-2 summarizes mid-run, breaking training's append-only trajectory; default
# it off (agent_kwargs override). Default agent only -- other agents are the caller's job.
_TERMINUS_2_DEFAULT_KWARGS: dict[str, Any] = {
    "enable_summarize": False,
    "proactive_summarization_threshold": 0,
}


class _AgentProtocol(StrEnum):
    """Wire protocol an agent speaks to reach the rollout controller.

    The train and eval controllers each expose exactly one model-facing route
    per rollout, ``POST /sessions/{id}/v1/chat/completions``.  An agent whose
    protocol is not reachable there is not admitted; the SDK does not translate
    between protocols.
    """

    CHAT_COMPLETIONS = "OpenAI Chat Completions"
    NONE = "none"


class _IdentityChannel(StrEnum):
    """How a binding's agent accepts the rollout's endpoint and credential.

    ``KWARGS`` suits an in-process agent that takes ``api_base``/``llm_kwargs``
    constructor arguments.  ``OPENAI_ENV`` suits an installed agent that runs in
    the task container and reads ``OPENAI_BASE_URL``/``OPENAI_API_KEY`` from its
    process environment, which is how a custom Harbor agent loop is wired.
    """

    KWARGS = "kwargs"
    OPENAI_ENV = "openai_env"
    NONE = "none"


@dataclass(frozen=True)
class _AgentBinding:
    name: str
    protocol: _AgentProtocol
    identity_channel: _IdentityChannel
    training_supported: bool
    # False only for a binding that drives no model traffic at all (``oracle``
    # runs the task's reference solution).  Such a binding can never be
    # training-supported because it emits no trajectory, so the training-parity
    # admission gate does not apply to it.
    emits_model_traffic: bool = True
    warning: str | None = None
    allowed_model_providers: frozenset[str] | None = None


# Admission is training-parity: a binding is registered here only when the
# training path supports it.  Eval deliberately does not get a wider agent set,
# because an eval-only agent would need protocol support the trainer cannot use.
# `oracle` is the one exception and is not an agent in the model sense.
_CUSTOM_CHAT_BINDING = "custom-chat-completions"
_CUSTOM_INSTALLED_CHAT_BINDING = "custom-installed-chat-completions"
# Bindings for an agent the caller supplies through AgentConfig.import_path.
_CUSTOM_BINDINGS = frozenset({_CUSTOM_CHAT_BINDING, _CUSTOM_INSTALLED_CHAT_BINDING})
_AGENT_BINDINGS: dict[str, _AgentBinding] = {
    DEFAULT_AGENT_NAME: _AgentBinding(
        name=DEFAULT_AGENT_NAME,
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.KWARGS,
        training_supported=True,
        allowed_model_providers=frozenset({"openai"}),
    ),
    "oracle": _AgentBinding(
        name="oracle",
        protocol=_AgentProtocol.NONE,
        identity_channel=_IdentityChannel.NONE,
        training_supported=False,
        emits_model_traffic=False,
        warning=(
            "The native Harbor oracle binding runs the task's reference solution "
            "and emits no model trajectory, so it is not training-safe. Use it to "
            "validate datasets and verifiers."
        ),
    ),
    # A caller's own agent loop, selected by AgentConfig.import_path. Both are
    # registered because they speak the one protocol the controller serves. The
    # SDK cannot inspect a custom agent's context management, so append-only
    # verification is the caller's responsibility rather than something claimed
    # here. The two differ only in how the agent accepts rollout identity.
    _CUSTOM_CHAT_BINDING: _AgentBinding(
        name=_CUSTOM_CHAT_BINDING,
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.KWARGS,
        training_supported=True,
        allowed_model_providers=frozenset({"openai"}),
        warning=(
            "The custom Chat Completions binding wires api_base/llm_kwargs identity "
            "into an agent the SDK cannot inspect. Only use it with an in-process "
            "agent that accepts that wiring, and confirm the agent keeps one "
            "append-only trajectory before using it for training."
        ),
    ),
    _CUSTOM_INSTALLED_CHAT_BINDING: _AgentBinding(
        name=_CUSTOM_INSTALLED_CHAT_BINDING,
        protocol=_AgentProtocol.CHAT_COMPLETIONS,
        identity_channel=_IdentityChannel.OPENAI_ENV,
        training_supported=True,
        allowed_model_providers=frozenset({"openai"}),
        warning=(
            "The custom installed Chat Completions binding wires OPENAI_BASE_URL/"
            "OPENAI_API_KEY into an agent the SDK cannot inspect. Only use it with "
            "a container-side agent that reads that environment, and confirm the "
            "agent keeps one append-only trajectory before using it for training."
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
        git_commit_id = md.get(GIT_COMMIT_KEY)
        if not git_commit_id or (
            isinstance(git_commit_id, str) and not git_commit_id.strip()
        ):
            logger.warning(
                "Native Harbor rollout %s uses an unpinned git task; set "
                "metadata[%r] to an immutable commit SHA",
                request.id,
                GIT_COMMIT_KEY,
            )
        return TaskConfig(
            git_url=md[GIT_URL_KEY],
            path=Path(task_path) if task_path else None,
            git_commit_id=git_commit_id,
        )

    name, _, ref = str(raw).partition("@")
    if "/" not in name:
        raise ValueError(
            f"metadata[{HARBOR_TASK_KEY!r}]={raw!r} must be a local path "
            "(./, /, ~), a git form (set git_url), or a package 'org/name[@ref]'"
        )
    effective_ref = ref or "latest"
    if effective_ref == "latest":
        logger.warning(
            "Native Harbor rollout %s uses package task %r with mutable ref "
            "%r; pin metadata[%r] to an immutable sha256 digest",
            request.id,
            name,
            effective_ref,
            HARBOR_TASK_KEY,
        )
    return TaskConfig(name=name, ref=effective_ref)


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


def _prewarm_task_label(task: TaskConfig) -> str:
    """Stable, human-readable task identity for aggregate startup failures."""
    if task.name is not None:
        return f"{task.name}@{task.ref}" if task.ref is not None else task.name
    if task.git_url is not None:
        commit = task.git_commit_id or "<unpinned>"
        path = str(task.path) if task.path is not None else "<unknown-path>"
        # Repository URLs may embed credentials. A task path plus immutable commit
        # is sufficient to identify the failed prewarm without leaking the URL.
        return f"git:{path}@{commit}"
    return str(task.path)


def _prewarm_failure_location(config: TrialConfig) -> str:
    """Describe retained details without promising a directory was created."""
    trial_path = config.trials_dir / config.trial_name
    if trial_path.exists():
        return f"inspect preserved trial {trial_path}"
    return "no trial directory was created"


def _harbor_builtin_name_for_import_path(import_path: str) -> str | None:
    """Return the Harbor agent name when an import path resolves to a built-in.

    Checked against Harbor's own registry rather than ``_AGENT_BINDINGS``: an
    agent that was deliberately left out of the binding table (because the
    controller cannot serve its wire protocol) must not become reachable again
    by naming its class through ``import_path``.
    """
    if ":" not in import_path:
        return None
    module_path, _, class_name = import_path.partition(":")
    try:
        imported = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError):
        return None

    for agent_name in AgentName:
        try:
            registered = AgentFactory.get_agent_class(agent_name)
        except (KeyError, ValueError, ImportError, AttributeError):
            continue
        if imported is registered:
            return agent_name.value
    return None


def _identity_env_keys(binding: _AgentBinding) -> frozenset[str]:
    """Environment slots the binding owns, which the caller must not set.

    Deliberately narrow. A custom agent legitimately carries other providers'
    credentials in ``agent.env`` -- it may route only part of its work to the
    rollout's model endpoint -- so only the two slots that select that endpoint
    are owned; everything else passes through untouched.
    """
    if binding.identity_channel == _IdentityChannel.OPENAI_ENV:
        return frozenset({"OPENAI_BASE_URL", "OPENAI_API_KEY"})
    return frozenset()


def _validate_agent_env(binding: _AgentBinding, agent_env: dict[str, str]) -> None:
    conflicts = sorted(_identity_env_keys(binding).intersection(agent_env))
    if conflicts:
        raise ValueError(
            f"Native Harbor binding {binding.name!r} owns agent.env identity "
            f"keys {conflicts!r}; remove them so model traffic cannot be "
            "redirected. Other providers' credentials pass through untouched."
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
) -> _AgentBinding:
    if agent_import_path is not None:
        builtin_name = _harbor_builtin_name_for_import_path(agent_import_path)
        if builtin_name is not None:
            registered = builtin_name in _AGENT_BINDINGS
            detail = (
                f"use agent_name={builtin_name!r} so its validated protocol "
                "binding and CLI pin cannot be bypassed"
                if registered
                else f"{builtin_name!r} has no Native Harbor binding because the "
                "controller cannot serve its wire protocol; import_path must not "
                "reintroduce it"
            )
            raise ValueError(
                f"agent_import_path resolves to Harbor's built-in "
                f"{builtin_name!r}; {detail}"
            )
        if binding_name is None:
            raise ValueError(
                "agent_import_path requires an explicit custom binding so the "
                "agent's wire protocol and identity channel are stated: "
                f"{_CUSTOM_CHAT_BINDING!r} for an in-process agent taking "
                f"api_base/llm_kwargs, {_CUSTOM_INSTALLED_CHAT_BINDING!r} for an "
                "installed agent reading OPENAI_BASE_URL/OPENAI_API_KEY"
            )
        if binding_name not in _CUSTOM_BINDINGS:
            raise ValueError(
                "agent_import_path only supports the custom bindings "
                f"{', '.join(sorted(_CUSTOM_BINDINGS))}; got {binding_name!r}"
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

    # Training-parity admission: eval does not get a wider agent set than
    # training. `oracle` drives no model at all, so the gate does not apply to
    # it; every other binding must be one the trainer can consume. There is
    # deliberately no opt-in flag to wave this through -- an escape hatch here is
    # how eval-only agents accumulate.
    if selected.emits_model_traffic and not selected.training_supported:
        raise ValueError(
            f"Native Harbor binding {selected.name!r} is not supported for "
            "training, and eval does not admit a wider agent set; supported "
            f"bindings: {', '.join(sorted(_AGENT_BINDINGS))}"
        )

    return selected


class NativeHarborBackend(ExecutionBackend):
    """Drive a harbor Trial per rollout and map its verifier reward."""

    model_name: str

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
        model_name: str | None = None,
        reward_key: str = DEFAULT_REWARD_KEY,
        trials_dir: Path | str = Path("native_trials"),
        task_resolver: TaskResolver | None = None,
        environment_config: HarborEnvironmentConfig | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_queue_depth: int | None = None,
        cleanup_successful_trials: bool = True,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1; the native harbor backend spawns a "
                "harbor Trial (often a container) per rollout, so unbounded "
                "concurrency would exhaust the host."
            )
        resolved_max_queue_depth = (
            max_concurrent if max_queue_depth is None else max_queue_depth
        )
        if resolved_max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
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
        )
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
        self._max_queue_depth = resolved_max_queue_depth
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
    def max_queue_depth(self) -> int:
        return self._max_queue_depth

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
            "binding": self._binding.name,
            "binding_protocol": self._binding.protocol.value,
            "protocol_capabilities": [_AgentProtocol.CHAT_COMPLETIONS.value],
            "training_supported": self._binding.training_supported,
            "max_concurrency": self._max_concurrency,
            "max_queue_depth": self._max_queue_depth,
        }

    async def prewarm(self, tasks: Sequence[TaskConfig]) -> None:
        """Run setup-only Harbor trials for ``tasks`` before serving rollouts.

        Prewarm trials share the backend's bounded, zero-retry ``TrialQueue`` but
        never enter the rollout callback or model-routing path. Every task is
        attempted so one startup failure cannot hide compatibility failures in the
        rest of the configured task list.
        """
        if not tasks:
            raise ValueError("prewarm requires at least one Harbor TaskConfig")

        configs = [self._build_prewarm_trial_config(task) for task in tasks]
        logger.info("Prewarming %d native Harbor task(s)", len(configs))
        outcomes = await asyncio.gather(
            *self._queue.submit_batch(configs),
            return_exceptions=True,
        )

        failures: list[str] = []
        for config, outcome in zip(configs, outcomes, strict=True):
            label = _prewarm_task_label(config.task)
            if isinstance(outcome, BaseException):
                # A cancelled server startup must remain cancellation, not be
                # converted into a compatibility failure after sibling tasks end.
                if not isinstance(outcome, Exception):
                    raise outcome
                failures.append(
                    f"{label} [{type(outcome).__name__}]; "
                    f"{_prewarm_failure_location(config)}"
                )
                continue

            exception_info = getattr(outcome, "exception_info", None)
            if exception_info is not None:
                exception_type = (
                    getattr(exception_info, "exception_type", None) or "HarborError"
                )
                failures.append(
                    f"{label} [{exception_type}]; {_prewarm_failure_location(config)}"
                )
                continue

            if self.cleanup_successful_trials:
                self._cleanup_trial(config.trial_name)

        if failures:
            message = (
                f"Native Harbor prewarm failed for {len(failures)} of "
                f"{len(configs)} task(s):\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )
            logger.error(message)
            raise RuntimeError(message)

        logger.info("Prewarmed %d native Harbor task(s)", len(configs))

    def prewarm_lifespan(
        self, tasks: Sequence[TaskConfig]
    ) -> Callable[[object], AbstractAsyncContextManager[None]]:
        """Return an ASGI lifespan that completes ``prewarm`` before startup.

        The task list is cloned when the lifespan is built, so later caller
        mutations cannot change which setup checks a configured server performs.
        """
        if not tasks:
            raise ValueError("prewarm requires at least one Harbor TaskConfig")
        frozen_tasks = tuple(task.model_copy(deep=True) for task in tasks)

        @asynccontextmanager
        async def lifespan(_app: object) -> AsyncIterator[None]:
            await self.prewarm(frozen_tasks)
            yield

        return lifespan

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

    def _build_prewarm_trial_config(self, task_cfg: TaskConfig) -> TrialConfig:
        """Clone constructor templates into one setup-only Harbor TrialConfig."""
        agent_cfg = self._agent_config.model_copy(deep=True)
        _validate_agent_config(agent_cfg)
        _validate_agent_env(self._binding, agent_cfg.env)

        # Installation deliberately omits per-rollout endpoint and credential
        # identity; only the agent's own defaults apply.
        if agent_cfg.name == DEFAULT_AGENT_NAME:
            agent_cfg.kwargs = {
                **_TERMINUS_2_DEFAULT_KWARGS,
                **agent_cfg.kwargs,
            }
        if self.agent_setup_timeout_sec is not None:
            agent_cfg.override_setup_timeout_sec = self.agent_setup_timeout_sec

        verifier_cfg = self._verifier_config.model_copy(deep=True)
        verifier_cfg.disable = True
        return TrialConfig(
            task=task_cfg.model_copy(deep=True),
            trial_name=f"{PREWARM_TRIAL_NAME_PREFIX}{uuid4().hex}",
            trials_dir=self.trials_dir,
            install_only=True,
            agent=agent_cfg,
            verifier=verifier_cfg,
            environment=self._environment_config.model_copy(deep=True),
        )

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
        self,
        request: ExecutionRequest,
        ctx: RolloutContext,
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

        if self._binding.identity_channel == _IdentityChannel.OPENAI_ENV:
            assert endpoint is not None
            # Scoped AgentConfig.env has the highest Harbor merge precedence, so
            # an installed agent reading these through Harbor's env resolution
            # sees this rollout's endpoint rather than host state. Any other
            # variable the caller set, including other providers' credentials,
            # is left exactly as supplied.
            env["OPENAI_BASE_URL"] = endpoint
            if api_key:
                env["OPENAI_API_KEY"] = api_key
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

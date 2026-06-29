import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    RolloutSample,
)


class SampleSource(ABC):
    """Produces the rollout's ``RolloutSample`` for grading.

    A rollout produces exactly one sample (one agent run, one reward). The
    active ``RolloutContext`` holds at most one ``SampleSource``; the
    integration registers a source that wraps its underlying object
    (typically an Agent or a Session) and the backend pulls the sample at
    grading time.
    """

    @abstractmethod
    async def get_sample(self) -> RolloutSample:
        """Return the rollout's current sample."""


rollout_contextvar: ContextVar["RolloutContext | None"] = ContextVar(
    "rollout_contextvar", default=None
)

# Env var names used by the container-side runners
CHAT_COMPLETIONS_URL_ENV = "OSMOSIS_CHAT_COMPLETIONS_URL"
API_KEY_ENV = "OSMOSIS_API_KEY"
ROLLOUT_ID_ENV = "OSMOSIS_ROLLOUT_ID"


@dataclass
class RolloutContext:
    """Ambient context for a rollout execution.

    For local backends, pass URL/key directly.
    For container runners, leave empty and they'll be read from env vars.
    """

    chat_completions_url: str = ""
    api_key: str | None = None
    rollout_id: str = ""
    sample_source: SampleSource | None = None

    def __post_init__(self) -> None:
        if not self.chat_completions_url:
            self.chat_completions_url = os.environ.get(CHAT_COMPLETIONS_URL_ENV, "")
        if not self.api_key:
            self.api_key = os.environ.get(API_KEY_ENV)
        if not self.rollout_id:
            self.rollout_id = os.environ.get(ROLLOUT_ID_ENV, "")

    def __enter__(self) -> "RolloutContext":
        rollout_contextvar.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        rollout_contextvar.set(None)

    def set_sample_source(self, source: SampleSource) -> None:
        """Register the single sample source for this rollout.

        A rollout has exactly one sample; integrations call this from their
        agent/session constructors so the backend can pull the sample at
        grading time. Attempting to register a second source raises, which
        catches the common bug of putting two ``OsmosisMemorySession``s or
        two ``OsmosisStrandsAgent``s in the same workflow.
        """
        if self.sample_source is not None:
            raise ValueError(
                "RolloutContext already has a sample source registered. "
                "A rollout produces one sample; construct a single agent/"
                "session per rollout."
            )
        self.sample_source = source

    async def get_sample(self) -> RolloutSample | None:
        """Return the rollout's sample, or ``None`` if no source was registered."""
        if self.sample_source is None:
            return None
        return await self.sample_source.get_sample()


def get_rollout_context() -> RolloutContext | None:
    return rollout_contextvar.get()


@dataclass
class GraderContext:
    """Context passed to ``Grader.grade``.

    Carries the single sample produced by the rollout. Graders set its
    reward via :meth:`set_reward`.
    """

    label: str | None = None
    sample: RolloutSample | None = None
    project_path: str | None = None
    metadata: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None

    def set_reward(self, reward: float) -> None:
        if self.sample is None:
            raise ValueError("GraderContext has no sample to reward")
        self.sample.reward = reward

    def set_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Set the rollout-level artifacts object (replaces any prior value)."""
        if not isinstance(artifacts, dict):
            raise TypeError("artifacts must be a dict")
        self.artifacts = artifacts


@dataclass
class AgentWorkflowContext[TConfig: AgentWorkflowConfig]:
    prompt: list[dict[str, Any]]
    config: TConfig | None = None
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        prompt: list[dict[str, Any]],
        config: TConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.prompt = prompt
        self.config = config
        self.metadata = metadata


@dataclass
class HarborAgentWorkflowContext[TConfig: AgentWorkflowConfig](
    AgentWorkflowContext[TConfig]
):
    """Context for agent workflow execution under HarborBackend.

    Extends AgentWorkflowContext with the Harbor ``BaseEnvironment``,
    allowing the workflow to interact with the container via
    ``environment.exec()``, ``environment.upload_file()``, etc.
    """

    environment: Any = None

    def __init__(
        self,
        prompt: list[dict[str, Any]],
        config: TConfig,
        environment: Any = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(prompt=prompt, config=config, metadata=metadata)
        self.environment = environment

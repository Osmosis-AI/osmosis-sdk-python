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
    """Produces a ``RolloutSample`` for grading from a framework-specific object.

    Each integration ships a small source class that wraps its underlying
    object (an Agent, a Session, etc.) and produces a sample on demand.
    The source is decoupled from the integration's framework classes so
    they do not need to inherit from this ABC.

    RolloutContext indexes registered sources by name and calls
    ``get_sample`` lazily at sample collection time, passing the
    registration name so the source can stamp it onto the returned
    ``RolloutSample.id``.
    """

    @abstractmethod
    def get_sample(self, name: str) -> RolloutSample:
        """Return the current rollout sample for the given registration name."""

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
    sample_sources: dict[str, SampleSource] = field(default_factory=dict)

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

    def register_sample_source(self, name: str, source: SampleSource) -> None:
        """Register a sample source under the given name. The context calls
        ``source.get_sample(name)`` lazily when ``get_samples()`` is invoked,
        so the source can keep reflecting state that mutates until the
        rollout ends.
        """
        if name in self.sample_sources:
            raise ValueError(
                f"Session with {name} already exists, please give your session "
                "or agent a unique name in the workflow"
            )
        self.sample_sources[name] = source

    def get_samples(self) -> dict[str, RolloutSample]:
        return {
            name: source.get_sample(name)
            for name, source in self.sample_sources.items()
        }


def get_rollout_context() -> RolloutContext | None:
    return rollout_contextvar.get()


@dataclass
class GraderContext:
    label: str | None = None
    samples: dict[str, RolloutSample] = field(default_factory=dict)
    project_path: str | None = None

    def get_samples(self) -> dict[str, RolloutSample]:
        return self.samples

    def set_sample_reward(self, sample_id: str, reward: float) -> None:
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        self.samples[sample_id].reward = reward


@dataclass
class AgentWorkflowContext[TConfig: AgentWorkflowConfig]:
    prompt: list[dict[str, Any]]
    config: TConfig | None = None

    def __init__(
        self,
        prompt: list[dict[str, Any]],
        config: TConfig | None = None,
    ):
        self.prompt = prompt
        self.config = config


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
    ):
        super().__init__(prompt=prompt, config=config)
        self.environment = environment

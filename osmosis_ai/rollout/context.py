import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    RolloutSample,
)

rollout_contextvar: ContextVar["RolloutContext | None"] = ContextVar(
    "rollout_contextvar", default=None
)
TConfig = TypeVar("TConfig", bound=AgentWorkflowConfig)

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
    registered_agents: dict[str, Any] = field(default_factory=dict)
    recorded_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

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

    def register_agent(self, sample_id: str, agent: Any) -> None:
        """Lazy path for frameworks (Strands) whose ``agent.messages`` is
        mutated in place; read at ``get_samples()`` time."""
        self.registered_agents[sample_id] = agent

    def record_sample(self, sample_id: str, messages: list[dict[str, Any]]) -> None:
        """Eager path for frameworks (openai-agents) whose trajectory is
        only observable after the SDK's run call returns."""
        if sample_id in self.recorded_samples or sample_id in self.registered_agents:
            raise ValueError(
                f"sample_id '{sample_id}' already used in this rollout; "
                f"pass sample_id= explicitly to disambiguate."
            )
        self.recorded_samples[sample_id] = messages

    def get_samples(self) -> dict[str, RolloutSample]:
        samples: dict[str, RolloutSample] = {
            sample_id: RolloutSample(id=sample_id, messages=agent.messages)
            for sample_id, agent in self.registered_agents.items()
        }
        for sample_id, messages in self.recorded_samples.items():
            samples[sample_id] = RolloutSample(id=sample_id, messages=messages)
        return samples


def get_rollout_context() -> RolloutContext | None:
    return rollout_contextvar.get()


@dataclass
class GraderContext:
    label: str | None = None
    samples: dict[str, RolloutSample] = field(default_factory=dict)
    workspace_path: str | None = None

    def get_samples(self) -> dict[str, RolloutSample]:
        return self.samples

    def set_sample_reward(self, sample_id: str, reward: float) -> None:
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        self.samples[sample_id].reward = reward


@dataclass
class AgentWorkflowContext(Generic[TConfig]):
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
class HarborAgentWorkflowContext(AgentWorkflowContext[TConfig]):
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

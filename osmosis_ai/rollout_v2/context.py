from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    RolloutSample,
)

rollout_contextvar: ContextVar = ContextVar("rollout_contextvar", default=None)
TConfig = TypeVar("TConfig", bound=AgentWorkflowConfig)


@dataclass
class ControllerAuth:
    api_key: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            "ControllerAuth(api_key=<redacted>)"
            if self.api_key
            else "ControllerAuth(api_key=None)"
        )

    def as_bearer_headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"Authorization": f"Bearer {self.api_key}"}


@dataclass
class RolloutContext:
    rollout_id: str
    chat_completions_url: str
    api_key: str | None = None
    _registered_agents: dict[str, Any] = field(default_factory=dict)

    def __enter__(self):
        rollout_contextvar.set(self)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        rollout_contextvar.set(None)

    def register_agent(self, sample_id: str, agent: Any) -> None:
        self._registered_agents[sample_id] = agent

    def get_samples(self) -> dict[str, RolloutSample]:
        return {
            sample_id: RolloutSample(
                id=sample_id,
                messages=agent.messages,
            )
            for sample_id, agent in self._registered_agents.items()
        }


def get_rollout_context() -> RolloutContext | None:
    return rollout_contextvar.get()


@dataclass
class GraderContext:
    label: str
    samples: dict[str, RolloutSample] = field(default_factory=dict)

    def get_samples(self) -> dict[str, RolloutSample]:
        return self.samples

    def set_sample_reward(self, sample_id: str, reward: float):
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        self.samples[sample_id].reward = reward


@dataclass
class AgentWorkflowContext(Generic[TConfig]):
    """General context for an agent, agnostic of the rollout context."""

    prompt: list[dict[str, Any]]
    config: TConfig

    def __init__(
        self,
        prompt: list[dict[str, Any]],
        config: TConfig,
    ):
        self.prompt = prompt
        self.config = config

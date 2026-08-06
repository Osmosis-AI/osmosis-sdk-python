"""Registered native Harbor agents and how each receives the model endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from harbor.models.trial.config import AgentConfig as HarborAgentConfig


@dataclass(frozen=True)
class NativeAgentBinding:
    wiring: Literal["env", "kwargs", "none"]
    trainable: bool = True
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


NATIVE_AGENTS: dict[str, NativeAgentBinding] = {
    "terminus-2": NativeAgentBinding(
        wiring="kwargs",
        # Summarization rewrites the running context, which forks the token
        # trajectory RL training needs to stay append-only.
        kwargs={"enable_summarize": False},
    ),
    "mini-swe-agent": NativeAgentBinding(
        wiring="env",
        env={"MSWEA_COST_TRACKING": "ignore_errors"},
    ),
    # Runs the task's reference solution with no model traffic; validates
    # datasets and verifiers, never produces training data.
    "oracle": NativeAgentBinding(wiring="none", trainable=False),
}


def native_binding(agent: Any) -> NativeAgentBinding | None:
    """The binding for a registered native agent name; None for workflow agents."""
    if not isinstance(agent, str) or ":" in agent:
        return None
    binding = NATIVE_AGENTS.get(agent)
    if binding is None:
        raise ValueError(
            f"unknown native agent {agent!r}; registered: {sorted(NATIVE_AGENTS)}"
        )
    return binding


def native_agent_config(
    name: str,
    binding: NativeAgentBinding,
    model_name: str,
    url: str,
    api_key: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> HarborAgentConfig:
    kwargs = {**binding.kwargs, **(extra_kwargs or {})}
    if binding.wiring == "none":
        return HarborAgentConfig(
            name=name, model_name=model_name, env=dict(binding.env), kwargs=kwargs
        )
    if binding.wiring == "env":
        env = {**binding.env, "OPENAI_API_BASE": url, "OPENAI_API_KEY": api_key}
        return HarborAgentConfig(
            name=name, model_name=model_name, env=env, kwargs=kwargs
        )
    # Kwargs-wired agents (terminus-2) silently drop a top-level api_key, so
    # the rollout key rides inside llm_kwargs. Endpoint wiring wins over user
    # kwargs: the rollout URL is not optional.
    llm_kwargs = {**kwargs.pop("llm_kwargs", {}), "api_key": api_key}
    return HarborAgentConfig(
        name=name,
        model_name=model_name,
        kwargs={**kwargs, "api_base": url, "llm_kwargs": llm_kwargs},
    )

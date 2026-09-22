"""Registered native Harbor agents and how each receives the model endpoint."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from harbor.models.trial.config import AgentConfig as HarborAgentConfig


@dataclass(frozen=True)
class NativeAgentBinding:
    wiring: Literal["env", "kwargs", "opencode", "none"]
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
    "opencode": NativeAgentBinding(wiring="opencode"),
    # Runs the task's reference solution with no model traffic; validates
    # datasets and verifiers, never produces training data.
    "oracle": NativeAgentBinding(wiring="none", trainable=False),
}


def native_prewarm_agent_config(
    name: str,
    binding: NativeAgentBinding,
    model_name: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> HarborAgentConfig:
    """Setup-only config: installs the agent with no endpoint or credentials."""
    return HarborAgentConfig(
        name=name,
        model_name=model_name,
        env=dict(binding.env),
        kwargs=copy.deepcopy({**binding.kwargs, **(extra_kwargs or {})}),
    )


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
    kwargs = copy.deepcopy({**binding.kwargs, **(extra_kwargs or {})})
    if binding.wiring == "none":
        return HarborAgentConfig(
            name=name, model_name=model_name, env=dict(binding.env), kwargs=kwargs
        )
    if binding.wiring == "opencode":
        if not isinstance(model_name, str):
            raise ValueError("OpenCode model_name must have the form provider/model")
        provider, separator, model_id = model_name.partition("/")
        if not separator or not provider or not model_id:
            raise ValueError("OpenCode model_name must have the form provider/model")
        config = kwargs.setdefault("opencode_config", {})
        # Built-in provider IDs have OpenCode-specific loaders. In particular,
        # "openai" selects sdk.responses even when npm is overridden. A private
        # provider ID keeps the session on the chat-completions protocol.
        session_provider = "osmosis-rollout"
        providers = config.setdefault("provider", {})
        provider_config = providers.pop(provider, {})
        suffix = 0
        while session_provider in providers:
            suffix += 1
            session_provider = f"osmosis-rollout-{suffix}"
        providers[session_provider] = provider_config
        provider_config["npm"] = "@ai-sdk/openai-compatible"
        provider_config.setdefault("options", {}).update(
            {"baseURL": url, "apiKey": "{env:OPENAI_API_KEY}"}
        )
        provider_config.setdefault("models", {}).setdefault(model_id, {})
        # Compaction rewrites history and breaks the training token trajectory.
        config.setdefault("compaction", {}).update({"auto": False, "prune": False})
        return HarborAgentConfig(
            name=name,
            model_name=f"{session_provider}/{model_id}",
            env={
                **binding.env,
                "OPENAI_API_BASE": url,
                "OPENAI_BASE_URL": url,
                "OPENAI_API_KEY": api_key,
            },
            kwargs=kwargs,
        )
    if binding.wiring == "env":
        # mini-swe-agent reads OPENAI_BASE_URL before OPENAI_API_BASE; set both
        # so a host-level value can never outrank the rollout endpoint.
        env = {
            **binding.env,
            "OPENAI_API_BASE": url,
            "OPENAI_BASE_URL": url,
            "OPENAI_API_KEY": api_key,
        }
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

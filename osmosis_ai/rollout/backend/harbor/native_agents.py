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
    # Model providers whose wire protocol the rollout endpoint serves; the
    # model prefix selects litellm's protocol (and, for env-wired agents,
    # which provider's key variables the agent derives). None = unrestricted,
    # for agents that emit no model traffic.
    providers: frozenset[str] | None = None


NATIVE_AGENTS: dict[str, NativeAgentBinding] = {
    "terminus-2": NativeAgentBinding(
        wiring="kwargs",
        # Summarization rewrites the running context, which forks the token
        # trajectory RL training needs to stay append-only.
        kwargs={"enable_summarize": False},
        providers=frozenset({"openai"}),
    ),
    "mini-swe-agent": NativeAgentBinding(
        wiring="env",
        env={"MSWEA_COST_TRACKING": "ignore_errors"},
        providers=frozenset({"openai"}),
    ),
    # Runs the task's reference solution with no model traffic; validates
    # datasets and verifiers, never produces training data.
    "oracle": NativeAgentBinding(wiring="none", trainable=False),
}


def validate_model_for_binding(
    name: str, binding: NativeAgentBinding, model_name: str
) -> None:
    """Reject models whose provider the rollout endpoint cannot serve.

    A dataset row switching the provider prefix would switch the wire protocol
    or route the rollout credential to a provider-derived endpoint.
    """
    if binding.providers is None:
        return
    provider, separator, _ = model_name.partition("/")
    if not separator or provider not in binding.providers:
        raise ValueError(
            f"native agent {name!r} requires a model prefixed with one of "
            f"{sorted(binding.providers)!r} so traffic stays on the rollout "
            f"endpoint; got {model_name!r}"
        )


def native_prewarm_agent_config(
    name: str, binding: NativeAgentBinding, model_name: str
) -> HarborAgentConfig:
    """Setup-only config: installs the agent with no endpoint or credentials."""
    return HarborAgentConfig(
        name=name,
        model_name=model_name,
        env=dict(binding.env),
        kwargs=dict(binding.kwargs),
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
) -> HarborAgentConfig:
    if binding.wiring == "none":
        return HarborAgentConfig(
            name=name,
            model_name=model_name,
            env=dict(binding.env),
            kwargs=dict(binding.kwargs),
        )
    if binding.wiring == "env":
        # mini-swe-agent reads OPENAI_BASE_URL before OPENAI_API_BASE; set
        # both spellings so a host-level value can never outrank the rollout
        # endpoint the credential belongs to.
        env = {
            **binding.env,
            "OPENAI_API_BASE": url,
            "OPENAI_BASE_URL": url,
            "OPENAI_API_KEY": api_key,
        }
        return HarborAgentConfig(
            name=name, model_name=model_name, env=env, kwargs=dict(binding.kwargs)
        )
    # Kwargs-wired agents (terminus-2) silently drop a top-level api_key, so
    # the rollout key rides inside llm_kwargs; controllers require
    # non-streaming responses. Controller-owned values win over user ones.
    kwargs = {**binding.kwargs, "api_base": url}
    llm_kwargs = dict(kwargs.get("llm_kwargs") or {})
    extra_body = dict(llm_kwargs.get("extra_body") or {})
    extra_body["stream"] = False
    llm_kwargs["api_key"] = api_key
    llm_kwargs["extra_body"] = extra_body
    kwargs["llm_kwargs"] = llm_kwargs
    return HarborAgentConfig(name=name, model_name=model_name, kwargs=kwargs)

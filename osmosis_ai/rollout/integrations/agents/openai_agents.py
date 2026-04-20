"""OpenAI Agents SDK integration for Osmosis rollouts.

Mirrors ``strands.py``: construct ``OsmosisOpenAIAgent`` inside
``AgentWorkflow.run()`` and call ``await agent.run(ctx.prompt)``. Header
injection (``x-sample-id`` / ``x-rollout-id``) uses ``HEADERS_OVERRIDE``,
a ContextVar in openai-agents, which is handoff-safe and
concurrency-safe.

Example::

    class MyWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            agent = OsmosisOpenAIAgent(
                name="multiply",
                tools=[multiply_tool],
            )
            await agent.run(ctx.prompt)
"""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, cast

from agents import Agent, RunConfig, Runner
from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from osmosis_ai.rollout.context import get_rollout_context

# openai-agents' Agent.__init__ has ``instructions`` and ``prompt`` as
# positional parameters 5 and 6. Rollout mode routes prompt content
# exclusively via ``ctx.prompt``, so we reject both — including the
# positional form — to avoid silent drift between controller and agent.
_FORBIDDEN_PROMPT_ARG_POSITIONS: dict[int, str] = {
    5: "instructions",
    6: "prompt",
}


def _raise_forbidden_prompt_arg(name: str) -> None:
    raise TypeError(
        f"OsmosisOpenAIAgent does not accept {name} in rollout mode. "
        "Prompt must come from ctx.prompt / controller initial_messages."
    )


def _validate_prompt_sources(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    for position, name in _FORBIDDEN_PROMPT_ARG_POSITIONS.items():
        if len(args) > position and args[position] is not None:
            _raise_forbidden_prompt_arg(name)

    for name in ("instructions", "prompt"):
        if kwargs.get(name) is not None:
            _raise_forbidden_prompt_arg(name)


class OsmosisOpenAIAgent(Agent):
    """Drop-in replacement for ``agents.Agent`` that wires up the
    rollout-owned Chat Completions endpoint.

    Reads ``RolloutContext`` from the active contextvar at construction
    time, so it MUST be constructed inside ``AgentWorkflow.run()``.
    Prompt semantics stay aligned with Traingate / ``OsmosisStrandsAgent``:
    rollout input comes from ``ctx.prompt`` only, so ``instructions`` and
    ``prompt`` are forbidden. If a ``model`` is passed explicitly, it is
    respected verbatim — advanced users can opt out of the default
    wiring for custom setups.

    Use ``await agent.run(ctx.prompt)`` to execute; it injects per-sample
    headers, forces tracing off, and records the trajectory as a
    ``RolloutSample``.
    """

    def __init__(
        self,
        *args: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        _validate_prompt_sources(args, kwargs)
        if model is None:
            ctx = get_rollout_context()
            if ctx is None:
                raise RuntimeError(
                    "OsmosisOpenAIAgent must be constructed inside an active "
                    "RolloutContext. Construct the agent inside "
                    "AgentWorkflow.run()."
                )
            client = AsyncOpenAI(
                base_url=ctx.chat_completions_url,
                api_key=ctx.api_key or "sk-osmosis-rollout",
            )
            model = OpenAIChatCompletionsModel(
                model="osmosis-rollout",
                openai_client=client,
            )
        kwargs.pop("instructions", None)
        kwargs.pop("prompt", None)
        super().__init__(*args, model=model, **kwargs)

    async def run(
        self,
        input: Any,
        *,
        sample_id: str | None = None,
        run_config: RunConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute this agent as a single rollout sample.

        Injects ``x-rollout-id`` / ``x-sample-id`` into every outbound
        ChatCompletions call via ``HEADERS_OVERRIDE`` (handoff-safe
        because it is a ContextVar), forces ``RunConfig.tracing_disabled``
        to True so rollout usage doesn't leak into the user's other
        openai-agents usage, and records the trajectory via
        ``ctx.record_sample`` after ``Runner.run`` returns.

        ``sample_id`` defaults to ``self.name``; collisions in the same
        rollout auto-suffix with a short uuid and emit a
        ``RuntimeWarning``. Pass ``sample_id=`` to disambiguate
        multi-round use.
        """
        ctx = get_rollout_context()
        if ctx is None:
            raise RuntimeError(
                "OsmosisOpenAIAgent.run requires an active RolloutContext. "
                "Call it from inside AgentWorkflow.run()."
            )

        resolved_sample_id = sample_id or _resolve_sample_id(ctx, self)

        rc = run_config or RunConfig()
        if rc.tracing_disabled is False:
            rc = dataclasses.replace(rc, tracing_disabled=True)

        with _headers_scope(
            {
                "x-rollout-id": ctx.rollout_id,
                "x-sample-id": resolved_sample_id,
            }
        ):
            result = await Runner.run(self, input, run_config=rc, **kwargs)

        ctx.record_sample(
            resolved_sample_id, cast(list[dict[str, Any]], result.to_input_list())
        )
        return result


@contextmanager
def _headers_scope(headers: dict[str, str]):
    """Set HEADERS_OVERRIDE for the duration of a block.

    Merges with any outer HEADERS_OVERRIDE so nested ``agent.run`` calls
    (unusual but not forbidden) compose headers rather than clobbering.
    """
    token = HEADERS_OVERRIDE.set({**(HEADERS_OVERRIDE.get() or {}), **headers})
    try:
        yield
    finally:
        HEADERS_OVERRIDE.reset(token)


def _resolve_sample_id(ctx: Any, agent: Any) -> str:
    base = getattr(agent, "name", None) or "sample"
    if base not in ctx.recorded_samples and base not in ctx.registered_agents:
        return base
    suffix = uuid.uuid4().hex[:6]
    warnings.warn(
        f"sample_id '{base}' already used in this rollout; "
        f"falling back to '{base}-{suffix}'. Pass sample_id= explicitly "
        f"to avoid this.",
        RuntimeWarning,
        stacklevel=3,
    )
    return f"{base}-{suffix}"

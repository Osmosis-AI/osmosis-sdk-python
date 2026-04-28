"""OpenAI Agents SDK integration for Osmosis rollouts.

Use the same ``Agent`` + ``Runner`` pattern as ``openai-agents``. Importing
from this module keeps user code close to the upstream SDK while routing model
calls through the rollout controller:

Example::

    from osmosis_ai.rollout.integrations.agents.openai_agents import Agent, Runner

    class MyWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            agent = Agent(name="multiply", tools=[multiply_tool])
            await Runner.run(agent, ctx.prompt, max_turns=8)
"""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, cast

from agents import Agent, RunConfig
from agents import Runner as OpenAIRunner
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE

from osmosis_ai.rollout.context import RolloutContext, get_rollout_context


class Runner:
    """Osmosis-compatible wrapper around ``agents.Runner``.

    The public call shape mirrors OpenAI Agents: define a normal
    ``agents.Agent`` and execute it with ``await Runner.run(...)``. The wrapper
    supplies the rollout-owned LiteLLM model, per-sample controller headers,
    tracing defaults, and sample recording. It always uses the upstream
    streaming runner because the rollout controller's supported path is SSE
    Chat Completions.
    """

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        *,
        context: Any | None = None,
        max_turns: int = 10,
        hooks: Any | None = None,
        run_config: RunConfig | None = None,
        error_handlers: Any | None = None,
        previous_response_id: str | None = None,
        auto_previous_response_id: bool = False,
        conversation_id: str | None = None,
        session: Any | None = None,
        sample_id: str | None = None,
    ) -> Any:
        """Run an OpenAI agent as one rollout sample over streaming SSE."""
        rollout_ctx = _require_rollout_context("Runner.run")
        resolved_sample_id = _resolve_sample_id(
            rollout_ctx, starting_agent, sample_id=sample_id
        )
        rc = _rollout_run_config(rollout_ctx, run_config)

        with _headers_scope(_rollout_headers(rollout_ctx, resolved_sample_id)):
            result = OpenAIRunner.run_streamed(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=rc,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )
            async for _ in result.stream_events():
                pass

        run_loop_exception = result.run_loop_exception
        if run_loop_exception is not None:
            raise run_loop_exception

        _record_result(rollout_ctx, resolved_sample_id, result)
        return result


@contextmanager
def _headers_scope(headers: dict[str, str]):
    """Set HEADERS_OVERRIDE for the duration of a block.

    Merges with any outer HEADERS_OVERRIDE so nested runner calls compose
    headers rather than clobbering.
    """
    token = HEADERS_OVERRIDE.set({**(HEADERS_OVERRIDE.get() or {}), **headers})
    try:
        yield
    finally:
        HEADERS_OVERRIDE.reset(token)


def _require_rollout_context(api_name: str) -> RolloutContext:
    ctx = get_rollout_context()
    if ctx is None:
        raise RuntimeError(
            f"{api_name} requires an active RolloutContext. "
            "Call it from inside AgentWorkflow.run()."
        )
    return ctx


def _rollout_run_config(ctx: RolloutContext, run_config: RunConfig | None) -> RunConfig:
    rc = run_config or RunConfig()
    updates: dict[str, Any] = {}
    if rc.model is None:
        # Match the Strands integration: hand controller URL/API key to LiteLLM
        # and keep request shaping in the upstream OpenAI Agents adapter.
        updates["model"] = LitellmModel(
            model="openai/osmosis-rollout",
            base_url=ctx.chat_completions_url,
            api_key=ctx.api_key,
        )
    if rc.tracing_disabled is False:
        updates["tracing_disabled"] = True
    return dataclasses.replace(rc, **updates) if updates else rc


def _rollout_headers(ctx: RolloutContext, sample_id: str) -> dict[str, str]:
    return {
        "x-rollout-id": ctx.rollout_id,
        "x-sample-id": sample_id,
    }


def _record_result(ctx: RolloutContext, sample_id: str, result: Any) -> None:
    ctx.record_sample(sample_id, cast(list[dict[str, Any]], result.to_input_list()))


def _resolve_sample_id(
    ctx: RolloutContext, agent: Any, *, sample_id: str | None = None
) -> str:
    if sample_id is not None:
        return sample_id
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


__all__ = ["Agent", "RunConfig", "Runner"]

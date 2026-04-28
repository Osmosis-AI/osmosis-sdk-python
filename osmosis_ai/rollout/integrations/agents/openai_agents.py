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

import asyncio
import contextlib
import dataclasses
import uuid
import warnings
from typing import Any, cast

from agents import Agent, RunConfig
from agents import Runner as OpenAIRunner
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from agents.result import RunResult, RunResultStreaming
from agents.run import DEFAULT_MAX_TURNS

from osmosis_ai.rollout.context import RolloutContext, get_rollout_context


@dataclasses.dataclass(slots=True)
class RolloutModelProvider(ModelProvider):
    """Model provider that keeps explicit model choices and supplies rollout default.

    OpenAI Agents resolves models in this order:
    ``run_config.model`` > ``agent.model`` > ``run_config.model_provider``.
    By implementing the controller model as the provider's ``None`` fallback,
    Osmosis preserves that upstream precedence while still making the rollout
    controller the default model inside a rollout server.
    """

    fallback: ModelProvider
    rollout_context: RolloutContext
    sample_id: str

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            return OsmosisOpenAIRolloutModel.for_sample(
                self.sample_id,
                self.rollout_context,
            )
        return self.fallback.get_model(model_name)

    async def aclose(self) -> None:
        await self.fallback.aclose()


class OsmosisOpenAIRolloutModel(LitellmModel):
    """Controller-backed LiteLLM model for one rollout sample.

    This is intentionally not exported: OpenAI Agents users select the rollout
    controller by leaving the model unspecified, while ``RolloutModelProvider``
    preserves explicit user model choices for ablations.
    """

    def __init__(self, *, headers: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rollout_headers = headers

    @classmethod
    def for_sample(
        cls, sample_id: str, rollout_ctx: RolloutContext
    ) -> OsmosisOpenAIRolloutModel:
        return cls(
            model="openai/osmosis-rollout",
            base_url=rollout_ctx.chat_completions_url,
            api_key=rollout_ctx.api_key,
            headers=_rollout_headers(rollout_ctx, sample_id),
        )

    def _merge_headers(self, model_settings: Any) -> dict[str, str]:
        headers = super()._merge_headers(model_settings)
        return {**headers, **self._rollout_headers}


class Runner:
    """Osmosis-compatible wrapper around ``agents.Runner``.

    The public call shape mirrors OpenAI Agents: define a normal
    ``agents.Agent`` and execute it with ``await Runner.run(...)``. The wrapper
    supplies the rollout-owned LiteLLM model as the default model provider,
    per-sample controller headers, tracing defaults, and sample recording.
    Explicit ``agent.model`` or ``run_config.model`` values keep upstream
    OpenAI Agents routing so ablations can swap in OpenAI or other models.
    """

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        *,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Any | None = None,
        run_config: RunConfig | None = None,
        error_handlers: Any | None = None,
        previous_response_id: str | None = None,
        auto_previous_response_id: bool = False,
        conversation_id: str | None = None,
        session: Any | None = None,
        sample_id: str | None = None,
    ) -> RunResult:
        """Run an OpenAI agent and record the sample when in a rollout.

        This is the recommended entry point for rollout training workflows.
        When the rollout controller default model is used, this method drains
        the upstream streaming run internally so the trainer's SSE-only path is
        respected and sample recording happens before the workflow returns.
        """
        rollout_ctx = get_rollout_context()
        if rollout_ctx is None:
            return await OpenAIRunner.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )

        resolved_sample_id = _resolve_sample_id(
            rollout_ctx, starting_agent, sample_id=sample_id
        )
        base_rc = run_config or RunConfig()

        if _needs_controller_streaming(starting_agent, base_rc):
            result = cls.run_streamed(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
                sample_id=resolved_sample_id,
            )
            async for _ in result.stream_events():
                pass
            return _streaming_to_run_result(result)

        rc = _rollout_run_config(rollout_ctx, run_config, resolved_sample_id)
        result = await OpenAIRunner.run(
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
        _record_result(rollout_ctx, resolved_sample_id, result)
        return result

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        *,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Any | None = None,
        run_config: RunConfig | None = None,
        error_handlers: Any | None = None,
        previous_response_id: str | None = None,
        auto_previous_response_id: bool = False,
        conversation_id: str | None = None,
        session: Any | None = None,
        sample_id: str | None = None,
    ) -> RunResult:
        """Synchronous wrapper matching upstream ``Runner.run_sync``.

        This exists for compatibility with OpenAI Agents scripts. It should not
        be used from ``AgentWorkflow.run()``, which is already async and usually
        runs inside an event loop.
        """
        if get_rollout_context() is None:
            return OpenAIRunner.run_sync(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )

        try:
            already_running_loop = asyncio.get_running_loop()
        except RuntimeError:
            already_running_loop = None

        if already_running_loop is not None:
            raise RuntimeError(
                "Runner.run_sync() cannot be called when an event loop is already running."
            )

        policy = asyncio.get_event_loop_policy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                default_loop = policy.get_event_loop()
            except RuntimeError:
                default_loop = policy.new_event_loop()
                policy.set_event_loop(default_loop)

        # Match upstream: keep the default loop reusable for session/helper state.
        task = default_loop.create_task(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
                sample_id=sample_id,
            )
        )
        try:
            return default_loop.run_until_complete(task)
        except BaseException:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    default_loop.run_until_complete(task)
            raise
        finally:
            if not default_loop.is_closed():
                with contextlib.suppress(RuntimeError):
                    default_loop.run_until_complete(default_loop.shutdown_asyncgens())

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Any | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        auto_previous_response_id: bool = False,
        conversation_id: str | None = None,
        session: Any | None = None,
        *,
        error_handlers: Any | None = None,
        sample_id: str | None = None,
    ) -> RunResultStreaming:
        """Run in streaming mode, recording after the stream completes.

        Advanced users may call this directly, but they must fully consume
        ``result.stream_events()``. Until the stream is drained, the agent run is
        incomplete and no rollout sample has been recorded for grading.
        """
        rollout_ctx = get_rollout_context()
        if rollout_ctx is None:
            return OpenAIRunner.run_streamed(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                error_handlers=error_handlers,
                previous_response_id=previous_response_id,
                auto_previous_response_id=auto_previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )

        resolved_sample_id = _resolve_sample_id(
            rollout_ctx, starting_agent, sample_id=sample_id
        )
        rc = _rollout_run_config(rollout_ctx, run_config, resolved_sample_id)
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
        return _record_after_stream(result, rollout_ctx, resolved_sample_id)


def _rollout_run_config(
    ctx: RolloutContext, run_config: RunConfig | None, sample_id: str
) -> RunConfig:
    rc = run_config or RunConfig()
    fallback_provider = rc.model_provider or MultiProvider()
    return dataclasses.replace(
        rc,
        model_provider=RolloutModelProvider(
            fallback=fallback_provider,
            rollout_context=ctx,
            sample_id=sample_id,
        ),
        tracing_disabled=True,
    )


def _needs_controller_streaming(agent: Agent[Any], run_config: RunConfig) -> bool:
    """Return True when a run may hit the controller-backed default model.

    The rollout controller path supports streaming chat completions. Explicit
    models can use upstream ``Runner.run`` normally, but any agent that falls
    back to the provider default needs the upstream streaming runner.
    """
    if run_config.model is not None:
        return False
    return not _static_agent_graph_has_explicit_models(agent)


def _static_agent_graph_has_explicit_models(
    agent: Agent[Any], seen: set[int] | None = None
) -> bool:
    if seen is None:
        seen = set()
    agent_id = id(agent)
    if agent_id in seen:
        return True
    seen.add(agent_id)

    if getattr(agent, "model", None) is None:
        return False

    for handoff in getattr(agent, "handoffs", []):
        handoff_agent = None
        if isinstance(handoff, Agent):
            handoff_agent = handoff
        else:
            agent_ref = getattr(handoff, "_agent_ref", None)
            if callable(agent_ref):
                handoff_agent = agent_ref()
        if handoff_agent is None:
            return False
        if not _static_agent_graph_has_explicit_models(
            cast(Agent[Any], handoff_agent), seen
        ):
            return False
    return True


def _record_after_stream(result: Any, ctx: RolloutContext, sample_id: str) -> Any:
    stream_events = result.stream_events
    recorded = False

    async def stream_events_with_recording():
        nonlocal recorded
        async for event in stream_events():
            yield event
        _raise_run_loop_exception(result)
        if not recorded:
            _record_result(ctx, sample_id, result)
            recorded = True

    result.stream_events = stream_events_with_recording
    return result


def _raise_run_loop_exception(result: Any) -> None:
    run_loop_exception = getattr(result, "run_loop_exception", None)
    if run_loop_exception is not None:
        raise run_loop_exception


def _streaming_to_run_result(result: Any) -> Any:
    if not isinstance(result, RunResultStreaming):
        return result
    return RunResult(
        input=result.input,
        new_items=result.new_items,
        raw_responses=result.raw_responses,
        final_output=result.final_output,
        input_guardrail_results=result.input_guardrail_results,
        output_guardrail_results=result.output_guardrail_results,
        tool_input_guardrail_results=result.tool_input_guardrail_results,
        tool_output_guardrail_results=result.tool_output_guardrail_results,
        context_wrapper=result.context_wrapper,
        _last_agent=result.last_agent,
        _last_processed_response=result._last_processed_response,
        _tool_use_tracker_snapshot=result._tool_use_tracker_snapshot,
        _current_turn_persisted_item_count=result._current_turn_persisted_item_count,
        _current_turn=result.current_turn,
        _model_input_items=result._model_input_items,
        _original_input=result._original_input,
        _conversation_id=result._conversation_id,
        _previous_response_id=result._previous_response_id,
        _auto_previous_response_id=result._auto_previous_response_id,
        max_turns=result.max_turns,
        interruptions=result.interruptions,
    )


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

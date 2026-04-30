import dataclasses
from typing import Any, cast

from agents import Agent, RunConfig
from agents import Runner as OpenAIRunner
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from agents.result import RunResultBase, RunResultStreaming

from osmosis_ai.rollout.context import RolloutContext, get_rollout_context


class OsmosisRolloutModel(LitellmModel):
    """LiteLLM model pointed at the rollout controller with sample headers."""

    def __init__(self, *, rollout_ctx: RolloutContext, sample_id: str) -> None:
        super().__init__(
            model="openai/osmosis-rollout",
            base_url=rollout_ctx.chat_completions_url,
            api_key=rollout_ctx.api_key,
        )
        self.rollout_ctx = rollout_ctx
        self.sample_id = sample_id

    def _merge_headers(self, model_settings: Any) -> dict[str, str]:
        return {
            **super()._merge_headers(model_settings),
            "x-rollout-id": self.rollout_ctx.rollout_id,
            "x-sample-id": self.sample_id,
        }


class OsmosisRolloutProvider(ModelProvider):
    """Returns the rollout model when no model name is given; defers named lookups.

    Upstream resolves models in the order ``run_config.model > agent.model >
    run_config.model_provider``, so handling only the unnamed case leaves
    explicit user models routed through their original provider.
    """

    def __init__(
        self,
        *,
        fallback: ModelProvider,
        rollout_ctx: RolloutContext,
        sample_id: str,
    ) -> None:
        self.fallback = fallback
        self.rollout_ctx = rollout_ctx
        self.sample_id = sample_id

    def get_model(self, model_name: str | None) -> Model:
        if model_name is not None:
            return self.fallback.get_model(model_name)
        return OsmosisRolloutModel(
            rollout_ctx=self.rollout_ctx, sample_id=self.sample_id
        )


class Runner:
    """Drop-in replacement for ``agents.Runner`` that records rollout samples."""

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        *,
        sample_id: str | None = None,
        **kwargs: Any,
    ) -> RunResultBase:
        rollout_ctx = get_rollout_context()
        if rollout_ctx is None:
            return await OpenAIRunner.run(starting_agent, input, **kwargs)

        sid = sample_id or starting_agent.name
        base_rc = kwargs.pop("run_config", None) or RunConfig()

        # Explicit model bypasses the SSE-only controller; record sample manually.
        if base_rc.model is not None or starting_agent.model is not None:
            run_config = _build_run_config(rollout_ctx, base_rc, sid)
            result = await OpenAIRunner.run(
                starting_agent, input, run_config=run_config, **kwargs
            )
            rollout_ctx.record_sample(
                sid, cast(list[dict[str, Any]], result.to_input_list())
            )
            return result

        streaming = cls.run_streamed(
            starting_agent, input, sample_id=sid, run_config=base_rc, **kwargs
        )
        async for _ in streaming.stream_events():
            pass
        return streaming

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[Any],
        input: Any,
        *,
        sample_id: str | None = None,
        **kwargs: Any,
    ) -> RunResultStreaming:
        rollout_ctx = get_rollout_context()
        run_config = kwargs.pop("run_config", None)
        if rollout_ctx is None:
            return OpenAIRunner.run_streamed(
                starting_agent, input, run_config=run_config, **kwargs
            )

        sid = sample_id or starting_agent.name
        streaming = OpenAIRunner.run_streamed(
            starting_agent,
            input,
            run_config=_build_run_config(rollout_ctx, run_config, sid),
            **kwargs,
        )
        _record_sample_on_drain(streaming, rollout_ctx, sid)
        return streaming

    @classmethod
    def run_sync(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            "Runner.run_sync is not supported; use 'await Runner.run(...)'."
        )


def _build_run_config(
    rollout_ctx: RolloutContext,
    base: RunConfig | None,
    sample_id: str,
) -> RunConfig:
    base = base or RunConfig()
    return dataclasses.replace(
        base,
        model_provider=OsmosisRolloutProvider(
            fallback=base.model_provider or MultiProvider(),
            rollout_ctx=rollout_ctx,
            sample_id=sample_id,
        ),
    )


def _record_sample_on_drain(
    streaming: RunResultStreaming,
    rollout_ctx: RolloutContext,
    sample_id: str,
) -> None:
    """Patch ``stream_events`` so the sample is recorded once the stream drains."""
    inner = streaming.stream_events
    drained = False

    async def stream_events():
        nonlocal drained
        async for event in inner():
            yield event
        if drained:
            return
        drained = True
        if streaming.run_loop_exception is not None:
            raise streaming.run_loop_exception
        rollout_ctx.record_sample(
            sample_id, cast(list[dict[str, Any]], streaming.to_input_list())
        )

    streaming.stream_events = stream_events

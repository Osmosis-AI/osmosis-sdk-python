from typing import Any, cast

from strands import Agent as StrandsAgent
from strands.models.litellm import LiteLLMModel
from strands.models.model import Model
from strands.types.content import Messages

from osmosis_ai.rollout.context import (
    SampleSource,
    get_rollout_context,
)
from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.messages import map_initial_messages_to_content_blocks


class StrandsAgentSampleSource(SampleSource):
    """Produces the rollout sample from a Strands agent's ``messages`` field.

    Strands accumulates the conversation on ``agent.messages`` in
    chat-completion format, so this just wraps that list.
    """

    def __init__(self, agent: StrandsAgent) -> None:
        self.agent = agent

    async def get_sample(self) -> RolloutSample:
        return RolloutSample(messages=list(self.agent.messages))


class OsmosisRolloutModel(LiteLLMModel):
    """Placeholder ``Model`` that carries litellm kwargs for workflow configs.

    Not a usable model on its own: ``OsmosisStrandsAgent`` replaces it with
    a real ``LiteLLMModel`` (wired to the active ``RolloutContext``) at
    agent construction time. It has no connection params, so any direct
    call into the ``Model`` API will fail.

    Subclassing ``LiteLLMModel`` (without invoking its ``__init__``) is
    purely a typing convenience so the placeholder satisfies
    ``StrandsAgent.model: Model`` -- same pattern as
    ``integrations.agents.openai_agents.OsmosisRolloutModel``.
    """

    def __init__(self, **litellm_kwargs: Any) -> None:
        self.litellm_kwargs: dict[str, Any] = litellm_kwargs


class OsmosisStrandsAgent(StrandsAgent):
    """Drop-in ``StrandsAgent`` that wires itself into the active rollout.

    If ``model`` is an ``OsmosisRolloutModel`` placeholder, this materializes
    a real ``LiteLLMModel`` against the active ``RolloutContext`` and
    registers the agent as the rollout's sample source. One
    ``OsmosisStrandsAgent`` per rollout (matching the single-sample model).
    """

    def __init__(
        self,
        *args: Any,
        messages: list[dict[str, Any]] | None = None,
        model: Model | str | None = None,
        **kwargs: Any,
    ) -> None:
        if messages:
            messages = map_initial_messages_to_content_blocks(messages)

        if isinstance(model, OsmosisRolloutModel):
            rollout_ctx = get_rollout_context()
            if rollout_ctx is None:
                raise RuntimeError(
                    "OsmosisRolloutModel requires an active RolloutContext. "
                    "Ensure the execution backend sets up the context before "
                    "running the workflow."
                )
            litellm_model: Model = LiteLLMModel(
                client_args={
                    "api_base": rollout_ctx.chat_completions_url,
                    "api_key": rollout_ctx.api_key,
                },
                model_id="openai/osmosis-rollout",
                **model.litellm_kwargs,
            )
            rollout_ctx.set_sample_source(StrandsAgentSampleSource(self))
            model = litellm_model

        super().__init__(
            *args,
            model=model,
            messages=cast(Messages | None, messages),
            **kwargs,
        )

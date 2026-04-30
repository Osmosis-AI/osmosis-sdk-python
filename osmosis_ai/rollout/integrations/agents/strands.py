import uuid
from collections.abc import AsyncGenerator
from typing import Any, TypeVar, cast

from strands import Agent as StrandsAgent
from strands.models.litellm import LiteLLMModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from osmosis_ai.rollout.context import RolloutContext, get_rollout_context
from osmosis_ai.rollout.utils.messages import map_initial_messages_to_content_blocks

T = TypeVar("T")


class OsmosisRolloutModel(LiteLLMModel):
    """Placeholder model that carries sampling params.

    At runtime, OsmosisStrandsAgent calls for_sample() which reads connection
    info from the active RolloutContext to create a concrete LiteLLMModel.
    """

    def for_sample(self, sample_id: str, rollout_ctx: RolloutContext) -> LiteLLMModel:
        headers = {
            "x-sample-id": sample_id,
            "x-rollout-id": rollout_ctx.rollout_id,
        }
        client_args = {
            "api_base": rollout_ctx.chat_completions_url,
            "api_key": rollout_ctx.api_key,
            "extra_headers": headers,
        }
        return LiteLLMModel(
            client_args=client_args,
            model_id="openai/osmosis-rollout",
            **self.config,  # type: ignore
        )

    def update_config(self, **model_config: Any) -> None:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")

    def get_config(self) -> Any:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")

    def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")

    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")


class OsmosisStrandsAgent(StrandsAgent):
    """Drop-in replacement for StrandsAgent that handles rollout model swap and
    sample registration transparently. The constructor signature matches StrandsAgent.
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
                    "Ensure the execution backend sets up the context before running the workflow."
                )
            sample_id = kwargs.get("name") or kwargs.get("agent_id") or uuid.uuid4().hex
            model = model.for_sample(sample_id, rollout_ctx)
            rollout_ctx.register_agent(sample_id, self)

        super().__init__(
            *args,
            model=model,
            messages=cast(Messages | None, messages),
            **kwargs,
        )

import uuid
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from strands import Agent as StrandsAgent
from strands.models.litellm import LiteLLMModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from osmosis_ai.rollout_v2.context import get_rollout_context
from osmosis_ai.rollout_v2.rollout_sample import RolloutSampleSource
from osmosis_ai.rollout_v2.types import RolloutSample

T = TypeVar("T")


class OsmosisRolloutModel(LiteLLMModel):
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
    def __init__(self, *args, model: Model | str | None = None, **kwargs):
        if (rollout_context := get_rollout_context()) and isinstance(
            model, OsmosisRolloutModel
        ):
            # Substitute the dummy OsmosisRolloutModel with the real litellm rollout model
            sample_id = kwargs.get("name") or kwargs.get("agent_id") or uuid.uuid4().hex
            headers = {
                "x-sample-id": sample_id,
                "x-rollout-id": rollout_context.rollout_id,
            }
            client_args = {
                "api_base": rollout_context.chat_completions_url,
                "api_key": (
                    rollout_context.controller_auth.api_key
                    if rollout_context.controller_auth
                    else None
                ),
                "extra_headers": headers,
            }
            model = LiteLLMModel(
                client_args=client_args,
                model_id="openai/osmosis-rollout",
                **model.config,  # type: ignore
            )
            rollout_context.rollout_sample_sources.append(
                StrandsRolloutSampleSource(self, sample_id)
            )

        super().__init__(*args, model=model, **kwargs)


class StrandsRolloutSampleSource(RolloutSampleSource):
    def __init__(self, agent: OsmosisStrandsAgent, sample_id: str):
        self.agent = agent
        self.sample_id = sample_id

    def make_rollout_sample(self) -> RolloutSample:
        return RolloutSample(
            id=self.sample_id,
            messages=self.get_messages(),
        )

    def get_messages(self) -> list:
        return self.agent.messages

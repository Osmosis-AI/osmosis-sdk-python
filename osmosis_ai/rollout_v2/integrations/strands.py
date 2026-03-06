import uuid
from typing import List, Optional, Any, AsyncGenerator, AsyncIterable, TypeVar, Dict

from strands import Agent as StrandsAgent
from strands.models.litellm import LiteLLMModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec, ToolChoice

from osmosis_ai.rollout_v2.types import RolloutSample
from osmosis_ai.rollout_v2.rollout_sample import RolloutSampleSource
from osmosis_ai.rollout_v2.context import get_rollout_context

T = TypeVar("T")

class OsmosisRolloutModel(LiteLLMModel):
    def update_config(self, **model_config: Any) -> None:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")

    def get_config(self) -> Any:
        raise NotImplementedError("This should not be called for OsmosisRolloutModel")

    def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
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
        if rollout_context := get_rollout_context():
            # Substitute the dummy OsmosisRolloutModel with the real litellm rollout model
            if isinstance(model, OsmosisRolloutModel):
                sample_id = kwargs.get("name", None) or kwargs.get("agent_id", None) or uuid.uuid4().hex
                headers = {
                    "x-sample-id": sample_id,
                    "x-rollout-id": rollout_context.rollout_id,
                }
                client_args = {
                    "api_base": rollout_context.chat_completions_url,
                    # TODO: support API key
                    "api_key": "<api key not supported yet>",
                    "extra_headers": headers,
                }
                model = LiteLLMModel(
                    client_args=client_args,
                    model_id="openai/osmosis-rollout",
                    **model.config, # type: ignore
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

    def get_messages(self) -> List:
        return self.agent.messages
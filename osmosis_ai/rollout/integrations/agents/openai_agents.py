"""OpenAI Agents SDK integration for Osmosis rollouts.

Usage::

    from agents import Agent, Runner
    from osmosis_ai.rollout.integrations.agents.openai_agents import (
        OsmosisRolloutModel,
        OsmosisSession,
    )

    agent = Agent(name="bot", instructions="...", model=OsmosisRolloutModel())
    session = OsmosisSession(name="math-rollout")
    result = await Runner.run(agent, prompt, session=session)

The model is stateless and can be shared across agents and runs. The session
is per-rollout: construct one per ``Runner.run`` so its name reaches the model
via a ContextVar set when the runner reads from the session.
"""

import uuid
from contextvars import ContextVar
from typing import Any, cast

from agents.agent_output import AgentOutputSchemaBase
from agents.extensions.models.litellm_model import LitellmModel
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem
from agents.memory.session import SessionABC
from agents.model_settings import ModelSettings
from agents.models.chatcmpl_converter import Converter
from agents.models.interface import ModelTracing
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from osmosis_ai.rollout.context import SampleSource, get_rollout_context
from osmosis_ai.rollout.types import RolloutSample

current_sample_id: ContextVar[str | None] = ContextVar(
    "osmosis_current_sample_id", default=None
)


class OsmosisSession(SessionABC):
    """In-memory session for a single Osmosis rollout.

    Owns a name, registers itself with the active ``RolloutContext`` (via
    an adapter) for sample collection, and publishes its sample id to a
    ContextVar each time the runner reads from or writes to it.
    ``OsmosisRolloutModel`` reads that ContextVar to stamp per-rollout
    headers.

    Pass exactly one ``OsmosisSession`` per ``Runner.run`` when using
    ``OsmosisRolloutModel``.
    """

    def __init__(self, name: str | None = None) -> None:
        ctx = get_rollout_context()
        if ctx is None:
            raise RuntimeError(
                "OsmosisSession requires an active RolloutContext. "
                "Wrap your workflow in `with RolloutContext(...):`."
            )
        self.name: str = name or uuid.uuid4().hex
        self.session_id: str = self.name
        self.items: list[TResponseInputItem] = []
        ctx.register_sample_source(self.name, OsmosisSessionSampleSource(self))

    async def get_items(
        self, limit: int | None = None
    ) -> list[TResponseInputItem]:
        current_sample_id.set(self.name)
        if limit is None:
            return list(self.items)
        return list(self.items[-limit:])

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        current_sample_id.set(self.name)
        self.items.extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.items.clear()


class OsmosisSessionSampleSource(SampleSource):
    """Produces a ``RolloutSample`` from an OsmosisSession's stored items.

    Decoupled from ``OsmosisSession`` so the session class stays a plain
    ``SessionABC`` implementation and the source can evolve independently
    (e.g., to add system instructions or filter reasoning items) without
    touching the session.
    """

    def __init__(self, session: OsmosisSession) -> None:
        self.session = session

    def get_sample(self, name: str) -> RolloutSample:
        messages = cast(
            list[dict[str, Any]],
            Converter.items_to_messages(list(self.session.items)),
        )
        return RolloutSample(id=name, messages=messages)


class OsmosisRolloutModel(LitellmModel):
    """Streaming-only LitellmModel pre-wired for the Osmosis completions server.

    Stateless: the per-call session name is read from a ContextVar that
    ``OsmosisSession`` publishes when the runner interacts with it. One model
    instance can be safely shared across agents and runs.

    Caveats this class handles for the user:
    - The Osmosis completions server is streaming-only, so ``get_response``
      is implemented by aggregating ``stream_response`` events. This makes
      ``Runner.run`` and ``Runner.run_sync`` work without forcing the user
      to switch to ``Runner.run_streamed``.
    - Per-rollout bookkeeping headers are injected on every model call.
    """

    def __init__(self, **litellm_kwargs: Any) -> None:
        ctx = get_rollout_context()
        if ctx is None:
            raise RuntimeError(
                "OsmosisRolloutModel requires an active RolloutContext. "
                "Wrap your workflow in `with RolloutContext(...):`."
            )
        self.rollout_id = ctx.rollout_id
        super().__init__(
            model="openai/osmosis-rollout",
            base_url=ctx.chat_completions_url,
            api_key=ctx.api_key,
            **litellm_kwargs,
        )

    def _merge_headers(self, model_settings: ModelSettings) -> dict[str, str]:
        sample_id = current_sample_id.get()
        if sample_id is None:
            raise RuntimeError(
                "OsmosisRolloutModel was called without an active OsmosisSession.\n"
                "Pass `session=OsmosisSession()` to Runner.run() so the model "
                "can stamp per-rollout headers and the runner can persist the "
                "conversation for grading.\n\n"
                "Example:\n"
                "    session = OsmosisSession()\n"
                "    result = await Runner.run(agent, prompt, session=session)\n"
            )
        return {
            **super()._merge_headers(model_settings),
            "x-sample-id": sample_id,
            "x-rollout-id": self.rollout_id,
        }

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        # Aggregate streaming events into a non-streaming ModelResponse so
        # Runner.run / Runner.run_sync work against the streaming-only server.
        output_items: list[Any] = []
        usage = Usage()
        async for event in self.stream_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        ):
            if event.type == "response.completed":
                final = event.response
                output_items = list(final.output)
                if final.usage:
                    usage = Usage(
                        requests=1,
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                        total_tokens=final.usage.total_tokens,
                        input_tokens_details=(
                            final.usage.input_tokens_details
                            or InputTokensDetails(cached_tokens=0)
                        ),
                        output_tokens_details=(
                            final.usage.output_tokens_details
                            or OutputTokensDetails(reasoning_tokens=0)
                        ),
                    )
        return ModelResponse(output=output_items, usage=usage, response_id=None)


__all__ = [
    "OsmosisRolloutModel",
    "OsmosisSession",
    "OsmosisSessionSampleSource",
]

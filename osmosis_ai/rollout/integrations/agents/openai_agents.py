import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.memory.session import SessionABC
from agents.model_settings import ModelSettings
from agents.usage import Usage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from osmosis_ai.rollout.context import SampleSource, get_rollout_context
from osmosis_ai.rollout.types import RolloutSample

current_sample_id: ContextVar[str | None] = ContextVar(
    "osmosis_current_sample_id", default=None
)


class OsmosisMemorySession(SessionABC):
    """In-memory session that doubles as the sample source for an Osmosis rollout.

    Behavior:
    - Inside a ``RolloutContext``: registers itself for sample collection
      (via ``SessionSampleSource``) and publishes its sample id
      to a ContextVar each time the runner reads from or writes to it.
      ``OsmosisRolloutModel`` reads that ContextVar to stamp per-rollout
      headers.
    - Outside a ``RolloutContext``: behaves as a plain in-memory
      ``SessionABC`` implementation, so the same workflow code can run
      locally without an Osmosis rollout.

    Pass exactly one ``OsmosisMemorySession`` per ``Runner.run`` when using
    ``OsmosisRolloutModel``.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name: str = name or uuid.uuid4().hex
        self.session_id: str = self.name
        self.items: list[TResponseInputItem] = []
        self._registered_context_ids: set[int] = set()
        self._bind_rollout_context()

    def _bind_rollout_context(self) -> None:
        ctx = get_rollout_context()
        if ctx is None:
            return
        ctx_id = id(ctx)
        if ctx_id in self._registered_context_ids:
            return
        ctx.register_sample_source(self.name, SessionSampleSource(self))
        self._registered_context_ids.add(ctx_id)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        self._bind_rollout_context()
        current_sample_id.set(self.name)
        if limit is None:
            return list(self.items)
        return list(self.items[-limit:])

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        self._bind_rollout_context()
        current_sample_id.set(self.name)
        self.items.extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.items.clear()


class SessionSampleSource(SampleSource):
    """Produces a ``RolloutSample`` from any OpenAI Agents SDK ``Session``.

    Works against any ``SessionABC`` by returning the items the runner
    persisted via ``add_items`` (canonical Responses-API ``TResponseInputItem``
    shape). We deliberately do not run ``Converter.items_to_messages`` here:
    that converter is lossy (collapses reasoning into ``reasoning_content``,
    rewrites ``file_search_call`` into a synthetic function call) and raises
    ``UserError`` for hosted-tool items, ``ItemReference``s, compaction
    items, and any unknown content shape, which would crash sample
    collection on perfectly successful runs. The canonical shape is the
    SDK's stable persisted format and is what graders should walk.
    """

    def __init__(self, session: SessionABC) -> None:
        self.session = session

    async def get_sample(self, name: str) -> RolloutSample:
        items = await self.session.get_items()
        return RolloutSample(id=name, messages=items)


class OsmosisRolloutModel(LitellmModel):
    """Placeholder ``Model`` that carries litellm kwargs for use in workflow configs.

    This is *not* a usable model on its own: ``OsmosisAgent`` replaces it
    with a real :class:`OsmosisLitellmModel` (wired to the active
    ``RolloutContext``) at agent construction time. Any direct call into
    ``stream_response`` / ``get_response`` raises ``NotImplementedError``
    because the placeholder has no connection params.

    Subclassing ``LitellmModel`` (without invoking its ``__init__``) is purely
    a typing convenience so the placeholder satisfies ``Agent.model: Model``.
    """

    def __init__(self, **litellm_kwargs: Any) -> None:
        self.litellm_kwargs = litellm_kwargs

    async def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError(
            "OsmosisRolloutModel is a placeholder. Wrap your agent with "
            "OsmosisAgent so the model is bound to the active RolloutContext."
        )
        yield  # pragma: no cover

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError(
            "OsmosisRolloutModel is a placeholder. Wrap your agent with "
            "OsmosisAgent so the model is bound to the active RolloutContext."
        )


class OsmosisLitellmModel(LitellmModel):
    """Streaming-only LitellmModel pre-wired for the Osmosis completions server.

    Stateless across calls: the per-call session name is read from a
    ContextVar that ``OsmosisMemorySession`` publishes when the runner interacts
    with it. One instance can be safely shared across agents and runs (within
    the rollout context that constructed it).

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
                "OsmosisLitellmModel requires an active RolloutContext. "
                "Construct it inside your workflow run, where the execution "
                "backend has set up the context."
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
                "OsmosisLitellmModel was called without an active OsmosisMemorySession.\n"
                "Pass `session=OsmosisMemorySession()` to Runner.run() so the "
                "model can stamp per-rollout headers and the runner can "
                "persist the conversation for grading."
            )
        return {
            **super()._merge_headers(model_settings),
            "x-sample-id": sample_id,
            "x-rollout-id": self.rollout_id,
        }

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        output_items: list[Any] = []
        usage = Usage()
        async for event in self.stream_response(*args, **kwargs):
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


class OsmosisAgent(Agent):
    """Drop-in replacement for ``Agent`` that materializes ``OsmosisRolloutModel``.

    If the ``model`` argument is an ``OsmosisRolloutModel`` placeholder, it is
    swapped for a fresh ``OsmosisLitellmModel`` bound to the active
    ``RolloutContext``. Otherwise it behaves exactly like ``Agent``.

    Construct inside your workflow ``run`` (where the rollout context is
    active), passing the model from your config.
    """

    def __init__(self, *args: Any, model: Any = None, **kwargs: Any) -> None:
        if isinstance(model, OsmosisRolloutModel):
            model = OsmosisLitellmModel(**model.litellm_kwargs)
        super().__init__(*args, model=model, **kwargs)


__all__ = [
    "OsmosisAgent",
    "OsmosisLitellmModel",
    "OsmosisMemorySession",
    "OsmosisRolloutModel",
    "SessionSampleSource",
]

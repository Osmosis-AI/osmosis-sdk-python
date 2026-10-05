"""Strands agent adapter for Osmosis rollouts.

Install ``osmosis-ai[strands]`` before importing this module.
"""

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from osmosis_ai._imports import raise_optional_dependency_error

try:
    from strands import Agent as StrandsAgent
    from strands.models.litellm import LiteLLMModel
    from strands.models.model import Model
    from strands.types.content import Messages
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="strands",
        feature="The Strands integration",
    )

from osmosis_ai.rollout.context import (
    SampleSource,
    get_rollout_context,
)
from osmosis_ai.rollout.types import RolloutSample

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "OsmosisRolloutModel",
    "OsmosisStrandsAgent",
]

_MEDIA_BLOCKS = ("image", "document", "video")


def _content_block_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert plain-string-content messages to content-block-content messages."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": [{"text": content}]})
        else:
            converted.append(message)
    return converted


def _preview_blocks(blocks: list[Any]) -> list[Any]:
    """Preview formatter input without blocks that warn on every capture.

    The formatter drops reasoning and location-sourced media anyway, and other
    media would only reach a preview as clipped base64.
    """
    result = []
    for block in blocks:
        if isinstance(block, dict):
            if "reasoningContent" in block:
                continue
            media = next((kind for kind in _MEDIA_BLOCKS if kind in block), None)
            tool_result = block.get("toolResult")
            if media is not None:
                if "location" in block[media].get("source", {}):
                    continue
                block = {"text": f"[{media} omitted from preview]"}
            elif isinstance(tool_result, dict) and isinstance(
                tool_result.get("content"), list
            ):
                block = {
                    **block,
                    "toolResult": {
                        **tool_result,
                        "content": _preview_blocks(tool_result["content"]),
                    },
                }
        result.append(block)
    return result


def _preview_messages(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            result.append(message)
            continue
        result.append({**message, "content": _preview_blocks(content)})
        # The formatter moves each tool result's images into a user message of
        # its own; keep that step so preview step IDs match the final trajectory.
        result.extend(
            {"role": "user", "content": [{"text": "[image omitted from preview]"}]}
            for block in content
            if isinstance(block, dict)
            and isinstance(tool_result := block.get("toolResult"), dict)
            and isinstance(tool_result.get("content"), list)
            and any(
                isinstance(item, dict) and "image" in item
                for item in tool_result["content"]
            )
        )
    return result


class StrandsAgentSampleSource(SampleSource):
    """Produces the rollout sample from a Strands agent's ``messages`` field.

    Strands accumulates the conversation on ``agent.messages`` in
    chat-completion format. The source keeps that native history for
    graders and stores a LiteLLM-formatted copy for trajectory
    persistence.
    """

    def __init__(self, agent: StrandsAgent) -> None:
        self.agent = agent

    def _to_trajectory_messages(
        self, messages: Sequence[Mapping[str, Any]], *, log_errors: bool = True
    ) -> Sequence[Mapping[str, Any]] | None:
        try:
            return LiteLLMModel.format_request_messages(cast(Any, list(messages)))
        except Exception:
            if log_errors:
                logger.warning(
                    "Failed to convert Strands messages for trajectory persistence",
                    exc_info=True,
                )
            return None

    async def get_sample(self) -> RolloutSample:
        messages = list(self.agent.messages)
        return RolloutSample(
            messages=messages,
            trajectory_messages=self._to_trajectory_messages(messages),
        )

    async def get_preview(self, max_messages: int) -> RolloutSample | None:
        if max_messages <= 0:
            return None
        messages = copy.deepcopy(self.agent.messages[-max_messages:])
        history_removed = self.agent.conversation_manager.removed_message_count > 0
        return RolloutSample(
            messages=messages,
            trajectory_messages=self._to_trajectory_messages(
                _preview_messages(messages), log_errors=False
            ),
            extra_fields={
                "_preview_truncated": history_removed
                or len(self.agent.messages) > max_messages,
                "_preview_turn": None
                if history_removed
                else sum(
                    message.get("role") == "assistant"
                    for message in self.agent.messages
                ),
            },
        )


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
            messages = _content_block_messages(messages)

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
                # The gateway resolves the actual provider and its supported params.
                model_id="litellm_proxy/osmosis-rollout",
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

"""Protocol definition for LLM clients used by RolloutContext.

Both OsmosisLLMClient (production) and ExternalLLMClient (test mode) satisfy
this interface through structural subtyping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from osmosis_ai.rollout.client import CompletionsResult
    from osmosis_ai.rollout.core.schemas import RolloutMetrics


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol defining the LLM client interface for RolloutContext.

    Both OsmosisLLMClient (production) and ExternalLLMClient (test mode)
    satisfy this interface through structural subtyping.
    """

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> "CompletionsResult":
        """Make a chat completion request.

        Args:
            messages: Full conversation message list (OpenAI format).
            **kwargs: Additional parameters (temperature, max_tokens, tools, etc.).

        Returns:
            CompletionsResult with message, token_ids, logprobs, usage, finish_reason.
        """
        ...

    def get_metrics(self) -> "RolloutMetrics":
        """Return accumulated metrics from this client session.

        Returns:
            RolloutMetrics with timing and token statistics.
        """
        ...


__all__ = ["LLMClientProtocol"]

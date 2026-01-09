"""Protocol definition for LLM clients used by RolloutContext.

This module defines the interface contract that both OsmosisLLMClient (production)
and TestLLMClient (test mode) satisfy through structural subtyping.

Why Protocol instead of ABC?
    - OsmosisLLMClient needs ZERO modifications (structural subtyping / duck typing)
    - Existing user code is completely unaffected
    - Type checkers validate compatibility automatically
    - Clear interface documentation for future maintainers

Example:
    from osmosis_ai.rollout.core.llm_client import LLMClientProtocol

    def create_context(llm: LLMClientProtocol) -> RolloutContext:
        # Works with both OsmosisLLMClient and TestLLMClient
        return RolloutContext(request=..., tools=..., llm=llm)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from osmosis_ai.rollout.client import CompletionsResult
    from osmosis_ai.rollout.core.schemas import RolloutMetrics


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol defining the LLM client interface for RolloutContext.

    This protocol specifies the contract between RolloutContext and LLM clients.
    Both OsmosisLLMClient (production) and TestLLMClient (test mode) satisfy
    this interface through structural subtyping.

    Why Protocol?
        - OsmosisLLMClient requires ZERO modifications (already satisfies protocol)
        - Existing user code is completely unaffected
        - Type checkers (mypy/pyright) validate compatibility at build time
        - Interface changes trigger type errors in all implementations
        - Self-documenting: the protocol IS the documentation

    Note:
        OsmosisLLMClient also has complete_rollout() method for TrainGate
        callbacks, but that's not part of this protocol since TestLLMClient
        doesn't need it (no TrainGate in test mode).

    Example:
        # Both of these satisfy LLMClientProtocol:
        production_client = OsmosisLLMClient(server_url="...", rollout_id="...")
        test_client = OpenAITestClient(api_key="...")

        # RolloutContext accepts either:
        ctx = RolloutContext(request=..., tools=..., llm=production_client)
        ctx = RolloutContext(request=..., tools=..., llm=test_client)
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

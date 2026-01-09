"""Base class for test mode LLM providers.

This module provides the abstract base class for implementing test mode
LLM providers. Each provider implements _do_completion() for their specific
API while the base class handles common functionality.

Key Design - Tools Injection:
    - Production: OsmosisLLMClient doesn't need tools param (TrainGate has them)
    - Test mode: Cloud APIs need tools in each request
    - Solution: set_tools() stores tools, chat_completions() auto-injects them

This keeps the chat_completions() signature compatible with OsmosisLLMClient
while allowing TestLLMClient to pass tools to cloud providers.

Example:
    class MyProvider(TestLLMClient):
        provider_name = "my_provider"

        async def _do_completion(self, messages, **kwargs):
            # Tools are already in kwargs["tools"] if set
            response = await my_api.complete(messages, **kwargs)
            return CompletionsResult(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionToolSchema,
    RolloutMetrics,
)


class TestLLMClient(ABC):
    """Abstract base class for test mode LLM providers.

    Satisfies LLMClientProtocol through structural subtyping.
    Returns CompletionsResult objects fully compatible with RolloutContext.chat().

    Key Design - Tools Injection:
        - Production: OsmosisLLMClient doesn't need tools param (TrainGate has them)
        - Test mode: Cloud APIs need tools in each request
        - Solution: set_tools() stores tools, chat_completions() auto-injects them

    This keeps the chat_completions() signature compatible with OsmosisLLMClient
    while allowing TestLLMClient to pass tools to cloud providers.

    Subclasses must:
        1. Set provider_name class attribute
        2. Implement _do_completion() for their API
        3. Call _record_usage() to track metrics

    Example:
        class OpenAITestClient(TestLLMClient):
            provider_name = "openai"

            def __init__(self, api_key=None, model="gpt-4o"):
                super().__init__()
                self.model = model
                self._client = AsyncOpenAI(api_key=api_key)

            async def _do_completion(self, messages, **kwargs):
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                self._record_usage(latency_ms, prompt_tokens, completion_tokens)
                return CompletionsResult(...)
    """

    # Must be set by subclass
    provider_name: str

    def __init__(self) -> None:
        """Initialize the test LLM client."""
        # Tools storage (injected per test row)
        self._tools: Optional[List[Dict[str, Any]]] = None

        # Metrics tracking (accumulated across calls within a row)
        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0

    def set_tools(self, tools: List[OpenAIFunctionToolSchema]) -> None:
        """Set tools for the current test row.

        Called by TestRunner before each row execution.
        Tools are converted to dict format for API calls.

        Note: We use exclude_none=True because OpenAI API rejects null values
        for optional fields like 'enum' (expects array or absent, not null).

        Args:
            tools: List of tool schemas from agent_loop.get_tools()
        """
        self._tools = [t.model_dump(exclude_none=True) for t in tools] if tools else None

    def clear_tools(self) -> None:
        """Clear tools after test row completion."""
        self._tools = None

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Make a chat completion request.

        Automatically injects tools (set via set_tools()) if not in kwargs.
        This signature matches OsmosisLLMClient for protocol compatibility.

        Args:
            messages: Full conversation message list.
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionsResult compatible with RolloutContext.chat()
        """
        # Auto-inject tools if set and not explicitly provided
        if self._tools is not None and "tools" not in kwargs:
            kwargs["tools"] = self._tools

        return await self._do_completion(messages, **kwargs)

    @abstractmethod
    async def _do_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Provider-specific completion implementation.

        Subclasses implement this for their specific API.
        Tools are already in kwargs["tools"] if applicable.

        Args:
            messages: Conversation messages.
            **kwargs: All parameters including tools.

        Returns:
            CompletionsResult with provider response.
        """
        ...

    def get_metrics(self) -> RolloutMetrics:
        """Return accumulated metrics (satisfies LLMClientProtocol).

        Returns:
            RolloutMetrics with current session statistics.
        """
        return RolloutMetrics(
            llm_latency_ms=self._llm_latency_ms,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    def reset_metrics(self) -> None:
        """Reset metrics to zero. Call this before each test row."""
        self._llm_latency_ms = 0.0
        self._num_llm_calls = 0
        self._prompt_tokens = 0
        self._response_tokens = 0

    def _record_usage(
        self,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record metrics from a completion call.

        Subclasses should call this after each API call.

        Args:
            latency_ms: Request latency in milliseconds.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        """
        self._llm_latency_ms += latency_ms
        self._num_llm_calls += 1
        self._prompt_tokens += prompt_tokens
        self._response_tokens += completion_tokens

    async def close(self) -> None:
        """Release resources. Override in subclass if needed."""
        pass

    async def __aenter__(self) -> "TestLLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = ["TestLLMClient"]

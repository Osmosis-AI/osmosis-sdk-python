"""OpenAI provider for test mode.

This module implements the OpenAI provider for test mode, using the
official OpenAI Python SDK for chat completions with tool calling.

Example:
    from osmosis_ai.rollout.test_mode.providers.openai import OpenAITestClient

    async with OpenAITestClient(model="gpt-4o-mini") as client:
        client.set_tools(tools)
        result = await client.chat_completions(messages)
        print(result.message)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.test_mode.exceptions import ProviderError
from osmosis_ai.rollout.test_mode.providers.base import TestLLMClient

logger = logging.getLogger(__name__)


class OpenAITestClient(TestLLMClient):
    """OpenAI provider for test mode.

    Implements _do_completion() to call OpenAI's chat completions API.
    Tools are automatically injected by the base class via set_tools().

    Features:
        - Full tool calling (function calling) support
        - Streaming disabled for simplicity (batch testing)
        - Metrics tracking for token usage and latency
        - Compatible with any OpenAI-compatible API (via base_url)

    Example:
        client = OpenAITestClient(
            api_key="sk-...",
            model="gpt-4o",
        )

        # Or use environment variable
        client = OpenAITestClient(model="gpt-4o-mini")  # Uses OPENAI_API_KEY

        # Or use OpenAI-compatible API
        client = OpenAITestClient(
            api_key="...",
            model="local-model",
            base_url="http://localhost:8000/v1",
        )
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI test client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY.
            model: Model name to use (default: gpt-4o).
            base_url: Optional base URL for OpenAI-compatible APIs.

        Raises:
            ProviderError: If openai package is not installed.
        """
        super().__init__()

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ProviderError(
                "OpenAI provider requires the 'openai' package. "
                "Install with: pip install openai"
            )

        self.model = model

        # Resolve API key (handle None, empty string, and whitespace-only values)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ProviderError(
                "OpenAI API key required. Provide api_key parameter or "
                "set OPENAI_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )

    async def _do_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Call OpenAI chat completions API.

        Tools are already in kwargs["tools"] if set via set_tools().

        Args:
            messages: Conversation messages in OpenAI format.
            **kwargs: Additional parameters (temperature, max_tokens, tools, etc.)

        Returns:
            CompletionsResult with the LLM response.

        Raises:
            ProviderError: If API call fails.
        """
        start_time = time.monotonic()

        # Extract tools from kwargs (injected by base class)
        tools = kwargs.pop("tools", None)

        # Build request with explicit parameters
        request_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
        }

        # Handle temperature and max_tokens with defaults
        if "temperature" in kwargs:
            request_kwargs["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            request_kwargs["max_tokens"] = kwargs.pop("max_tokens")

        # Add tools if present
        if tools:
            request_kwargs["tools"] = tools

        # Add optional parameters if provided
        if "top_p" in kwargs:
            request_kwargs["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            request_kwargs["stop"] = kwargs.pop("stop")
        if "seed" in kwargs:
            request_kwargs["seed"] = kwargs.pop("seed")

        # Pass any remaining kwargs to OpenAI
        request_kwargs.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Provide more helpful error messages for common errors
            error_str = str(e).lower()
            if "rate" in error_str and "limit" in error_str:
                raise ProviderError(
                    f"OpenAI rate limit exceeded. Try reducing dataset size with --limit, "
                    f"or use a model with higher rate limits (e.g., gpt-4o-mini). "
                    f"Original error: {e}"
                ) from e
            elif "authentication" in error_str or "api key" in error_str:
                raise ProviderError(
                    f"OpenAI authentication failed. Check your API key is valid. "
                    f"Original error: {e}"
                ) from e
            elif "quota" in error_str or "billing" in error_str:
                raise ProviderError(
                    f"OpenAI quota/billing issue. Check your account has available credits. "
                    f"Original error: {e}"
                ) from e
            else:
                raise ProviderError(f"OpenAI API error: {e}") from e

        # Record metrics
        latency_ms = (time.monotonic() - start_time) * 1000
        usage = response.usage
        self._record_usage(
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        # Convert to CompletionsResult
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)

        return CompletionsResult(
            message=message,
            token_ids=[],  # Not needed for testing
            logprobs=[],  # Not needed for testing
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
        )

    async def close(self) -> None:
        """Close the OpenAI async client."""
        if self._client:
            await self._client.close()


__all__ = ["OpenAITestClient"]

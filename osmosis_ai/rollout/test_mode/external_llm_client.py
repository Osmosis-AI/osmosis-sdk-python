"""External LLM client for test mode using LiteLLM.

Supports 100+ providers: OpenAI, Anthropic, Groq, Ollama, etc.

Model format (LiteLLM convention):
    - "gpt-4o" -> auto-prefixed to "openai/gpt-4o"
    - "anthropic/claude-sonnet-4-20250514"
    - "groq/llama-3.1-70b-versatile"
    - "ollama/llama3.1" (with api_base="http://localhost:11434")

See https://docs.litellm.ai/docs/providers for full list.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

from osmosis_ai.rollout.client import CompletionsResult

from osmosis_ai.rollout.core.schemas import RolloutMetrics
from osmosis_ai.rollout.test_mode.exceptions import ProviderError

logger = logging.getLogger(__name__)


def _get_provider_message(e: Exception) -> str:
    """Extract provider message without litellm wrapper."""
    msg = getattr(e, "message", str(e))
    prefix = f"litellm.{type(e).__name__}: "
    if msg.startswith(prefix):
        return msg[len(prefix) :]
    return msg


def _get_litellm():
    """Lazy import LiteLLM to avoid hard dependency.

    Returns:
        litellm module.

    Raises:
        ProviderError: If litellm is not installed.
    """
    try:
        import litellm

        return litellm
    except ImportError:
        raise ProviderError(
            "LiteLLM is required for test mode. "
            "Install with: pip install 'osmosis-ai[test-mode]' or pip install litellm"
        )


class ExternalLLMClient:
    """LLM client wrapping LiteLLM for test mode.

    Handles message format conversion, tool calling translation, and metrics tracking.
    In production, LLM calls go through TrainGate; in test mode, this client
    lets you test against real LLM APIs locally.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize the external LLM client.

        Args:
            model: Model name. Can be:
                - Simple name: "gpt-4o" (auto-prefixed to "openai/gpt-4o")
                - LiteLLM format: "provider/model" (e.g., "anthropic/claude-sonnet-4-20250514")
            api_key: Optional API key (or set via environment variable).
            api_base: Optional custom API base URL (for local models or proxies).

        Raises:
            ProviderError: If litellm package is not installed.
        """
        self._litellm = _get_litellm()

        # Store exception types for error handling (lazy import friendly)
        self._RateLimitError = self._litellm.RateLimitError
        self._AuthenticationError = self._litellm.AuthenticationError
        self._APIError = self._litellm.APIError
        self._BudgetExceededError = self._litellm.BudgetExceededError
        self._Timeout = self._litellm.Timeout
        self._ContextWindowExceededError = self._litellm.ContextWindowExceededError

        # Auto-prefix simple model names with "openai/"
        if "/" not in model:
            model = f"openai/{model}"

        self.model = model
        self._api_key = api_key
        self._api_base = api_base

        # Tools storage (injected per test row)
        self._tools: Optional[List[Dict[str, Any]]] = None

        # Metrics tracking (accumulated across calls within a row)
        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0

    def set_tools(self, tools: List[Any]) -> None:
        """Set tools for the current test row.

        Called by LocalTestRunner before each row execution.
        Tools are converted to dict format for API calls.

        Note: We use exclude_none=True because LLM APIs reject null values
        for optional fields like 'enum' (expects array or absent, not null).

        Args:
            tools: List of tool schemas from agent_loop.get_tools()
        """
        if tools:
            self._tools = [
                t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
                for t in tools
            ]
        else:
            self._tools = None

    def clear_tools(self) -> None:
        """Clear tools after test row completion."""
        self._tools = None

    def reset_metrics(self) -> None:
        """Reset metrics to zero. Call this before each test row."""
        self._llm_latency_ms = 0.0
        self._num_llm_calls = 0
        self._prompt_tokens = 0
        self._response_tokens = 0

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Make a chat completion request.

        LiteLLM automatically handles:
        - Message format conversion (OpenAI <-> Anthropic <-> Gemini)
        - Tool/function calling format translation
        - Error standardization to OpenAI format

        Args:
            messages: Full conversation message list (OpenAI format).
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionsResult with message in OpenAI format.

        Raises:
            ProviderError: If API call fails.
        """
        start_time = time.monotonic()

        # Build request kwargs, starting with defaults
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Add optional authentication/endpoint config
        if self._api_key:
            request_kwargs["api_key"] = self._api_key
        if self._api_base:
            request_kwargs["api_base"] = self._api_base

        # Add tools if set for the client
        if self._tools is not None:
            request_kwargs["tools"] = self._tools

        # Allow any user-provided kwargs to override defaults and add new ones
        request_kwargs.update(kwargs)

        try:
            # Make async call via LiteLLM
            # LiteLLM returns response in OpenAI format regardless of provider
            response = await self._litellm.acompletion(**request_kwargs)
        except self._RateLimitError as e:
            raise ProviderError(
                f"Rate limit exceeded. Try reducing dataset size with --limit. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._AuthenticationError as e:
            raise ProviderError(
                f"Authentication failed. Check your API key is valid. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._BudgetExceededError as e:
            raise ProviderError(
                f"Budget/quota exceeded. Check your account has available credits. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._Timeout as e:
            raise ProviderError(
                f"Request timed out. The model may be slow or network issues occurred. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._ContextWindowExceededError as e:
            raise ProviderError(
                f"Context window exceeded. Try reducing max_turns or message history. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._APIError as e:
            raise ProviderError(
                f"LLM API error: {_get_provider_message(e)}"
            ) from e
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}") from e

        # Record metrics
        latency_ms = (time.monotonic() - start_time) * 1000
        usage = response.usage
        self._llm_latency_ms += latency_ms
        self._num_llm_calls += 1
        self._prompt_tokens += usage.prompt_tokens if usage else 0
        self._response_tokens += usage.completion_tokens if usage else 0

        # Convert to CompletionsResult
        # LiteLLM already returns OpenAI-compatible format
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

    def get_metrics(self) -> RolloutMetrics:
        """Return accumulated metrics.

        Returns:
            RolloutMetrics with current session statistics.
        """
        return RolloutMetrics(
            llm_latency_ms=self._llm_latency_ms,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    async def close(self) -> None:
        """Release resources. LiteLLM manages connections internally."""
        pass

    async def __aenter__(self) -> "ExternalLLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = ["ExternalLLMClient"]

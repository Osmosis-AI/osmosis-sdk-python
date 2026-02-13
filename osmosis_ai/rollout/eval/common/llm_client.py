"""External LLM client for local rollout workflows via LiteLLM.

Supports providers such as OpenAI, Anthropic, Groq, Ollama, and others.
"""

from __future__ import annotations

import logging
import inspect
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
from osmosis_ai.rollout.eval.common.errors import ProviderError

logger = logging.getLogger(__name__)


def _get_provider_message(e: Exception) -> str:
    """Extract provider message without litellm wrapper."""
    msg = getattr(e, "message", str(e))
    prefix = f"litellm.{type(e).__name__}: "
    if msg.startswith(prefix):
        return msg[len(prefix) :]
    return msg


def _get_litellm():
    """Lazy import LiteLLM to avoid hard dependency."""
    try:
        import litellm

        # LiteLLM's atexit cleanup can emit "coroutine was never awaited"
        # warnings on Python 3.12 when asyncio.run() has already closed the loop.
        # We perform explicit async cleanup in ExternalLLMClient.close() instead.
        if hasattr(litellm, "_async_client_cleanup_registered"):
            litellm._async_client_cleanup_registered = True

        return litellm
    except ImportError as exc:
        raise ProviderError(
            "LiteLLM is required for local rollout mode. "
            "Install with: pip install litellm"
        ) from exc


class ExternalLLMClient:
    """LLM client wrapping LiteLLM for local rollout execution."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._litellm = _get_litellm()

        self._RateLimitError = self._litellm.RateLimitError
        self._AuthenticationError = self._litellm.AuthenticationError
        self._APIError = self._litellm.APIError
        self._BudgetExceededError = self._litellm.BudgetExceededError
        self._Timeout = self._litellm.Timeout
        self._ContextWindowExceededError = self._litellm.ContextWindowExceededError

        if "/" not in model:
            model = f"openai/{model}"

        self.model = model
        self._api_key = api_key
        self._api_base = api_base

        self._tools: Optional[List[Dict[str, Any]]] = None

        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0
        self._closed = False

    def set_tools(self, tools: List[Any]) -> None:
        """Set tools for the current execution row."""
        if tools:
            self._tools = [
                t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
                for t in tools
            ]
        else:
            self._tools = None

    def clear_tools(self) -> None:
        """Clear tools after row completion."""
        self._tools = None

    def reset_metrics(self) -> None:
        """Reset metrics before each row."""
        self._llm_latency_ms = 0.0
        self._num_llm_calls = 0
        self._prompt_tokens = 0
        self._response_tokens = 0

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Make a chat completion request via LiteLLM."""
        start_time = time.monotonic()

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self._api_key:
            request_kwargs["api_key"] = self._api_key
        if self._api_base:
            request_kwargs["api_base"] = self._api_base
        if self._tools is not None:
            request_kwargs["tools"] = self._tools

        request_kwargs.update(kwargs)

        try:
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
            raise ProviderError(f"LLM API error: {_get_provider_message(e)}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}") from e

        latency_ms = (time.monotonic() - start_time) * 1000
        usage = response.usage
        self._llm_latency_ms += latency_ms
        self._num_llm_calls += 1
        self._prompt_tokens += usage.prompt_tokens if usage else 0
        self._response_tokens += usage.completion_tokens if usage else 0

        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)

        return CompletionsResult(
            message=message,
            token_ids=[],
            logprobs=[],
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
        )

    def get_metrics(self) -> RolloutMetrics:
        """Return accumulated metrics for the current row."""
        return RolloutMetrics(
            llm_latency_ms=self._llm_latency_ms,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    async def close(self) -> None:
        """Release resources and explicitly close LiteLLM async clients."""
        if self._closed:
            return
        self._closed = True
        self.clear_tools()

        cleanup = getattr(self._litellm, "close_litellm_async_clients", None)
        if not callable(cleanup):
            return

        try:
            cleanup_result = cleanup()
            if inspect.isawaitable(cleanup_result):
                await cleanup_result
        except Exception as exc:
            logger.debug("LiteLLM async client cleanup failed: %s", exc)

    async def __aenter__(self) -> "ExternalLLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

__all__ = ["ExternalLLMClient"]

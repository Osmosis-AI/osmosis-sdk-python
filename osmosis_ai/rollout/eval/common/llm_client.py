"""External LLM client for local rollout workflows via LiteLLM.

Supports providers such as OpenAI, Anthropic, Groq, Ollama, and others.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from osmosis_ai._litellm_compat import APIConnectionError as _APIConnectionError
from osmosis_ai._litellm_compat import AuthenticationError as _AuthenticationError
from osmosis_ai._litellm_compat import BudgetExceededError as _BudgetExceededError
from osmosis_ai._litellm_compat import (
    ContextWindowExceededError as _ContextWindowExceededError,
)
from osmosis_ai._litellm_compat import RateLimitError as _RateLimitError
from osmosis_ai._litellm_compat import Timeout as _LitellmTimeout
from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import RolloutMetrics
from osmosis_ai.rollout.eval.common.errors import ProviderError, SystemicProviderError

logger: logging.Logger = logging.getLogger(__name__)


_NOISE_MARKERS = (
    "\nReceived Model Group=",
    "\nTraceback",
    "\nRequest to ",
    "\ngive]",
)

_MAX_PROVIDER_MSG_LEN = 200


def _get_provider_message(e: Exception) -> str:
    """Extract provider message without litellm wrapper, truncating noise."""
    msg = getattr(e, "message", str(e))
    prefix = f"litellm.{type(e).__name__}: "
    if msg.startswith(prefix):
        msg = msg[len(prefix) :]

    # Strip litellm noise after common markers
    for marker in _NOISE_MARKERS:
        idx = msg.find(marker)
        if idx > 0:
            msg = msg[:idx]
            break

    msg = msg.strip()
    if len(msg) > _MAX_PROVIDER_MSG_LEN:
        msg = msg[:_MAX_PROVIDER_MSG_LEN] + "..."
    return msg


def _is_missing_provider_error(message: str) -> bool:
    """Return True if the message indicates missing/invalid LiteLLM provider."""
    normalized = message.lower()
    return (
        "llm provider not provided" in normalized
        or "pass in the llm provider you are trying to call" in normalized
    )


def _is_connection_error_message(message: str) -> bool:
    """Return True if an APIError message actually indicates a connection failure.

    LiteLLM sometimes wraps connection errors as ``InternalServerError``
    (a subclass of ``APIError``) instead of ``APIConnectionError``.
    """
    normalized = message.lower()
    return (
        "connection error" in normalized
        or "connection refused" in normalized
        or "name or service not known" in normalized
        or "nodename nor servname provided" in normalized
        or "connect call failed" in normalized
    )


def _first_line(message: str) -> str:
    """Return the first non-empty line to avoid multi-line noise."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return message.strip()


def _format_provider_hint(
    model: str,
    api_base: str | None,
    raw_message: str,
) -> str:
    """Build a concise actionable error for provider/model format issues."""
    detail = _first_line(raw_message)
    if api_base:
        return (
            "Cannot connect to custom endpoint. "
            f"Received model='{model}', base_url='{api_base}'. Details: {detail}"
        )
    return (
        "Invalid LiteLLM model format. Use 'provider/model' "
        "(for example: openai/gpt-5-mini). "
        f"Received model='{model}'. Details: {detail}"
    )


def _get_litellm():
    """Lazy import LiteLLM to avoid hard dependency."""
    try:
        import litellm

        # Suppress noisy "Give Feedback" and "LiteLLM.Info" messages that
        # litellm prints to stdout/stderr on every exception.
        litellm.suppress_debug_info = True

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
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._litellm = _get_litellm()

        self._RateLimitError = _RateLimitError
        self._AuthenticationError = _AuthenticationError
        self._BudgetExceededError = _BudgetExceededError
        self._Timeout = _LitellmTimeout
        self._ContextWindowExceededError = _ContextWindowExceededError
        self._APIConnectionError = _APIConnectionError

        # Preserve the user's original model name for display purposes.
        self.display_name = model

        if api_base:
            # Custom endpoint: always route through openai/ provider in litellm.
            if not model.startswith("openai/"):
                model = f"openai/{model}"
        else:
            # Standard litellm routing: auto-prefix bare names with openai/.
            if "/" not in model:
                model = f"openai/{model}"

        self.model = model
        self._api_key = api_key
        self._api_base = api_base

        self._tools: list[dict[str, Any]] | None = None

        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0
        self._closed = False

    @property
    def api_key(self) -> str | None:
        """The API key used for authentication."""
        return self._api_key

    @property
    def api_base(self) -> str | None:
        """The base URL for API requests."""
        return self._api_base

    def set_tools(self, tools: list[Any]) -> None:
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
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Make a chat completion request via LiteLLM."""
        start_time = time.monotonic()

        request_kwargs: dict[str, Any] = {
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
        except self._AuthenticationError as e:
            raise SystemicProviderError(
                f"Authentication failed. Check your API key is valid. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._BudgetExceededError as e:
            raise SystemicProviderError(
                f"Budget/quota exceeded. Check your account has available credits. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._APIConnectionError as e:
            raise SystemicProviderError(
                f"Cannot connect to API endpoint. Check your --base-url or network. "
                f"Details: {_get_provider_message(e)}"
            ) from e
        except self._RateLimitError as e:
            raise ProviderError(
                f"Rate limit exceeded. Try reducing dataset size with --limit. "
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
        except Exception as e:
            raise self._classify_unknown_error(e) from e

        latency_ms = (time.monotonic() - start_time) * 1000
        # litellm's ModelResponse stubs are incomplete — usage/choices/message
        # are valid at runtime but not fully reflected in the published types.
        usage = response.usage  # type: ignore[union-attr]
        self._llm_latency_ms += latency_ms
        self._num_llm_calls += 1
        self._prompt_tokens += usage.prompt_tokens if usage else 0
        self._response_tokens += usage.completion_tokens if usage else 0

        choice = response.choices[0]  # type: ignore[union-attr]
        message = choice.message.model_dump(exclude_none=True)  # type: ignore[union-attr]

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

    async def preflight_check(self) -> None:
        """Send a minimal request to verify connectivity and authentication.

        Raises:
            SystemicProviderError: If the provider is unreachable, authentication
                fails, or the model does not exist.
        """
        try:
            preflight_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
            if self._api_key:
                preflight_kwargs["api_key"] = self._api_key
            if self._api_base:
                preflight_kwargs["api_base"] = self._api_base
            await self._litellm.acompletion(**preflight_kwargs)
        except self._RateLimitError:
            # Rate-limited means the endpoint is reachable and authenticated.
            return
        except (
            self._AuthenticationError,
            self._BudgetExceededError,
            self._APIConnectionError,
        ) as e:
            raise SystemicProviderError(_get_provider_message(e)) from e
        except Exception as e:
            classified = self._classify_unknown_error(e)
            if isinstance(classified, SystemicProviderError):
                raise classified from e
            # Non-systemic errors (transient 5xx, etc.) — don't block.
            logger.debug("Preflight non-fatal error: %s", e)

    def _classify_unknown_error(self, e: Exception) -> ProviderError:
        """Classify an exception not caught by the specific litellm handlers.

        LiteLLM exception classes don't actually inherit from ``litellm.APIError``
        (they inherit from ``openai.*`` counterparts instead), so errors like
        ``InternalServerError`` fall through to the generic ``except Exception``
        block.  This method inspects the error's attributes and message to
        determine the correct :class:`ProviderError` subclass.
        """
        msg = _get_provider_message(e)
        status_code = getattr(e, "status_code", None)

        # Systemic: auth / forbidden / model-not-found
        if status_code in (401, 403, 404):
            return SystemicProviderError(f"LLM API error (HTTP {status_code}): {msg}")

        # Systemic: connection errors wrapped as InternalServerError etc.
        if _is_connection_error_message(msg):
            return SystemicProviderError(
                f"Cannot connect to API endpoint. "
                f"Check your --base-url or network. Details: {msg}"
            )

        # Systemic: missing / invalid provider
        if _is_missing_provider_error(msg):
            return SystemicProviderError(
                _format_provider_hint(
                    model=self.model,
                    api_base=self._api_base,
                    raw_message=msg,
                )
            )

        # Has a status_code → litellm API error (non-systemic)
        if status_code is not None:
            return ProviderError(f"LLM API error (HTTP {status_code}): {msg}")

        return ProviderError(f"Unexpected error: {msg}")

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

    async def __aenter__(self) -> ExternalLLMClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


__all__ = ["ExternalLLMClient"]

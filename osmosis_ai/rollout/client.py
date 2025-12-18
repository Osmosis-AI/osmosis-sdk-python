"""HTTP client for calling TrainGate's LLM completion endpoint.

This module provides the OsmosisLLMClient for making HTTP requests
to TrainGate's /v1/chat/completions endpoint with automatic retries,
connection pooling, and observability integration.

Example:
    async with OsmosisLLMClient(
        server_url="http://trainer:8080",
        rollout_id="rollout-123",
    ) as client:
        result = await client.chat_completions(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        print(result.message)

        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[...],
        )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from osmosis_ai.rollout.config.settings import RolloutClientSettings, get_settings
from osmosis_ai.rollout.core.exceptions import (
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
)
from osmosis_ai.rollout.core.schemas import (
    CompletionsRequest,
    RolloutMetrics,
    RolloutResponse,
    RolloutStatus,
)
from osmosis_ai.rollout.observability.logging import get_logger, set_rollout_id
from osmosis_ai.rollout.observability.metrics import get_metrics
from osmosis_ai.rollout.observability.tracing import SpanNames, span

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompletionsResult:
    """Result from LLM completion call.

    This is returned by OsmosisLLMClient.chat_completions() and contains
    the assistant's response along with token data for training.

    Attributes:
        message: Assistant message dict with optional tool_calls.
        token_ids: Token IDs from the response (for training).
        logprobs: Log probabilities (for training).
        usage: Token usage statistics.
        finish_reason: Why the completion ended ("stop", "tool_calls", etc.).

    Example:
        result = await client.chat_completions(messages)

        if result.has_tool_calls:
            for tool_call in result.tool_calls:
                # Execute tool
                pass
        else:
            print(result.content)
    """

    message: Dict[str, Any]
    token_ids: List[int]
    logprobs: List[float]
    usage: Dict[str, int]
    finish_reason: str

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return bool(self.message.get("tool_calls"))

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Get tool calls from the response."""
        return self.message.get("tool_calls", [])

    @property
    def content(self) -> Optional[str]:
        """Get text content from the response."""
        return self.message.get("content")


class OsmosisLLMClient:
    """HTTP client for calling TrainGate's /v1/chat/completions endpoint.

    Handles connection pooling, retries with exponential backoff,
    error categorization, and metrics collection.

    Features:
        - Automatic retries for 5xx errors and timeouts
        - Exponential backoff between retries
        - Connection pooling and keepalive
        - Metrics collection for monitoring
        - Tracing integration (when enabled)

    Example:
        async with OsmosisLLMClient(
            server_url="http://trainer:8080",
            rollout_id="rollout-123",
        ) as client:
            result = await client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
            )
            print(result.message)

            await client.complete_rollout(
                status="COMPLETED",
                final_messages=[...],
            )

    Attributes:
        server_url: TrainGate base URL.
        rollout_id: Unique rollout identifier.
        api_key: Optional Bearer token for authentication.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum retry attempts for 5xx errors.
        complete_rollout_retries: Maximum retries for completion callback.
    """

    def __init__(
        self,
        server_url: str,
        rollout_id: str,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        complete_rollout_retries: Optional[int] = None,
        settings: Optional[RolloutClientSettings] = None,
    ):
        """Initialize the LLM client.

        Args:
            server_url: TrainGate base URL (e.g., "http://trainer:8080").
            rollout_id: Unique rollout identifier for session routing.
            api_key: Optional Bearer token for authentication.
            timeout_seconds: Request timeout in seconds. Defaults to settings.
            max_retries: Maximum retry attempts for 5xx errors. Defaults to settings.
            complete_rollout_retries: Maximum retries for completion. Defaults to settings.
            settings: Client settings. Defaults to global settings.
        """
        # Load settings
        if settings is None:
            settings = get_settings().client

        self.server_url = server_url.rstrip("/")
        self.rollout_id = rollout_id
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds or settings.timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self.complete_rollout_retries = (
            complete_rollout_retries
            if complete_rollout_retries is not None
            else settings.complete_rollout_retries
        )

        # Settings for connection pool
        self._max_connections = settings.max_connections
        self._max_keepalive_connections = settings.max_keepalive_connections
        self._retry_base_delay = settings.retry_base_delay
        self._retry_max_delay = settings.retry_max_delay

        # HTTP client (lazy initialized)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # Metrics tracking
        self._llm_latency_ms: float = 0.0
        self._num_llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._response_tokens: int = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (thread-safe)."""
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                ),
                headers=headers,
            )
            return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff."""
        delay = self._retry_base_delay * (2 ** attempt)
        return min(delay, self._retry_max_delay)

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop: Any = None,
        logprobs: bool = True,
        **kwargs: Any,
    ) -> CompletionsResult:
        """Call TrainGate's /v1/chat/completions endpoint.

        Messages should be the FULL conversation history (append-only).
        Previous messages must not be modified.

        Args:
            messages: Full conversation message list.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            max_tokens: Maximum response tokens.
            stop: Optional stop sequences.
            logprobs: Whether to return log probabilities.

        Returns:
            CompletionsResult with LLM response and token data.

        Raises:
            OsmosisTransportError: Network/connection errors.
            OsmosisServerError: 5xx errors from TrainGate.
            OsmosisValidationError: 4xx errors (invalid request).
            OsmosisTimeoutError: Request timed out.
        """
        from osmosis_ai.rollout.utils import normalize_stop

        request = CompletionsRequest(
            rollout_id=self.rollout_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=normalize_stop(stop),
            logprobs=logprobs,
            **kwargs,
        )

        client = await self._get_client()
        url = f"{self.server_url}/v1/chat/completions"
        last_error: Optional[Exception] = None
        metrics = get_metrics()

        # Set context for logging
        set_rollout_id(self.rollout_id)

        for attempt in range(self.max_retries + 1):
            start_time = time.monotonic()

            with span(
                SpanNames.LLM_CHAT_COMPLETIONS,
                attributes={
                    "rollout_id": self.rollout_id,
                    "attempt": attempt + 1,
                    "message_count": len(messages),
                },
            ) as s:
                try:
                    # Use JSON mode to ensure enums/complex types are serialized
                    response = await client.post(
                        url, json=request.model_dump(mode="json", exclude_none=True)
                    )
                    elapsed_ms = (time.monotonic() - start_time) * 1000

                    if response.status_code == 200:
                        data = response.json()

                        # Update internal metrics
                        self._llm_latency_ms += elapsed_ms
                        self._num_llm_calls += 1
                        prompt_tokens = 0
                        completion_tokens = 0
                        if "usage" in data:
                            prompt_tokens = data["usage"].get("prompt_tokens", 0)
                            completion_tokens = data["usage"].get("completion_tokens", 0)
                            self._prompt_tokens += prompt_tokens
                            self._response_tokens += completion_tokens

                        # Update Prometheus metrics
                        metrics.llm_requests_total.labels(status="success").inc()
                        metrics.llm_request_duration_seconds.labels(status="success").observe(
                            elapsed_ms / 1000
                        )
                        metrics.llm_tokens_total.labels(type="prompt").inc(prompt_tokens)
                        metrics.llm_tokens_total.labels(type="completion").inc(completion_tokens)

                        # Set span attributes
                        s.set_attribute("status", "success")
                        s.set_attribute("prompt_tokens", prompt_tokens)
                        s.set_attribute("completion_tokens", completion_tokens)

                        # Extract response
                        choice = data["choices"][0]
                        return CompletionsResult(
                            message=choice["message"],
                            token_ids=data.get("token_ids", []),
                            logprobs=data.get("logprobs", []),
                            usage=data.get("usage", {}),
                            finish_reason=choice.get("finish_reason", "stop"),
                        )

                    if 400 <= response.status_code < 500:
                        detail = response.text
                        logger.error(
                            "traingate_validation_error",
                            status_code=response.status_code,
                            detail=detail[:200],
                        )
                        metrics.llm_requests_total.labels(status="error").inc()
                        metrics.llm_request_duration_seconds.labels(status="error").observe(
                            elapsed_ms / 1000
                        )
                        s.set_attribute("status", "validation_error")
                        s.set_attribute("status_code", response.status_code)
                        raise OsmosisValidationError(detail, response.status_code)

                    if response.status_code >= 500:
                        detail = response.text
                        logger.warning(
                            "traingate_server_error",
                            status_code=response.status_code,
                            attempt=attempt + 1,
                            max_attempts=self.max_retries + 1,
                            detail=detail[:200],
                        )
                        metrics.llm_requests_total.labels(status="error").inc()
                        metrics.llm_request_duration_seconds.labels(status="error").observe(
                            elapsed_ms / 1000
                        )
                        s.set_attribute("status", "server_error")
                        s.set_attribute("status_code", response.status_code)
                        last_error = OsmosisServerError(detail, response.status_code)
                    else:
                        detail = f"Unexpected status code {response.status_code}"
                        logger.error("traingate_unexpected_status", detail=detail)
                        raise OsmosisValidationError(detail, response.status_code)

                except httpx.TimeoutException as e:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    logger.warning(
                        "request_timeout",
                        attempt=attempt + 1,
                        max_attempts=self.max_retries + 1,
                        error=str(e),
                    )
                    s.set_attribute("status", "timeout")
                    metrics.llm_requests_total.labels(status="timeout").inc()
                    metrics.llm_request_duration_seconds.labels(status="timeout").observe(
                        elapsed_ms / 1000
                    )
                    last_error = OsmosisTimeoutError(str(e))

                except httpx.RequestError as e:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    error_detail = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
                    logger.warning(
                        "transport_error",
                        attempt=attempt + 1,
                        max_attempts=self.max_retries + 1,
                        error=error_detail,
                    )
                    s.set_attribute("status", "transport_error")
                    metrics.llm_requests_total.labels(status="error").inc()
                    metrics.llm_request_duration_seconds.labels(status="error").observe(
                        elapsed_ms / 1000
                    )
                    last_error = OsmosisTransportError(error_detail)

                except OsmosisValidationError:
                    raise

            if attempt < self.max_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.info(
                    "retrying_chat_completions",
                    delay_seconds=delay,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

        raise OsmosisTransportError("No attempts made")

    async def complete_rollout(
        self,
        status: str,
        final_messages: List[Dict[str, Any]],
        finish_reason: str = "stop",
        error_message: Optional[str] = None,
        metrics: Optional[RolloutMetrics] = None,
    ) -> None:
        """Notify TrainGate that rollout is complete.

        Must be called exactly once per rollout. Retries on transient errors.

        Args:
            status: "COMPLETED" or "ERROR".
            final_messages: Final conversation messages.
            finish_reason: Why the rollout ended.
            error_message: Error message if status="ERROR".
            metrics: Optional execution metrics.

        Raises:
            OsmosisTransportError: Network/connection errors after retries.
            OsmosisServerError: 5xx errors from TrainGate after retries.
            OsmosisValidationError: 4xx errors (not retried).
            OsmosisTimeoutError: Request timed out after retries.
        """
        # Build metrics if not provided
        if metrics is None:
            metrics = RolloutMetrics(
                llm_latency_ms=self._llm_latency_ms,
                num_llm_calls=self._num_llm_calls,
                prompt_tokens=self._prompt_tokens,
                response_tokens=self._response_tokens,
            )

        response_data = RolloutResponse(
            rollout_id=self.rollout_id,
            status=RolloutStatus(status),
            final_messages=final_messages,
            finish_reason=finish_reason,
            error_message=error_message,
            metrics=metrics,
        )

        client = await self._get_client()
        url = f"{self.server_url}/v1/rollout/completed"
        last_error: Optional[Exception] = None

        for attempt in range(self.complete_rollout_retries + 1):
            with span(
                SpanNames.LLM_COMPLETE_ROLLOUT,
                attributes={
                    "rollout_id": self.rollout_id,
                    "status": status,
                    "attempt": attempt + 1,
                },
            ) as s:
                try:
                    # Use JSON mode to ensure enums/complex types are serialized
                    response = await client.post(
                        url,
                        json=response_data.model_dump(mode="json", exclude_none=True),
                    )

                    if response.status_code == 200:
                        logger.info(
                            "rollout_completion_acknowledged",
                            rollout_id=self.rollout_id,
                            status=status,
                        )
                        s.set_attribute("ack", True)
                        return

                    if 400 <= response.status_code < 500:
                        detail = response.text
                        logger.error(
                            "completion_validation_error",
                            status_code=response.status_code,
                            detail=detail[:200],
                        )
                        s.set_attribute("status", "validation_error")
                        raise OsmosisValidationError(detail, response.status_code)

                    if response.status_code >= 500:
                        detail = response.text
                        logger.warning(
                            "completion_server_error",
                            status_code=response.status_code,
                            attempt=attempt + 1,
                            max_attempts=self.complete_rollout_retries + 1,
                        )
                        s.set_attribute("status", "server_error")
                        last_error = OsmosisServerError(detail, response.status_code)

                except httpx.TimeoutException as e:
                    logger.warning(
                        "completion_timeout",
                        attempt=attempt + 1,
                        max_attempts=self.complete_rollout_retries + 1,
                    )
                    s.set_attribute("status", "timeout")
                    last_error = OsmosisTimeoutError(str(e))

                except httpx.RequestError as e:
                    error_detail = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
                    logger.warning(
                        "completion_transport_error",
                        attempt=attempt + 1,
                        error=error_detail,
                    )
                    s.set_attribute("status", "transport_error")
                    last_error = OsmosisTransportError(error_detail)

                except OsmosisValidationError:
                    raise

            if attempt < self.complete_rollout_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.info("retrying_complete_rollout", delay_seconds=delay)
                await asyncio.sleep(delay)

        if last_error is not None:
            logger.error(
                "rollout_completion_failed",
                rollout_id=self.rollout_id,
                attempts=self.complete_rollout_retries + 1,
            )
            raise last_error

        raise OsmosisTransportError("No attempts made")

    def get_metrics(self) -> RolloutMetrics:
        """Get current metrics from this client session.

        Returns:
            RolloutMetrics with accumulated statistics.
        """
        return RolloutMetrics(
            llm_latency_ms=self._llm_latency_ms,
            num_llm_calls=self._num_llm_calls,
            prompt_tokens=self._prompt_tokens,
            response_tokens=self._response_tokens,
        )

    async def __aenter__(self) -> "OsmosisLLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = [
    "CompletionsResult",
    "OsmosisLLMClient",
]

import asyncio
import logging
from typing import Any

import httpx

_shared_client: httpx.AsyncClient | None = None


def get_shared_client(timeout_seconds: float = 10.0) -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=timeout_seconds)
    return _shared_client


async def close_shared_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


async def post_json_with_retry(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    timeout_seconds: float = 10.0,
) -> httpx.Response:
    """POST JSON with retry + exponential backoff for transient failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    client = get_shared_client(timeout_seconds)
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"Retryable status code: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exception = exc
            is_last_attempt = attempt >= max_attempts
            should_retry = _is_retryable_exception(exc)
            if is_last_attempt or not should_retry:
                raise

            delay = min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            logging.warning(
                "POST callback failed (attempt %s/%s): %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                str(exc),
                delay,
            )
            await asyncio.sleep(delay)

    # This should be unreachable, but keeps type-checkers satisfied.
    raise RuntimeError(
        "POST request failed without raising an exception"
    ) from last_exception


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in {429, 500, 502, 503, 504}

    return False

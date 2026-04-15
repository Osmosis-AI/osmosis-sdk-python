"""Tests for osmosis_ai.rollout.utils.http."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from osmosis_ai.rollout.utils.http import (
    _is_retryable_exception,
    close_shared_client,
    get_shared_client,
    post_json_with_retry,
)


class TestIsRetryableException:
    def test_request_error_is_retryable(self):
        exc = httpx.ConnectError("connection refused")
        assert _is_retryable_exception(exc) is True

    def test_429_is_retryable(self):
        request = httpx.Request("POST", "http://example.com")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("rate limited", request=request, response=response)
        assert _is_retryable_exception(exc) is True

    def test_500_is_retryable(self):
        request = httpx.Request("POST", "http://example.com")
        response = httpx.Response(500, request=request)
        exc = httpx.HTTPStatusError("server error", request=request, response=response)
        assert _is_retryable_exception(exc) is True

    def test_502_503_504_are_retryable(self):
        for code in [502, 503, 504]:
            request = httpx.Request("POST", "http://example.com")
            response = httpx.Response(code, request=request)
            exc = httpx.HTTPStatusError("err", request=request, response=response)
            assert _is_retryable_exception(exc) is True

    def test_400_is_not_retryable(self):
        request = httpx.Request("POST", "http://example.com")
        response = httpx.Response(400, request=request)
        exc = httpx.HTTPStatusError("bad request", request=request, response=response)
        assert _is_retryable_exception(exc) is False

    def test_generic_exception_is_not_retryable(self):
        assert _is_retryable_exception(RuntimeError("oops")) is False


class TestGetSharedClient:
    async def test_returns_client(self):
        client = get_shared_client()
        assert isinstance(client, httpx.AsyncClient)
        await close_shared_client()

    async def test_reuses_same_client(self):
        c1 = get_shared_client()
        c2 = get_shared_client()
        assert c1 is c2
        await close_shared_client()

    async def test_recreates_after_close(self):
        c1 = get_shared_client()
        await close_shared_client()
        c2 = get_shared_client()
        assert c1 is not c2
        await close_shared_client()


class TestPostJsonWithRetry:
    async def test_invalid_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            await post_json_with_retry(
                url="http://localhost", payload={}, max_attempts=0
            )

    async def test_non_retryable_error_raises_immediately(self):
        """A 400 error should raise immediately without retrying."""
        request = httpx.Request("POST", "http://test/callback")
        response_400 = httpx.Response(400, request=request)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response_400)

        with patch(
            "osmosis_ai.rollout.utils.http.get_shared_client",
            return_value=mock_client,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await post_json_with_retry(
                    url="http://test/callback",
                    payload={"key": "value"},
                    max_attempts=3,
                )
        # Only one attempt — 400 is not retryable.
        assert mock_client.post.await_count == 1

    async def test_success_on_first_try(self):
        request = httpx.Request("POST", "http://test/callback")
        response_200 = httpx.Response(200, request=request, json={"ok": True})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response_200)

        with patch(
            "osmosis_ai.rollout.utils.http.get_shared_client",
            return_value=mock_client,
        ):
            resp = await post_json_with_retry(
                url="http://test/callback",
                payload={"data": 1},
                max_attempts=3,
            )
        assert resp.status_code == 200
        assert mock_client.post.await_count == 1

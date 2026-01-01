# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.client."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from osmosis_ai.rollout import (
    OsmosisLLMClient,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
    RolloutMetrics,
)
from osmosis_ai.rollout.client import CompletionsResult


# =============================================================================
# OsmosisLLMClient Initialization Tests
# =============================================================================


def test_client_initialization() -> None:
    """Verify client initialization with required parameters."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )
    assert client.server_url == "http://localhost:8080"
    assert client.rollout_id == "test-123"
    assert client.api_key is None
    assert client.timeout_seconds == 300.0
    assert client.max_retries == 3


def test_client_initialization_with_all_params() -> None:
    """Verify client initialization with all parameters."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        api_key="test-key",
        timeout_seconds=60.0,
        max_retries=5,
        complete_rollout_retries=3,
    )
    assert client.api_key == "test-key"
    assert client.timeout_seconds == 60.0
    assert client.max_retries == 5
    assert client.complete_rollout_retries == 3


def test_client_default_complete_rollout_retries() -> None:
    """Verify default complete_rollout_retries is 2."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )
    assert client.complete_rollout_retries == 2


def test_client_url_trailing_slash_removed() -> None:
    """Verify trailing slash is removed from server_url."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080/",
        rollout_id="test-123",
    )
    assert client.server_url == "http://localhost:8080"


def test_client_url_multiple_trailing_slashes_removed() -> None:
    """Verify multiple trailing slashes are removed."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080///",
        rollout_id="test-123",
    )
    # rstrip("/") removes all trailing slashes
    assert client.server_url == "http://localhost:8080"


# =============================================================================
# CompletionsResult Tests
# =============================================================================


def test_completions_result_has_tool_calls_true() -> None:
    """Verify has_tool_calls is True when tool_calls present."""
    result = CompletionsResult(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "test"}}],
        },
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        finish_reason="tool_calls",
    )
    assert result.has_tool_calls is True


def test_completions_result_has_tool_calls_false_empty() -> None:
    """Verify has_tool_calls is False for empty tool_calls."""
    result = CompletionsResult(
        message={"role": "assistant", "content": "Hello", "tool_calls": []},
        token_ids=[1, 2, 3],
        logprobs=[-0.1],
        usage={},
        finish_reason="stop",
    )
    assert result.has_tool_calls is False


def test_completions_result_has_tool_calls_false_missing() -> None:
    """Verify has_tool_calls is False when tool_calls missing."""
    result = CompletionsResult(
        message={"role": "assistant", "content": "Hello"},
        token_ids=[1, 2, 3],
        logprobs=[-0.1],
        usage={},
        finish_reason="stop",
    )
    assert result.has_tool_calls is False


def test_completions_result_tool_calls_property() -> None:
    """Verify tool_calls property returns the list."""
    tool_calls = [{"id": "call_1", "function": {"name": "add"}}]
    result = CompletionsResult(
        message={"role": "assistant", "tool_calls": tool_calls},
        token_ids=[],
        logprobs=[],
        usage={},
        finish_reason="tool_calls",
    )
    assert result.tool_calls == tool_calls


def test_completions_result_tool_calls_empty_default() -> None:
    """Verify tool_calls returns empty list when missing."""
    result = CompletionsResult(
        message={"role": "assistant", "content": "Hi"},
        token_ids=[],
        logprobs=[],
        usage={},
        finish_reason="stop",
    )
    assert result.tool_calls == []


def test_completions_result_content_property() -> None:
    """Verify content property returns message content."""
    result = CompletionsResult(
        message={"role": "assistant", "content": "Hello world"},
        token_ids=[],
        logprobs=[],
        usage={},
        finish_reason="stop",
    )
    assert result.content == "Hello world"


def test_completions_result_content_none() -> None:
    """Verify content returns None when not present."""
    result = CompletionsResult(
        message={"role": "assistant", "tool_calls": []},
        token_ids=[],
        logprobs=[],
        usage={},
        finish_reason="stop",
    )
    assert result.content is None


def test_completions_result_is_frozen() -> None:
    """Verify CompletionsResult is immutable."""
    result = CompletionsResult(
        message={"role": "assistant", "content": "Hi"},
        token_ids=[1, 2],
        logprobs=[-0.1],
        usage={},
        finish_reason="stop",
    )
    # dataclass(frozen=True) should prevent attribute assignment
    with pytest.raises(AttributeError):
        result.finish_reason = "modified"


# =============================================================================
# OsmosisLLMClient.get_metrics Tests
# =============================================================================


def test_client_get_metrics_initial() -> None:
    """Verify initial metrics are zero."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )
    metrics = client.get_metrics()

    assert metrics.llm_latency_ms == 0.0
    assert metrics.num_llm_calls == 0
    assert metrics.prompt_tokens == 0
    assert metrics.response_tokens == 0


# =============================================================================
# OsmosisLLMClient.chat_completions Tests
# =============================================================================


@pytest.mark.asyncio
async def test_chat_completions_success() -> None:
    """Verify successful chat_completions call."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "resp-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "token_ids": [1, 2, 3],
        "logprobs": [-0.1, -0.2],
    }

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response

    client._client = mock_http_client

    result = await client.chat_completions(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
    )

    assert result.message["content"] == "Hello!"
    assert result.finish_reason == "stop"
    assert result.token_ids == [1, 2, 3]
    assert result.usage["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_chat_completions_ignores_unknown_kwargs() -> None:
    """Verify chat_completions accepts and ignores unknown kwargs."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "resp-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    result = await client.chat_completions(
        messages=[{"role": "user", "content": "Hi"}],
        seed=123,  # Unknown kwargs are ignored
    )
    assert result.message["content"] == "OK"

    payload = mock_http_client.post.call_args[1]["json"]
    assert "seed" not in payload


@pytest.mark.asyncio
async def test_chat_completions_normalizes_stop_string() -> None:
    """Verify stop can be provided as a string and is normalized to List[str]."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "resp-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    await client.chat_completions(
        messages=[{"role": "user", "content": "Hi"}],
        stop="END",
    )

    payload = mock_http_client.post.call_args[1]["json"]
    assert payload["stop"] == ["END"]


@pytest.mark.asyncio
async def test_chat_completions_updates_metrics() -> None:
    """Verify metrics are updated after successful call."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    await client.chat_completions(messages=[])

    metrics = client.get_metrics()
    assert metrics.num_llm_calls == 1
    assert metrics.prompt_tokens == 20
    assert metrics.response_tokens == 10
    assert metrics.llm_latency_ms > 0


@pytest.mark.asyncio
async def test_chat_completions_no_retry_on_4xx() -> None:
    """Verify 4xx errors are not retried."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        max_retries=3,
    )

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with pytest.raises(OsmosisValidationError) as exc_info:
        await client.chat_completions(messages=[])

    assert exc_info.value.status_code == 400
    # Should only be called once (no retries)
    assert mock_http_client.post.call_count == 1


@pytest.mark.asyncio
async def test_chat_completions_retry_on_5xx() -> None:
    """Verify 5xx errors trigger retries."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        max_retries=2,
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(OsmosisServerError) as exc_info:
            await client.chat_completions(messages=[])

    assert exc_info.value.status_code == 500
    # Should be called max_retries + 1 times
    assert mock_http_client.post.call_count == 3


@pytest.mark.asyncio
async def test_chat_completions_timeout_raises_error() -> None:
    """Verify timeout raises OsmosisTimeoutError."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        max_retries=0,
    )

    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = httpx.TimeoutException("Timeout")
    client._client = mock_http_client

    with pytest.raises(OsmosisTimeoutError):
        await client.chat_completions(messages=[])


@pytest.mark.asyncio
async def test_chat_completions_transport_error() -> None:
    """Verify network errors raise OsmosisTransportError."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        max_retries=0,
    )

    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")
    client._client = mock_http_client

    with pytest.raises(OsmosisTransportError):
        await client.chat_completions(messages=[])


# =============================================================================
# OsmosisLLMClient.complete_rollout Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complete_rollout_success() -> None:
    """Verify successful complete_rollout call."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    await client.complete_rollout(
        status="COMPLETED",
        final_messages=[{"role": "assistant", "content": "Done"}],
        finish_reason="stop",
    )

    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert "/v1/rollout/completed" in call_args[0][0]


@pytest.mark.asyncio
async def test_complete_rollout_with_error_status() -> None:
    """Verify complete_rollout with ERROR status."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    await client.complete_rollout(
        status="ERROR",
        final_messages=[],
        finish_reason="error",
        error_message="Something went wrong",
    )

    call_args = mock_http_client.post.call_args
    request_body = call_args[1]["json"]
    assert request_body["status"] == "ERROR"
    assert request_body["error_message"] == "Something went wrong"


@pytest.mark.asyncio
async def test_complete_rollout_payload_is_json_serializable() -> None:
    """Ensure complete_rollout payload can be JSON-serialized (enums, etc.)."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    await client.complete_rollout(
        status="COMPLETED",
        final_messages=[{"role": "assistant", "content": "Done"}],
    )

    payload = mock_http_client.post.call_args[1]["json"]
    # Should not raise (this would catch Enum / non-JSON types)
    json.dumps(payload)
    assert payload["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_complete_rollout_with_metrics() -> None:
    """Verify complete_rollout with provided metrics."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    metrics = RolloutMetrics(
        total_latency_ms=5000.0,
        num_llm_calls=10,
    )

    await client.complete_rollout(
        status="COMPLETED",
        final_messages=[],
        metrics=metrics,
    )

    call_args = mock_http_client.post.call_args
    request_body = call_args[1]["json"]
    assert request_body["metrics"]["total_latency_ms"] == 5000.0
    assert request_body["metrics"]["num_llm_calls"] == 10


@pytest.mark.asyncio
async def test_complete_rollout_server_error() -> None:
    """Verify complete_rollout raises on server error."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal error"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with pytest.raises(OsmosisServerError):
        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[],
        )


@pytest.mark.asyncio
async def test_complete_rollout_validation_error() -> None:
    """Verify complete_rollout raises on validation error."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.text = "Validation failed"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with pytest.raises(OsmosisValidationError) as exc_info:
        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[],
        )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_complete_rollout_retries_on_5xx() -> None:
    """Verify complete_rollout retries on 5xx errors."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        complete_rollout_retries=2,
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(OsmosisServerError) as exc_info:
            await client.complete_rollout(
                status="COMPLETED",
                final_messages=[],
            )

    assert exc_info.value.status_code == 500
    # Should be called complete_rollout_retries + 1 times
    assert mock_http_client.post.call_count == 3


@pytest.mark.asyncio
async def test_complete_rollout_retries_on_timeout() -> None:
    """Verify complete_rollout retries on timeout."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        complete_rollout_retries=1,
    )

    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = httpx.TimeoutException("Timeout")
    client._client = mock_http_client

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(OsmosisTimeoutError):
            await client.complete_rollout(
                status="COMPLETED",
                final_messages=[],
            )

    # Should be called complete_rollout_retries + 1 times
    assert mock_http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_complete_rollout_no_retry_on_4xx() -> None:
    """Verify complete_rollout does not retry on 4xx errors."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        complete_rollout_retries=3,
    )

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with pytest.raises(OsmosisValidationError):
        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[],
        )

    # Should only be called once (no retries for 4xx)
    assert mock_http_client.post.call_count == 1


@pytest.mark.asyncio
async def test_complete_rollout_succeeds_after_retry() -> None:
    """Verify complete_rollout succeeds after transient failure."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        complete_rollout_retries=2,
    )

    # First call fails, second succeeds
    mock_fail_response = MagicMock()
    mock_fail_response.status_code = 500
    mock_fail_response.text = "Temporary error"

    mock_success_response = MagicMock()
    mock_success_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = [mock_fail_response, mock_success_response]
    client._client = mock_http_client

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Should not raise
        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[],
        )

    assert mock_http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_complete_rollout_zero_retries() -> None:
    """Verify complete_rollout works with zero retries."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
        complete_rollout_retries=0,
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server error"

    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    with pytest.raises(OsmosisServerError):
        await client.complete_rollout(
            status="COMPLETED",
            final_messages=[],
        )

    # Should only be called once with zero retries
    assert mock_http_client.post.call_count == 1


# =============================================================================
# OsmosisLLMClient Context Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_client_async_context_manager() -> None:
    """Verify async context manager properly closes client."""
    mock_http_client = AsyncMock()
    # Patch httpx.AsyncClient to avoid creating a real SSL context in test env.
    with patch("osmosis_ai.rollout.client.httpx.AsyncClient", return_value=mock_http_client):
        async with OsmosisLLMClient(
            server_url="http://localhost:8080",
            rollout_id="test-123",
        ) as client:
            assert client is not None
            # Create the internal client to test cleanup
            await client._get_client()
            assert client._client is mock_http_client

        # After exit, client should be closed
        mock_http_client.aclose.assert_awaited_once()
        assert client._client is None


@pytest.mark.asyncio
async def test_client_close_without_init() -> None:
    """Verify close() works even if client never initialized."""
    client = OsmosisLLMClient(
        server_url="http://localhost:8080",
        rollout_id="test-123",
    )
    # Should not raise
    await client.close()
    assert client._client is None

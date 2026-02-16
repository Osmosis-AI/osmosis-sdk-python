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

"""Tests for osmosis_ai.rollout.testing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osmosis_ai.rollout.testing import (
    RolloutCompletionTracker,
    _should_use_tools,
    create_mock_trainer_app,
    fake_prompt_token_ids,
    fake_token_ids,
    patch_httpx_for_mock_trainer,
)

# Check if FastAPI is available for testing
try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

requires_fastapi = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


# =============================================================================
# fake_token_ids Tests
# =============================================================================


def test_fake_token_ids_empty() -> None:
    """Verify fake_token_ids handles empty string."""
    assert fake_token_ids("") == []


def test_fake_token_ids_basic() -> None:
    """Verify fake_token_ids generates sequential IDs."""
    result = fake_token_ids("hello")
    assert result == [0, 1, 2, 3, 4]


def test_fake_token_ids_length() -> None:
    """Verify fake_token_ids generates one ID per character."""
    text = "The quick brown fox"
    result = fake_token_ids(text)
    assert len(result) == len(text)


def test_fake_token_ids_deterministic() -> None:
    """Verify fake_token_ids returns the same result for the same input."""
    result1 = fake_token_ids("test string")
    result2 = fake_token_ids("test string")
    assert result1 == result2


def test_fake_token_ids_always_starts_at_zero() -> None:
    """Verify fake_token_ids always starts the sequence at 0."""
    result = fake_token_ids("abc")
    assert result[0] == 0
    assert result[-1] == 2


# =============================================================================
# fake_prompt_token_ids Tests
# =============================================================================


def test_fake_prompt_token_ids_empty() -> None:
    """Verify fake_prompt_token_ids handles empty messages."""
    result = fake_prompt_token_ids([])
    # Should still return at least 10 tokens (10 * max(1, 0) = 10)
    assert len(result) == 10


def test_fake_prompt_token_ids_single() -> None:
    """Verify fake_prompt_token_ids with single message."""
    result = fake_prompt_token_ids([{"role": "user", "content": "Hi"}])
    assert len(result) == 10  # 10 * 1


def test_fake_prompt_token_ids_multiple() -> None:
    """Verify fake_prompt_token_ids scales with message count."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    result = fake_prompt_token_ids(messages)
    assert len(result) == 30  # 10 * 3


def test_fake_prompt_token_ids_returns_sequential_ids() -> None:
    """Verify fake_prompt_token_ids returns sequential integers starting at 0."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = fake_prompt_token_ids(messages)
    assert result == list(range(20))


def test_fake_prompt_token_ids_scales_linearly() -> None:
    """Verify that each additional message adds exactly 10 tokens."""
    for n in range(1, 6):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(n)]
        result = fake_prompt_token_ids(messages)
        assert len(result) == 10 * n


# =============================================================================
# RolloutCompletionTracker Tests
# =============================================================================


def test_tracker_initial_state() -> None:
    """Verify tracker starts empty."""
    tracker = RolloutCompletionTracker()
    assert tracker.responses == []
    assert not tracker.event.is_set()


def test_tracker_record() -> None:
    """Verify tracker records responses."""
    tracker = RolloutCompletionTracker()
    tracker.record({"rollout_id": "test-123", "status": "COMPLETED"})

    assert len(tracker.responses) == 1
    assert tracker.responses[0]["rollout_id"] == "test-123"
    assert tracker.event.is_set()


def test_tracker_record_multiple() -> None:
    """Verify tracker records multiple responses."""
    tracker = RolloutCompletionTracker()
    tracker.record({"rollout_id": "test-1"})
    tracker.record({"rollout_id": "test-2"})

    assert len(tracker.responses) == 2


def test_tracker_clear() -> None:
    """Verify tracker clear resets state."""
    tracker = RolloutCompletionTracker()
    tracker.record({"data": "test"})
    tracker.clear()

    assert tracker.responses == []
    assert not tracker.event.is_set()


def test_tracker_wait_timeout() -> None:
    """Verify tracker wait times out when no response."""
    tracker = RolloutCompletionTracker()
    result = tracker.wait(timeout=0.01)
    assert result is False


def test_tracker_wait_success() -> None:
    """Verify tracker wait returns True when event is set."""
    tracker = RolloutCompletionTracker()
    tracker.event.set()
    result = tracker.wait(timeout=0.01)
    assert result is True


def test_tracker_clear_after_multiple_records() -> None:
    """Verify tracker clear removes all recorded responses and resets the event."""
    tracker = RolloutCompletionTracker()
    tracker.record({"rollout_id": "r1"})
    tracker.record({"rollout_id": "r2"})
    tracker.record({"rollout_id": "r3"})

    assert len(tracker.responses) == 3
    assert tracker.event.is_set()

    tracker.clear()

    assert tracker.responses == []
    assert not tracker.event.is_set()
    # After clearing, wait should timeout
    assert tracker.wait(timeout=0.01) is False


def test_tracker_record_after_clear() -> None:
    """Verify tracker can record new responses after being cleared."""
    tracker = RolloutCompletionTracker()
    tracker.record({"rollout_id": "old"})
    tracker.clear()
    tracker.record({"rollout_id": "new"})

    assert len(tracker.responses) == 1
    assert tracker.responses[0]["rollout_id"] == "new"
    assert tracker.event.is_set()


# =============================================================================
# _should_use_tools Tests
# =============================================================================


def test_should_use_tools_returns_true_for_calculate_keyword() -> None:
    """Verify _should_use_tools detects 'calculate' in user message."""
    msg = {"role": "user", "content": "Please calculate 5 + 3"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_add_keyword() -> None:
    """Verify _should_use_tools detects 'add' in user message."""
    msg = {"role": "user", "content": "Can you add these numbers?"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_sum_keyword() -> None:
    """Verify _should_use_tools detects 'sum' in user message."""
    msg = {"role": "user", "content": "What is the sum of 10 and 20?"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_plus_keyword() -> None:
    """Verify _should_use_tools detects 'plus' in user message."""
    msg = {"role": "user", "content": "5 plus 3 equals what?"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_subtract_keyword() -> None:
    """Verify _should_use_tools detects 'subtract' in user message."""
    msg = {"role": "user", "content": "subtract 3 from 10"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_multiply_keyword() -> None:
    """Verify _should_use_tools detects 'multiply' in user message."""
    msg = {"role": "user", "content": "multiply 4 by 5"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_true_for_divide_keyword() -> None:
    """Verify _should_use_tools detects 'divide' in user message."""
    msg = {"role": "user", "content": "divide 20 by 4"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_is_case_insensitive() -> None:
    """Verify _should_use_tools matches keywords regardless of case."""
    msg = {"role": "user", "content": "CALCULATE the total"}
    assert _should_use_tools(msg) is True


def test_should_use_tools_returns_false_for_non_calculator_message() -> None:
    """Verify _should_use_tools returns False for generic messages."""
    msg = {"role": "user", "content": "Hello, how are you?"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_assistant_role() -> None:
    """Verify _should_use_tools returns False when role is not 'user'."""
    msg = {"role": "assistant", "content": "I will calculate that for you"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_system_role() -> None:
    """Verify _should_use_tools returns False for system messages even with keywords."""
    msg = {"role": "system", "content": "You can add numbers"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_tool_role() -> None:
    """Verify _should_use_tools returns False for tool messages."""
    msg = {"role": "tool", "content": "The sum is 8"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_non_string_content() -> None:
    """Verify _should_use_tools returns False when content is not a string."""
    msg = {"role": "user", "content": 12345}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_none_content() -> None:
    """Verify _should_use_tools returns False when content is None."""
    msg = {"role": "user", "content": None}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_missing_content() -> None:
    """Verify _should_use_tools returns False when content key is missing."""
    msg = {"role": "user"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_missing_role() -> None:
    """Verify _should_use_tools returns False when role key is missing."""
    msg = {"content": "calculate 5 + 3"}
    assert _should_use_tools(msg) is False


def test_should_use_tools_returns_false_for_list_content() -> None:
    """Verify _should_use_tools returns False when content is a list (multimodal)."""
    msg = {"role": "user", "content": [{"type": "text", "text": "calculate 5+3"}]}
    assert _should_use_tools(msg) is False


# =============================================================================
# create_mock_trainer_app Tests
# =============================================================================


@pytest.fixture
def mock_trainer_client():
    """Create mock trainer app and test client."""
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed")
    app = create_mock_trainer_app()
    return TestClient(app)


@requires_fastapi
def test_mock_trainer_health(mock_trainer_client) -> None:
    """Verify mock trainer health endpoint."""
    response = mock_trainer_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mock-trainer"


@requires_fastapi
def test_mock_trainer_completions_basic(mock_trainer_client) -> None:
    """Verify mock trainer completions endpoint."""
    response = mock_trainer_client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-123",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-123"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


@requires_fastapi
def test_mock_trainer_completions_with_tool_keyword(mock_trainer_client) -> None:
    """Verify mock trainer generates tool calls for calculator keywords."""
    response = mock_trainer_client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-123",
            "messages": [{"role": "user", "content": "Please calculate 5 + 3"}],
        },
    )
    assert response.status_code == 200
    data = response.json()

    message = data["choices"][0]["message"]
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "add"


@requires_fastapi
def test_mock_trainer_completions_after_tool_result(mock_trainer_client) -> None:
    """Verify mock trainer responds correctly after tool result."""
    response = mock_trainer_client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-123",
            "messages": [
                {"role": "user", "content": "Calculate 5 + 3"},
                {
                    "role": "assistant",
                    "content": "I'll help",
                    "tool_calls": [{"id": "call_1", "function": {"name": "add"}}],
                },
                {"role": "tool", "content": "8", "tool_call_id": "call_1"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()

    message = data["choices"][0]["message"]
    assert "tool_calls" not in message or message.get("tool_calls") is None
    assert message["content"] == "The calculation is complete."


@requires_fastapi
def test_mock_trainer_completions_returns_tokens(mock_trainer_client) -> None:
    """Verify mock trainer returns token information."""
    response = mock_trainer_client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-123",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    data = response.json()

    assert "token_ids" in data
    assert "logprobs" in data
    assert "prompt_token_ids" in data
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@requires_fastapi
def test_mock_trainer_rollout_completed(mock_trainer_client) -> None:
    """Verify mock trainer rollout completed endpoint."""
    response = mock_trainer_client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "test-123",
            "status": "COMPLETED",
            "final_messages": [{"role": "assistant", "content": "Done"}],
            "finish_reason": "stop",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@requires_fastapi
def test_mock_trainer_get_completed_rollout(mock_trainer_client) -> None:
    """Verify mock trainer stores and returns completed rollouts."""
    # First complete a rollout
    mock_trainer_client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "test-456",
            "status": "COMPLETED",
            "final_messages": [{"role": "assistant", "content": "Done"}],
            "finish_reason": "stop",
        },
    )

    # Then retrieve it
    response = mock_trainer_client.get("/v1/rollout/completed/test-456")
    assert response.status_code == 200
    data = response.json()
    assert data["rollout_id"] == "test-456"
    assert data["status"] == "COMPLETED"


@requires_fastapi
def test_mock_trainer_get_missing_rollout(mock_trainer_client) -> None:
    """Verify mock trainer returns empty for missing rollout."""
    response = mock_trainer_client.get("/v1/rollout/completed/nonexistent")
    assert response.status_code == 200
    assert response.json() == {}


@requires_fastapi
def test_mock_trainer_with_tracker() -> None:
    """Verify mock trainer uses tracker when provided."""
    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)
    client = TestClient(app)

    client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "tracked-123",
            "status": "COMPLETED",
            "final_messages": [],
            "finish_reason": "stop",
        },
    )

    assert len(tracker.responses) == 1
    assert tracker.responses[0]["rollout_id"] == "tracked-123"
    assert tracker.event.is_set()


@requires_fastapi
def test_mock_trainer_custom_tool_generator() -> None:
    """Verify mock trainer uses custom tool call generator."""

    def custom_generator(message):
        if "weather" in message.get("content", "").lower():
            return [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ]
        return None

    app = create_mock_trainer_app(tool_call_generator=custom_generator)
    client = TestClient(app)

    # Should trigger custom tool call
    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
        },
    )
    data = response.json()
    message = data["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"

    # Should not trigger (no calculator keywords either)
    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test",
            "messages": [{"role": "user", "content": "Hello there"}],
        },
    )
    data = response.json()
    message = data["choices"][0]["message"]
    assert "tool_calls" not in message or message.get("tool_calls") is None


@requires_fastapi
def test_mock_trainer_completions_with_empty_messages() -> None:
    """Verify mock trainer handles empty messages list gracefully."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-empty",
            "messages": [],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # When messages list is empty, it uses default placeholder message
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "OK."


@requires_fastapi
def test_mock_trainer_completions_token_ids_match_response_length() -> None:
    """Verify token IDs length matches response content length."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-tokens",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert len(data["token_ids"]) == len(content)
    assert len(data["logprobs"]) == len(data["token_ids"])


@requires_fastapi
def test_mock_trainer_completions_usage_totals_are_consistent() -> None:
    """Verify that usage total_tokens equals prompt_tokens + completion_tokens."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-usage",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Tell me more"},
            ],
        },
    )
    data = response.json()
    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@requires_fastapi
def test_mock_trainer_completions_tool_call_response_has_correct_structure() -> None:
    """Verify tool call response includes content alongside tool_calls."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-tool-struct",
            "messages": [{"role": "user", "content": "Please add 5 and 3"}],
        },
    )
    data = response.json()
    message = data["choices"][0]["message"]

    # Tool call responses should have both content and tool_calls
    assert message["content"] == "I'll help you with that calculation."
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    assert tool_call["id"] == "call_mock_add"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "add"
    assert tool_call["function"]["arguments"] == '{"a": 5, "b": 3}'


@requires_fastapi
def test_mock_trainer_without_tracker_still_stores_rollouts() -> None:
    """Verify completed rollouts are stored even without a tracker."""
    app = create_mock_trainer_app(tracker=None)
    client = TestClient(app)

    client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "no-tracker-123",
            "status": "COMPLETED",
            "final_messages": [{"role": "assistant", "content": "Done"}],
            "finish_reason": "stop",
        },
    )

    response = client.get("/v1/rollout/completed/no-tracker-123")
    assert response.status_code == 200
    data = response.json()
    assert data["rollout_id"] == "no-tracker-123"


@requires_fastapi
def test_mock_trainer_completions_model_field_echoed() -> None:
    """Verify the model field from request is echoed in response."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "custom-model-v1",
        },
    )
    data = response.json()
    assert data["model"] == "custom-model-v1"


@requires_fastapi
def test_mock_trainer_completions_rollout_id_echoed_as_response_id() -> None:
    """Verify rollout_id is used as the response id."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "rollout_id": "unique-id-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    data = response.json()
    assert data["id"] == "unique-id-xyz"


@requires_fastapi
def test_mock_trainer_completed_rollout_overwrites_previous() -> None:
    """Verify completing the same rollout_id overwrites the previous entry."""
    app = create_mock_trainer_app()
    client = TestClient(app)

    # First completion
    client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "overwrite-test",
            "status": "COMPLETED",
            "final_messages": [{"role": "assistant", "content": "First"}],
            "finish_reason": "stop",
        },
    )

    # Second completion with same rollout_id
    client.post(
        "/v1/rollout/completed",
        json={
            "rollout_id": "overwrite-test",
            "status": "ERROR",
            "final_messages": [{"role": "assistant", "content": "Second"}],
            "finish_reason": "error",
            "error_message": "something failed",
        },
    )

    response = client.get("/v1/rollout/completed/overwrite-test")
    data = response.json()
    assert data["status"] == "ERROR"
    assert data["finish_reason"] == "error"


def test_create_mock_trainer_app_raises_import_error_when_fastapi_missing() -> None:
    """Verify create_mock_trainer_app raises ImportError when FastAPI is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="FastAPI is required"):
            create_mock_trainer_app()


# =============================================================================
# patch_httpx_for_mock_trainer Tests
# =============================================================================


@requires_fastapi
async def test_patch_httpx_routes_completions_to_mock_trainer(monkeypatch) -> None:
    """Verify patch_httpx_for_mock_trainer routes /v1/chat/completions to the mock."""
    import httpx

    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            "http://fake-server:8080/v1/chat/completions",
            json={
                "rollout_id": "patched-test",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "patched-test"
    assert data["choices"][0]["message"]["role"] == "assistant"


@requires_fastapi
async def test_patch_httpx_routes_rollout_completed_to_mock_trainer(
    monkeypatch,
) -> None:
    """Verify patch_httpx_for_mock_trainer routes /v1/rollout/completed to the mock."""
    import httpx

    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            "http://fake-server:8080/v1/rollout/completed",
            json={
                "rollout_id": "completed-test",
                "status": "COMPLETED",
                "final_messages": [],
                "finish_reason": "stop",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert len(tracker.responses) == 1
    assert tracker.responses[0]["rollout_id"] == "completed-test"


@requires_fastapi
async def test_patch_httpx_does_not_intercept_unrelated_urls(monkeypatch) -> None:
    """Verify patch_httpx_for_mock_trainer does not intercept non-rollout URLs."""
    import httpx

    app = create_mock_trainer_app()
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)

    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        async with httpx.AsyncClient() as http_client:
            await http_client.post(
                "http://127.0.0.1:1/v1/some/other/endpoint",
                json={"data": "test"},
                timeout=0.1,
            )


@requires_fastapi
async def test_patch_httpx_completions_response_is_valid_httpx_response(
    monkeypatch,
) -> None:
    """Verify the patched response is a proper httpx.Response object."""
    import httpx

    app = create_mock_trainer_app()
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            "http://any-host/v1/chat/completions",
            json={
                "rollout_id": "response-type-test",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert response.request.method == "POST"
    assert "v1/chat/completions" in str(response.request.url)

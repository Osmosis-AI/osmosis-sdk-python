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

import pytest

from osmosis_ai.rollout.testing import (
    RolloutCompletionTracker,
    create_mock_trainer_app,
    fake_prompt_token_ids,
    fake_token_ids,
)

# Check if FastAPI is available for testing
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

requires_fastapi = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="FastAPI not installed"
)


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

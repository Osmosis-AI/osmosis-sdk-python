# Testing Your Agent

This guide covers unit testing and local testing strategies for rollout agents. These techniques let you validate your agent logic without deploying to training infrastructure or connecting to external LLM providers.

## Using Mock Trainer

The SDK provides a mock trainer for local testing without a real TrainGate server.

```python
# test_with_mock_trainer.py

import pytest
from fastapi.testclient import TestClient

from osmosis_ai.rollout.testing import (
    create_mock_trainer_app,
    RolloutCompletionTracker,
    patch_httpx_for_mock_trainer,
)


@pytest.fixture
def mock_trainer(monkeypatch):
    """Set up mock trainer with completion tracking."""
    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)
    return client, tracker


def test_rollout_with_mock_trainer(mock_trainer):
    """Test complete rollout flow with mock trainer."""
    client, tracker = mock_trainer

    # Your rollout will use the mock trainer
    # ...

    # Wait for completion callback
    assert tracker.wait(timeout=5.0)
    assert len(tracker.responses) == 1
    assert tracker.responses[0]["status"] == "COMPLETED"
```

## Custom Tool Call Generator

You can customize when the mock trainer generates tool calls:

```python
def weather_tool_generator(message):
    """Generate weather tool calls for weather-related messages."""
    if "weather" in message.get("content", "").lower():
        return [
            {
                "id": "call_weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]
    return None

app = create_mock_trainer_app(tool_call_generator=weather_tool_generator)
```

## Example Test Using FastAPI TestClient

```python
# test_my_agent.py

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from osmosis_ai.rollout import create_app

from my_agent import MyAgentLoop


@pytest.fixture
def app():
    """Create test application."""
    return create_app(MyAgentLoop())


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_init_returns_tools(client):
    """Test /v1/rollout/init returns tools."""
    response = client.post(
        "/v1/rollout/init",
        json={
            "rollout_id": "test-123",
            "server_url": "http://localhost:8080",
            "messages": [{"role": "user", "content": "Hello"}],
            "completion_params": {"temperature": 0.7},
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "tools" in data


@pytest.mark.asyncio
async def test_agent_run():
    """Test agent run directly."""
    from unittest.mock import MagicMock
    from osmosis_ai.rollout import RolloutContext, RolloutRequest, RolloutMetrics

    agent = MyAgentLoop()

    # Create mock context
    mock_llm = MagicMock()
    mock_llm.get_metrics.return_value = RolloutMetrics()
    mock_llm.chat_completions = AsyncMock(return_value=MagicMock(
        message={"role": "assistant", "content": "Hello!"},
        has_tool_calls=False,
    ))

    request = RolloutRequest(
        rollout_id="test",
        server_url="http://localhost",
        messages=[{"role": "user", "content": "Hi"}],
        completion_params={},
    )

    ctx = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
    )

    result = await agent.run(ctx)

    assert result.status == "COMPLETED"
```

## Testing Utilities Reference

The `osmosis_ai.rollout.testing` module provides the following public utilities for testing agent loops without a real TrainGate server.

### `create_mock_trainer_app(tracker=None, tool_call_generator=None)`

Creates a mock trainer FastAPI application that implements the same HTTP endpoints as a real TrainGate server:

- `POST /v1/chat/completions` -- returns deterministic LLM responses with fake token IDs
- `POST /v1/rollout/completed` -- accepts completion callbacks
- `GET /v1/rollout/completed/{rollout_id}` -- queries completed rollouts
- `GET /health` -- health check

By default, the mock generates tool calls when it detects calculator-related keywords (e.g. "add", "calculate", "multiply") in the user message. Pass a custom `tool_call_generator` function to override this behavior.

### `RolloutCompletionTracker`

Thread-safe tracker that captures `/v1/rollout/completed` callbacks during tests.

| Attribute / Method | Description |
|--------------------|-------------|
| `event` | `threading.Event` that is set when a completion is received |
| `responses` | List of captured completion response dicts |
| `record(response)` | Record a response and signal the event |
| `clear()` | Clear recorded responses and reset the event |
| `wait(timeout=5.0)` | Block until a completion is received; returns `True` on success, `False` on timeout |

### `patch_httpx_for_mock_trainer(client, monkeypatch)`

Patches `httpx.AsyncClient.post` so that any requests to `/v1/chat/completions` or `/v1/rollout/completed` are routed to the mock trainer `TestClient` instead of making real HTTP calls. All other requests pass through unchanged.

```python
@pytest.fixture
def mock_trainer(monkeypatch):
    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)
    return client, tracker
```

### `fake_token_ids(text)`

Generates deterministic fake token IDs for testing. Returns a list of sequential integers, one per character in the input text (e.g. `fake_token_ids("hello")` returns `[0, 1, 2, 3, 4]`).

### `fake_prompt_token_ids(messages)`

Generates deterministic fake prompt token IDs for testing. The token count scales with the number of messages to simulate realistic prompt growth. Returns `list(range(10 * max(1, len(messages))))`.

> **Note:** These fake token ID functions produce deterministic output suitable for snapshot testing but do not correspond to any real tokenizer.

## See Also

- [Examples](./examples.md) -- agent implementations and utilities
- [Test Mode](../test-mode.md) -- testing with cloud LLMs

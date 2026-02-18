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

### `fake_token_ids(text)` / `fake_prompt_token_ids(messages)`

Generate deterministic fake token IDs for testing. These produce deterministic output suitable for snapshot testing but do not correspond to any real tokenizer.

For a complete test file example, see the [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) repository.

## See Also

- [Examples](./examples.md) -- agent implementations and utilities
- [Test Mode](../test-mode.md) -- testing with cloud LLMs

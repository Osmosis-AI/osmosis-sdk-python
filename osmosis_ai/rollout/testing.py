"""Testing utilities for Osmosis remote rollout SDK.

This module provides mock trainer implementations and test fixtures for
testing agent loops without requiring a real TrainGate server.

Example:
    from osmosis_ai.rollout.testing import (
        create_mock_trainer_app,
        RolloutCompletionTracker,
        patch_httpx_for_mock_trainer,
    )

    # Create mock trainer with completion tracking
    tracker = RolloutCompletionTracker()
    app = create_mock_trainer_app(tracker=tracker)

    # Use in tests with monkeypatch
    from fastapi.testclient import TestClient
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pytest
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from osmosis_ai.rollout.core.schemas import (
    CompletionUsage,
    CompletionsRequest,
    CompletionsResponse,
    RolloutResponse,
)


def fake_token_ids(text: str) -> List[int]:
    """Generate deterministic fake token IDs for testing.

    Args:
        text: The text to generate token IDs for.

    Returns:
        List of sequential integers (one per character).
    """
    return list(range(len(text)))


def fake_prompt_token_ids(messages: List[Dict[str, Any]]) -> List[int]:
    """Generate deterministic fake prompt token IDs for testing.

    Token count grows with message count to simulate real behavior.

    Args:
        messages: List of message dicts.

    Returns:
        List of sequential integers.
    """
    return list(range(10 * max(1, len(messages))))


@dataclass
class RolloutCompletionTracker:
    """Track rollout completion callbacks in tests.

    Thread-safe tracker for capturing /v1/rollout/completed callbacks.

    Attributes:
        event: Threading event that is set when a completion is received.
        responses: List of captured completion responses.

    Example:
        tracker = RolloutCompletionTracker()
        app = create_mock_trainer_app(tracker=tracker)

        # After rollout completes...
        tracker.event.wait(timeout=5.0)
        assert len(tracker.responses) == 1
        assert tracker.responses[0]["status"] == "COMPLETED"
    """

    event: threading.Event = field(default_factory=threading.Event)
    responses: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, response: Dict[str, Any]) -> None:
        """Record a completion response and signal the event.

        Args:
            response: The completion response dict.
        """
        self.responses.append(response)
        self.event.set()

    def clear(self) -> None:
        """Clear recorded responses and reset the event."""
        self.responses.clear()
        self.event.clear()

    def wait(self, timeout: float = 5.0) -> bool:
        """Wait for a completion callback.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if a completion was received, False if timeout.
        """
        return self.event.wait(timeout=timeout)


def _should_use_tools(last_message: Dict[str, Any]) -> bool:
    """Determine if the mock trainer should return tool calls.

    This heuristic detects calculator-related keywords to trigger tool use.

    Args:
        last_message: The last message in the conversation.

    Returns:
        True if tool calls should be included in response.
    """
    if last_message.get("role") != "user":
        return False
    content = last_message.get("content")
    if not isinstance(content, str):
        return False
    keywords = ["calculate", "add", "sum", "plus", "subtract", "multiply", "divide"]
    return any(k in content.lower() for k in keywords)


def create_mock_trainer_app(
    tracker: Optional[RolloutCompletionTracker] = None,
    tool_call_generator: Optional[Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]] = None,
) -> "FastAPI":
    """Create a mock trainer FastAPI application for testing.

    The mock trainer implements:
    - POST /v1/chat/completions: Returns deterministic LLM responses
    - POST /v1/rollout/completed: Accepts completion callbacks
    - GET /v1/rollout/completed/{rollout_id}: Query completed rollouts
    - GET /health: Health check

    By default, the mock generates tool calls when it detects calculator-related
    keywords in the user message. You can customize this behavior by providing
    a custom tool_call_generator function.

    Args:
        tracker: Optional tracker to record completion callbacks.
        tool_call_generator: Optional function to generate tool calls.
            Takes the last message and returns a list of tool calls or None.

    Returns:
        FastAPI application ready for testing.

    Example:
        from fastapi.testclient import TestClient

        app = create_mock_trainer_app()
        client = TestClient(app)

        response = client.post("/v1/chat/completions", json={
            "rollout_id": "test-123",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 200
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required for create_mock_trainer_app(). "
            "Install it with: pip install fastapi"
        )

    app = FastAPI(title="Mock Trainer Server")

    # In-memory storage of completed rollouts
    completed_rollouts: Dict[str, Dict[str, Any]] = {}

    @app.post("/v1/chat/completions")
    async def completions(request: CompletionsRequest) -> CompletionsResponse:
        messages = list(request.messages)
        last_message = messages[-1] if messages else {"role": "user", "content": ""}

        # Determine response based on message content
        tool_calls: Optional[List[Dict[str, Any]]] = None

        if tool_call_generator is not None:
            tool_calls = tool_call_generator(last_message)
        elif _should_use_tools(last_message):
            tool_calls = [
                {
                    "id": "call_mock_add",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'},
                }
            ]

        if tool_calls:
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": "I'll help you with that calculation.",
                "tool_calls": tool_calls,
            }
        elif last_message.get("role") == "tool":
            assistant_message = {
                "role": "assistant",
                "content": "The calculation is complete.",
            }
        else:
            assistant_message = {
                "role": "assistant",
                "content": "OK.",
            }

        response_text = assistant_message.get("content") or ""
        response_token_ids = fake_token_ids(str(response_text))
        prompt_token_ids = fake_prompt_token_ids(messages)

        return CompletionsResponse(
            id=request.rollout_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "stop",
                }
            ],
            usage=CompletionUsage(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(response_token_ids),
                total_tokens=len(prompt_token_ids) + len(response_token_ids),
            ),
            token_ids=response_token_ids,
            logprobs=[0.0] * len(response_token_ids),
            prompt_token_ids=prompt_token_ids,
        )

    @app.post("/v1/rollout/completed")
    async def rollout_completed(response: RolloutResponse) -> Dict[str, Any]:
        payload = response.model_dump(mode="json", exclude_none=True)
        completed_rollouts[response.rollout_id] = payload

        if tracker is not None:
            tracker.record(payload)

        return {"status": "ok"}

    @app.get("/v1/rollout/completed/{rollout_id}")
    async def get_completed_rollout(rollout_id: str) -> Dict[str, Any]:
        return completed_rollouts.get(rollout_id, {})

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "service": "mock-trainer"}

    return app


def patch_httpx_for_mock_trainer(
    client: "TestClient",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Patch httpx.AsyncClient to route requests to the mock trainer.

    This patches httpx.AsyncClient.post to redirect /v1/chat/completions
    and /v1/rollout/completed requests to the mock trainer TestClient.

    Args:
        client: FastAPI TestClient for the mock trainer app.
        monkeypatch: pytest monkeypatch fixture.

    Example:
        @pytest.fixture
        def mock_trainer(monkeypatch):
            tracker = RolloutCompletionTracker()
            app = create_mock_trainer_app(tracker=tracker)
            client = TestClient(app)
            patch_httpx_for_mock_trainer(client, monkeypatch)
            return client, tracker
    """
    import httpx

    original_post = httpx.AsyncClient.post

    async def mock_post(self, url: str, **kwargs):
        if "/v1/chat/completions" in url:
            resp = client.post("/v1/chat/completions", **kwargs)
            return httpx.Response(
                status_code=resp.status_code,
                json=resp.json(),
                request=httpx.Request("POST", url),
            )
        if "/v1/rollout/completed" in url:
            resp = client.post("/v1/rollout/completed", **kwargs)
            return httpx.Response(
                status_code=resp.status_code,
                json=resp.json(),
                request=httpx.Request("POST", url),
            )
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)


__all__ = [
    "create_mock_trainer_app",
    "RolloutCompletionTracker",
    "patch_httpx_for_mock_trainer",
    "fake_token_ids",
    "fake_prompt_token_ids",
]

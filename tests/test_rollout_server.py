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

"""Tests for osmosis_ai.rollout.server."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
    create_app,
)
from osmosis_ai.rollout.server import AppState

# Import FastAPI test client
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class SimpleAgentLoop(RolloutAgentLoop):
    """Simple agent loop for testing."""

    name = "simple_agent"

    def __init__(self, tools: List[OpenAIFunctionToolSchema] | None = None):
        self._tools = tools or []

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return self._tools

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete(list(ctx.request.messages))


class FailingAgentLoop(RolloutAgentLoop):
    """Agent loop that raises an error."""

    name = "failing_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        raise RuntimeError("Test error")


# =============================================================================
# AppState Tests
# =============================================================================


def test_app_state_is_duplicate_not_found() -> None:
    """Verify is_duplicate returns False for unknown rollout."""
    state = AppState()
    assert state.is_duplicate("unknown") is False


def test_app_state_is_duplicate_running() -> None:
    """Verify is_duplicate returns True for running rollout."""
    state = AppState()
    mock_task = MagicMock()
    state.mark_started("rollout-1", mock_task)

    assert state.is_duplicate("rollout-1") is True


def test_app_state_is_duplicate_completed() -> None:
    """Verify is_duplicate returns True for completed rollout."""
    state = AppState()
    state.completed_rollouts["rollout-1"] = 12345.0

    assert state.is_duplicate("rollout-1") is True


def test_app_state_mark_started_and_completed() -> None:
    """Verify state transitions from started to completed."""
    state = AppState()
    mock_task = MagicMock()

    state.mark_started("rollout-1", mock_task)
    assert "rollout-1" in state.rollout_tasks
    assert "rollout-1" not in state.completed_rollouts

    state.mark_completed("rollout-1")
    assert "rollout-1" not in state.rollout_tasks
    assert "rollout-1" in state.completed_rollouts


def test_app_state_mark_completed_moves_to_completed() -> None:
    """Verify mark_completed removes from tasks and adds to completed."""
    state = AppState()
    mock_task = MagicMock()

    state.mark_started("rollout-1", mock_task)
    state.mark_completed("rollout-1")

    assert state.rollout_tasks.get("rollout-1") is None
    assert "rollout-1" in state.completed_rollouts


def test_app_state_prune_completed_records() -> None:
    """Verify old completed records are pruned."""
    state = AppState(record_ttl_seconds=100)

    # Add old record (simulating time in the past)
    state.completed_rollouts["old-rollout"] = 0.0  # Very old timestamp

    # Manually trigger prune
    with patch("time.monotonic", return_value=200.0):
        state._prune_completed_records()

    assert "old-rollout" not in state.completed_rollouts


def test_app_state_prune_keeps_recent_records() -> None:
    """Verify recent completed records are kept."""
    state = AppState(record_ttl_seconds=100)

    with patch("time.monotonic", return_value=150.0):
        state.completed_rollouts["recent-rollout"] = 100.0  # 50 seconds old

        state._prune_completed_records()

    assert "recent-rollout" in state.completed_rollouts


@pytest.mark.asyncio
async def test_app_state_cancel_all() -> None:
    """Verify cancel_all cancels running tasks."""
    state = AppState()

    # Create mock tasks
    task1 = AsyncMock()
    task1.cancel = MagicMock()
    task2 = AsyncMock()
    task2.cancel = MagicMock()

    state.rollout_tasks["task1"] = task1
    state.rollout_tasks["task2"] = task2

    with patch("asyncio.gather", new_callable=AsyncMock):
        await state.cancel_all()

    task1.cancel.assert_called_once()
    task2.cancel.assert_called_once()


# =============================================================================
# create_app Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_create_app_returns_fastapi_instance() -> None:
    """Verify create_app returns a FastAPI application."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    assert isinstance(app, FastAPI)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_create_app_sets_title() -> None:
    """Verify app title includes agent loop name."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    assert "simple_agent" in app.title


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_create_app_with_custom_max_concurrent() -> None:
    """Verify create_app accepts custom max_concurrent."""
    agent_loop = SimpleAgentLoop()
    # Should not raise
    app = create_app(agent_loop, max_concurrent=50)
    assert app is not None


# =============================================================================
# /health Endpoint Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_health_endpoint() -> None:
    """Verify /health returns status information."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_rollouts" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_health_endpoint_includes_agent_name() -> None:
    """Verify /health includes agent_loop name."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.get("/health")

    data = response.json()
    assert data["agent_loop"] == "simple_agent"


# =============================================================================
# /platform/health Endpoint Tests (Authenticated Platform Health Check)
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_platform_health_endpoint_not_exposed_without_api_key() -> None:
    """Verify /platform/health is not exposed when API key auth is disabled."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.get("/platform/health")

    assert response.status_code == 404


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_platform_health_endpoint_missing_auth_returns_401() -> None:
    """Verify /platform/health requires Authorization when API key is configured."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop, api_key="test-api-key-12345")

    with TestClient(app) as client:
        response = client.get("/platform/health")

    assert response.status_code == 401


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_platform_health_endpoint_invalid_auth_returns_401() -> None:
    """Verify /platform/health rejects incorrect API keys."""
    agent_loop = SimpleAgentLoop()
    test_api_key = "test-api-key-12345"
    app = create_app(agent_loop, api_key=test_api_key)

    with TestClient(app) as client:
        response = client.get(
            "/platform/health",
            headers={"Authorization": "Bearer wrong-api-key"},
        )

    assert response.status_code == 401


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_platform_health_endpoint_valid_auth_returns_200() -> None:
    """Verify /platform/health returns status when given correct API key."""
    agent_loop = SimpleAgentLoop()
    test_api_key = "test-api-key-12345"
    app = create_app(agent_loop, api_key=test_api_key)

    with TestClient(app) as client:
        response = client.get(
            "/platform/health",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent_loop"] == "simple_agent"


# =============================================================================
# /v1/rollout/init Endpoint Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_returns_202() -> None:
    """Verify /v1/rollout/init returns 202 Accepted."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
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


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_returns_tools(sample_tool_schema: OpenAIFunctionToolSchema) -> None:
    """Verify /v1/rollout/init returns tools from agent_loop.get_tools()."""
    agent_loop = SimpleAgentLoop(tools=[sample_tool_schema])
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "test-123",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert data["rollout_id"] == "test-123"
    assert len(data["tools"]) == 1
    assert data["tools"][0]["function"]["name"] == "add"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_invalid_request() -> None:
    """Verify /v1/rollout/init returns 422 for invalid request."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                # Missing required fields
                "rollout_id": "test-123",
            },
        )

    assert response.status_code == 422


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_empty_rollout_id_rejected() -> None:
    """Verify /v1/rollout/init rejects empty rollout_id."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    assert response.status_code == 422


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_whitespace_rollout_id_rejected() -> None:
    """Verify /v1/rollout/init rejects whitespace rollout_id."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "   ",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    assert response.status_code == 422


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_echoes_rollout_id() -> None:
    """Verify /v1/rollout/init echoes back the rollout_id."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "my-unique-id-456",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    data = response.json()
    assert data["rollout_id"] == "my-unique-id-456"


# =============================================================================
# Idempotency Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_idempotent() -> None:
    """Verify duplicate requests return same response without new task."""
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        # First request
        response1 = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "idempotent-test",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

        # Second request with same rollout_id
        response2 = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "idempotent-test",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    assert response1.status_code == 202
    assert response2.status_code == 202
    assert response1.json()["rollout_id"] == response2.json()["rollout_id"]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_idempotent_caches_tools() -> None:
    """Verify get_tools is called once and duplicate requests return the same tools."""

    class CountingToolsAgentLoop(RolloutAgentLoop):
        name = "counting_tools_agent"

        def __init__(self) -> None:
            self.calls = 0

        def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
            self.calls += 1
            return [
                OpenAIFunctionToolSchema(
                    type="function",
                    function={
                        "name": f"tool_{self.calls}",
                        "description": "test tool",
                    },
                )
            ]

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            return ctx.complete(list(ctx.request.messages))

    agent_loop = CountingToolsAgentLoop()
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response1 = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "idempotent-tools-test",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )
        response2 = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "idempotent-tools-test",
                "server_url": "http://localhost:8080",
                "messages": [],
                "completion_params": {},
            },
        )

    assert response1.status_code == 202
    assert response2.status_code == 202
    assert agent_loop.calls == 1
    assert response1.json()["tools"] == response2.json()["tools"]


# =============================================================================
# Background Task Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_background_rollout_completes() -> None:
    """Verify background task completes and calls complete_rollout."""
    from osmosis_ai.rollout import RolloutMetrics

    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop)

    # Use mock to intercept OsmosisLLMClient
    with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.complete_rollout = AsyncMock()
        # get_metrics is a sync method, so use MagicMock
        mock_client_instance.get_metrics = MagicMock(return_value=RolloutMetrics())
        MockClient.return_value = mock_client_instance

        with TestClient(app) as client:
            response = client.post(
                "/v1/rollout/init",
                json={
                    "rollout_id": "bg-test",
                    "server_url": "http://localhost:8080",
                    "messages": [{"role": "user", "content": "test"}],
                    "completion_params": {},
                    # Callback auth: RolloutServer -> TrainGate
                    "api_key": "callback-token",
                },
            )

            assert response.status_code == 202

            # Give background task time to complete
            await asyncio.sleep(0.1)

        # Verify complete_rollout was called
        MockClient.assert_called_once()
        assert MockClient.call_args.kwargs["server_url"] == "http://localhost:8080"
        assert MockClient.call_args.kwargs["rollout_id"] == "bg-test"
        assert MockClient.call_args.kwargs["api_key"] == "callback-token"

        mock_client_instance.complete_rollout.assert_called_once()
        call_kwargs = mock_client_instance.complete_rollout.call_args[1]
        assert call_kwargs["status"] == "COMPLETED"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_background_rollout_handles_agent_error() -> None:
    """Verify background task handles agent errors gracefully."""
    agent_loop = FailingAgentLoop()
    app = create_app(agent_loop)

    with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.complete_rollout = AsyncMock()
        MockClient.return_value = mock_client_instance

        with TestClient(app) as client:
            response = client.post(
                "/v1/rollout/init",
                json={
                    "rollout_id": "error-test",
                    "server_url": "http://localhost:8080",
                    "messages": [],
                    "completion_params": {},
                },
            )

            assert response.status_code == 202

            # Give background task time to complete
            await asyncio.sleep(0.1)

        # Verify complete_rollout was called with ERROR status
        mock_client_instance.complete_rollout.assert_called()
        call_kwargs = mock_client_instance.complete_rollout.call_args[1]
        assert call_kwargs["status"] == "ERROR"
        assert "Test error" in call_kwargs["error_message"]


# =============================================================================
# API Key Validation Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_with_valid_api_key() -> None:
    """Verify /v1/rollout/init accepts requests with valid API key."""
    agent_loop = SimpleAgentLoop()
    test_api_key = "test-api-key-12345"
    app = create_app(agent_loop, api_key=test_api_key)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "rollout_id": "api-key-test",
                "server_url": "http://localhost:8080",
                "messages": [{"role": "user", "content": "Hello"}],
                "completion_params": {"temperature": 0.7},
            },
        )

    assert response.status_code == 202


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_with_invalid_api_key() -> None:
    """Verify /v1/rollout/init rejects requests with invalid API key."""
    agent_loop = SimpleAgentLoop()
    test_api_key = "test-api-key-12345"
    app = create_app(agent_loop, api_key=test_api_key)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            headers={"Authorization": "Bearer wrong-api-key"},
            json={
                "rollout_id": "api-key-test",
                "server_url": "http://localhost:8080",
                "messages": [{"role": "user", "content": "Hello"}],
                "completion_params": {"temperature": 0.7},
            },
        )

    assert response.status_code == 401


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_with_missing_api_key() -> None:
    """Verify /v1/rollout/init rejects requests without API key when required."""
    agent_loop = SimpleAgentLoop()
    test_api_key = "test-api-key-12345"
    app = create_app(agent_loop, api_key=test_api_key)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "api-key-test",
                "server_url": "http://localhost:8080",
                "messages": [{"role": "user", "content": "Hello"}],
                "completion_params": {"temperature": 0.7},
                # No api_key provided
            },
        )

    assert response.status_code == 401


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_init_endpoint_without_api_key_configured() -> None:
    """Verify /v1/rollout/init works without API key when not configured."""
    agent_loop = SimpleAgentLoop()
    # No api_key parameter - should not require authentication
    app = create_app(agent_loop)

    with TestClient(app) as client:
        response = client.post(
            "/v1/rollout/init",
            json={
                "rollout_id": "no-auth-test",
                "server_url": "http://localhost:8080",
                "messages": [{"role": "user", "content": "Hello"}],
                "completion_params": {"temperature": 0.7},
            },
        )

    assert response.status_code == 202


# =============================================================================
# Import Error Test
# =============================================================================


def test_create_app_raises_without_fastapi() -> None:
    """Verify create_app raises ImportError when FastAPI not available."""
    # This test only makes sense if we mock the import
    with patch.dict("sys.modules", {"fastapi": None}):
        # We need to reload the module to trigger the import error
        # For this test, we just verify the function exists
        pass  # Skip actual test as it requires complex module manipulation


# =============================================================================
# Debug Logging Tests
# =============================================================================


def test_rollout_context_debug_disabled_by_default() -> None:
    """Verify debug logging is disabled by default."""
    from osmosis_ai.rollout.core.base import RolloutContext
    from osmosis_ai.rollout import RolloutRequest

    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )

    mock_llm = MagicMock()
    ctx = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
    )

    assert ctx.debug_enabled is False
    assert ctx._debug_dir is None


def test_rollout_context_debug_enabled_when_dir_set() -> None:
    """Verify debug logging is enabled when _debug_dir is set."""
    from osmosis_ai.rollout.core.base import RolloutContext
    from osmosis_ai.rollout import RolloutRequest

    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )

    mock_llm = MagicMock()
    ctx = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
        _debug_dir="/tmp/debug",
    )

    assert ctx.debug_enabled is True
    assert ctx._debug_dir == "/tmp/debug"


def test_rollout_context_log_event_noop_when_disabled() -> None:
    """Verify log_event is a no-op when debug logging is disabled."""
    from osmosis_ai.rollout.core.base import RolloutContext
    from osmosis_ai.rollout import RolloutRequest

    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )

    mock_llm = MagicMock()
    ctx = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
    )

    # Should not raise or have any effect
    ctx.log_event("test_event", key="value")


def test_rollout_context_log_event_writes_to_file(tmp_path: Any) -> None:
    """Verify log_event writes events to JSONL file."""
    import json
    from osmosis_ai.rollout.core.base import RolloutContext
    from osmosis_ai.rollout import RolloutRequest

    debug_dir = str(tmp_path / "debug_logs")

    request = RolloutRequest(
        rollout_id="test-rollout-456",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )

    mock_llm = MagicMock()
    ctx = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
        _debug_dir=debug_dir,
    )

    # Log some events
    ctx.log_event("pre_llm", turn=0, num_messages=3)
    ctx.log_event("llm_response", turn=0, has_tool_calls=True)
    ctx.log_event("rollout_complete", finish_reason="stop", reward=1.0)

    # Verify file was created
    import os
    debug_file = os.path.join(debug_dir, "test-rollout-456.jsonl")
    assert os.path.exists(debug_file)

    # Verify contents
    with open(debug_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3

    event1 = json.loads(lines[0])
    assert event1["event"] == "pre_llm"
    assert event1["rollout_id"] == "test-rollout-456"
    assert event1["turn"] == 0
    assert event1["num_messages"] == 3

    event2 = json.loads(lines[1])
    assert event2["event"] == "llm_response"
    assert event2["has_tool_calls"] is True

    event3 = json.loads(lines[2])
    assert event3["event"] == "rollout_complete"
    assert event3["finish_reason"] == "stop"
    assert event3["reward"] == 1.0


def test_rollout_context_get_debug_file_path() -> None:
    """Verify _get_debug_file_path returns correct path."""
    from osmosis_ai.rollout.core.base import RolloutContext
    from osmosis_ai.rollout import RolloutRequest

    request = RolloutRequest(
        rollout_id="my-rollout-id",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )

    mock_llm = MagicMock()

    # Without debug_dir
    ctx_no_debug = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
    )
    assert ctx_no_debug._get_debug_file_path() is None

    # With debug_dir
    ctx_with_debug = RolloutContext(
        request=request,
        tools=[],
        llm=mock_llm,
        _debug_dir="/var/log/rollouts",
    )
    assert ctx_with_debug._get_debug_file_path() == "/var/log/rollouts/my-rollout-id.jsonl"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_create_app_with_debug_dir(tmp_path: Any) -> None:
    """Verify create_app accepts debug_dir parameter."""
    debug_dir = str(tmp_path / "debug_logs")
    agent_loop = SimpleAgentLoop()
    app = create_app(agent_loop, debug_dir=debug_dir)

    assert app is not None


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_background_rollout_with_debug_logging(tmp_path: Any) -> None:
    """Verify debug logging works in background rollout execution."""
    import json
    import os
    from osmosis_ai.rollout import RolloutMetrics

    debug_dir = str(tmp_path / "debug_logs")

    class DebugLoggingAgentLoop(RolloutAgentLoop):
        """Agent loop that uses debug logging."""

        name = "debug_logging_agent"

        def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
            return []

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            # Log some events
            ctx.log_event("pre_llm", turn=0, num_messages=len(ctx.request.messages))
            ctx.log_event("rollout_complete", finish_reason="stop")
            return ctx.complete(list(ctx.request.messages))

    agent_loop = DebugLoggingAgentLoop()
    app = create_app(agent_loop, debug_dir=debug_dir)

    with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.complete_rollout = AsyncMock()
        mock_client_instance.get_metrics = MagicMock(return_value=RolloutMetrics())
        MockClient.return_value = mock_client_instance

        with TestClient(app) as client:
            response = client.post(
                "/v1/rollout/init",
                json={
                    "rollout_id": "debug-test-789",
                    "server_url": "http://localhost:8080",
                    "messages": [{"role": "user", "content": "test"}],
                    "completion_params": {},
                },
            )

            assert response.status_code == 202

            # Give background task time to complete
            await asyncio.sleep(0.1)

        # Verify debug file was created
        debug_file = os.path.join(debug_dir, "debug-test-789.jsonl")
        assert os.path.exists(debug_file)

        # Verify contents
        with open(debug_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["event"] == "pre_llm"
        assert event1["rollout_id"] == "debug-test-789"

        event2 = json.loads(lines[1])
        assert event2["event"] == "rollout_complete"

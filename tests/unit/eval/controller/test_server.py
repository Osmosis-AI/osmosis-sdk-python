import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from osmosis_ai.eval.controller.server import EvalControllerServer, _heartbeat_interval
from osmosis_ai.eval.controller.state import ControllerRolloutState
from osmosis_ai.rollout.types import GraderStatus, RolloutSample, RolloutStatus


class FakeBridge:
    def __init__(self, response: Any | None = None) -> None:
        self.response = response or SimpleNamespace(
            id="chatcmpl-1",
            created=123,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content="hello", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.systemic_errors: dict[str, str] = {}
        self.exc: Exception | None = None

    async def complete(
        self, rollout_id: str, sample_id: str, body: dict[str, Any]
    ) -> Any:
        self.calls.append((rollout_id, sample_id, body))
        if self.exc is not None:
            raise self.exc
        return self.response

    def collect_systemic_error(self, rollout_id: str) -> str | None:
        return self.systemic_errors.pop(rollout_id, None)


@pytest.fixture
async def client() -> AsyncIterator[
    tuple[httpx.AsyncClient, EvalControllerServer, FakeBridge]
]:
    bridge = FakeBridge()
    server = EvalControllerServer(api_key="secret", bridge=bridge)
    state = ControllerRolloutState("r1")
    server.register_rollout_state(state)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as async_client:
        yield async_client, server, bridge


def auth_headers(rollout_id: str = "r1", sample_id: str = "s1") -> dict[str, str]:
    return {
        "Authorization": "Bearer secret",
        "x-rollout-id": rollout_id,
        "x-sample-id": sample_id,
    }


def sse_data_events(text: str) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_health_returns_ok(client) -> None:
    async_client, _, _ = client

    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_requires_bearer_auth_after_json_parse(client) -> None:
    async_client, _, _ = client

    response = await async_client.post(
        "/chat/completions",
        headers={"x-rollout-id": "r1", "x-sample-id": "s1"},
        json={"model": "openai/osmosis-rollout", "messages": []},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_reports_missing_headers_after_json_parse(client) -> None:
    async_client, _, _ = client

    response = await async_client.post(
        "/chat/completions",
        headers={"Authorization": "Bearer secret"},
        json={"model": "openai/osmosis-rollout", "messages": []},
    )

    assert response.status_code == 400
    assert "x-rollout-id" in response.text
    assert "x-sample-id" in response.text


@pytest.mark.asyncio
async def test_missing_stream_defaults_to_sse_one_payload_and_marks_completion(
    client,
) -> None:
    async_client, server, bridge = client

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "tools": [{"name": "lookup"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.startswith(": ping\n\n")
    events = sse_data_events(response.text)
    assert len(events) == 2
    assert events[-1] == "[DONE]"
    assert '"object":"chat.completion.chunk"' in events[0]
    assert bridge.calls[0][2]["messages"] == [{"role": "user", "content": "hello"}]
    state = server.get_rollout_state("r1")
    assert state is not None
    assert state.completed_sample_ids == {"s1"}
    assert state.completion_counts == {"s1": 1}
    assert state.total_tokens == 5


@pytest.mark.asyncio
async def test_stream_false_returns_json_payload(client) -> None:
    async_client, _, _ = client

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert response.json()["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_missing_model_uses_osmosis_rollout_response_model(client) -> None:
    async_client, _, _ = client

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={"messages": [], "stream": False},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "osmosis-rollout"


@pytest.mark.asyncio
async def test_invalid_multi_turn_mode_returns_400_with_allowed_values(client) -> None:
    async_client, _, _ = client

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": [],
            "multi_turn_mode": "other",
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert "multi_sample" in response.text
    assert "single_sample" in response.text


@pytest.mark.asyncio
async def test_invalid_message_item_returns_400_before_streaming(client) -> None:
    async_client, server, bridge = client
    state = server.get_rollout_state("r1")
    assert state is not None

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": ["bad"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "messages must be a list of objects"
    assert bridge.calls == []
    assert state.systemic_error is None


@pytest.mark.asyncio
async def test_single_sample_reuses_completed_branch_tools(client) -> None:
    async_client, server, bridge = client

    await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": [{"role": "user", "content": "first"}],
            "tools": [{"name": "first"}],
            "multi_turn_mode": "single_sample",
            "stream": False,
        },
    )
    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={
            "model": "openai/osmosis-rollout",
            "messages": [{"role": "user", "content": "second"}],
            "tools": [{"name": "second"}],
            "multi_turn_mode": "single_sample",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert bridge.calls[-1][2]["tools"] == [{"name": "first"}]
    assert server.get_rollout_state("r1").completion_counts == {"s1": 2}


@pytest.mark.asyncio
async def test_callback_schema_validation_and_stale_rollout_are_ok(client) -> None:
    async_client, _, _ = client

    invalid = await async_client.post(
        "/v1/rollout/completed",
        headers={"Authorization": "Bearer secret"},
        json={"rollout_id": "unknown"},
    )
    stale = await async_client.post(
        "/v1/rollout/completed",
        headers={"Authorization": "Bearer secret"},
        json={"rollout_id": "unknown", "status": "success"},
    )

    assert invalid.status_code == 422
    assert stale.status_code == 200
    assert stale.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_known_callbacks_resolve_state_futures_and_fields(client) -> None:
    async_client, server, _ = client
    state = server.get_rollout_state("r1")
    assert state is not None

    rollout_future = state.rollout_future
    grader_future = state.grader_future
    rollout = await async_client.post(
        "/v1/rollout/completed",
        headers={"Authorization": "Bearer secret"},
        json={"rollout_id": "r1", "status": "success"},
    )
    grader = await async_client.post(
        "/v1/grader/completed",
        headers={"Authorization": "Bearer secret"},
        json={
            "rollout_id": "r1",
            "status": "success",
            "samples": {"s1": {"id": "s1", "reward": 1.0}},
        },
    )

    assert rollout.json() == {"status": "ok"}
    assert grader.json() == {"status": "ok"}
    assert (
        await asyncio.wait_for(rollout_future, timeout=0.1)
    ).status is RolloutStatus.SUCCESS
    grader_request = await asyncio.wait_for(grader_future, timeout=0.1)
    assert grader_request.status is GraderStatus.SUCCESS
    assert grader_request.samples == {"s1": RolloutSample(id="s1", reward=1.0)}
    assert state.rollout_callback is not None
    assert state.grader_callback is not None


@pytest.mark.asyncio
async def test_bridge_exception_json_collects_systemic_error(client) -> None:
    async_client, server, bridge = client
    bridge.exc = RuntimeError("provider down")
    bridge.systemic_errors["r1"] = "bad auth"
    state = server.get_rollout_state("r1")
    assert state is not None
    rollout_future = state.rollout_future

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={"model": "openai/osmosis-rollout", "messages": [], "stream": False},
    )

    assert response.status_code == 502
    assert response.json() == {"error": "provider down"}
    assert state.systemic_error == "bad auth"
    callback = await asyncio.wait_for(rollout_future, timeout=0.1)
    assert callback.status is RolloutStatus.FAILURE
    assert callback.err_message == "bad auth"


@pytest.mark.asyncio
async def test_bridge_exception_sse_yields_error_done_and_resolves_waiter(
    client,
) -> None:
    async_client, server, bridge = client
    bridge.exc = RuntimeError("provider down")
    state = server.get_rollout_state("r1")
    assert state is not None
    rollout_future = state.rollout_future

    response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(),
        json={"model": "openai/osmosis-rollout", "messages": []},
    )

    assert response.status_code == 200
    assert 'event: error\ndata: {"error":"provider down"}' in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    callback = await asyncio.wait_for(rollout_future, timeout=0.1)
    assert callback.status is RolloutStatus.FAILURE
    assert callback.err_message == "provider down"


@pytest.mark.asyncio
async def test_stream_completion_cancels_inflight_task_on_disconnect() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowBridge(FakeBridge):
        async def complete(
            self, rollout_id: str, sample_id: str, body: dict[str, Any]
        ) -> Any:
            self.calls.append((rollout_id, sample_id, body))
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    bridge = SlowBridge()
    server = EvalControllerServer(api_key="secret", bridge=bridge)
    server.register_rollout_state(ControllerRolloutState("r1"))
    stream = server._stream_completion(
        "r1",
        "s1",
        "multi_sample",
        {"model": "openai/osmosis-rollout", "messages": []},
    )

    assert await stream.__anext__() == ": ping\n\n"
    next_frame = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    next_frame.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_frame

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert server.get_rollout_state("r1").completion_counts == {}


@pytest.mark.asyncio
async def test_state_not_found_json_and_sse_paths(client) -> None:
    async_client, _, _ = client

    json_response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(rollout_id="missing"),
        json={"model": "openai/osmosis-rollout", "messages": [], "stream": False},
    )
    sse_response = await async_client.post(
        "/chat/completions",
        headers=auth_headers(rollout_id="missing"),
        json={"model": "openai/osmosis-rollout", "messages": []},
    )

    assert json_response.status_code == 500
    assert json_response.json()["detail"] == "rollout not found"
    assert 'event: error\ndata: {"error":"rollout not found"}' in sse_response.text
    assert sse_response.text.endswith("data: [DONE]\n\n")


def test_register_get_pop_rollout_state_methods() -> None:
    server = EvalControllerServer(api_key="secret", bridge=FakeBridge())
    state = ControllerRolloutState("r1")

    server.register_rollout_state(state)

    assert server.get_rollout_state("r1") is state
    assert server.pop_rollout_state("r1") is state
    assert server.get_rollout_state("r1") is None
    assert server.pop_rollout_state("r1") is None


def test_heartbeat_interval_uses_env_with_sane_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSE_HEARTBEAT_INTERVAL_S", "0")
    assert _heartbeat_interval() == 15.0
    monkeypatch.setenv("SSE_HEARTBEAT_INTERVAL_S", "bad")
    assert _heartbeat_interval() == 15.0
    monkeypatch.setenv("SSE_HEARTBEAT_INTERVAL_S", "2.5")
    assert _heartbeat_interval() == 2.5


def test_controller_package_lazily_exports_server_without_heavy_imports() -> None:
    script = "\n".join(
        [
            "import sys",
            "import osmosis_ai.eval.controller as controller",
            "assert 'osmosis_ai.eval.controller.server' not in sys.modules",
            "assert 'osmosis_ai.eval.controller.state' not in sys.modules",
            "assert 'osmosis_ai.eval.controller.litellm_bridge' not in sys.modules",
            "assert controller.EvalControllerServer.__name__ == 'EvalControllerServer'",
            "assert 'osmosis_ai.eval.controller.server' in sys.modules",
            "assert 'litellm' not in sys.modules",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

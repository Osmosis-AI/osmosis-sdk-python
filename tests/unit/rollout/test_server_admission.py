"""Admission control and cancellation on the rollout server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server import app as server_app_module
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.types import ExecutionRequest, ExecutionResult, RolloutStatus


@pytest.fixture(autouse=True)
def _isolate_side_effects(monkeypatch):
    """Terminal callbacks succeed without a network; archiving is disabled."""

    async def _delivered(*, url, payload, headers):
        class _Resp:
            status_code = 200

        return _Resp()

    async def _no_archive(**kwargs):
        return None

    monkeypatch.setattr(server_app_module, "post_json_with_retry", _delivered)
    monkeypatch.setattr(server_app_module, "save_trajectory", _no_archive)


class StubBackend(ExecutionBackend):
    def __init__(self, capacity: bool = True) -> None:
        self.capacity = capacity
        self.executed = False
        self.execute_count = 0
        self.cancel_args: tuple | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        self.executed = True
        self.execute_count += 1
        await on_workflow_complete(ExecutionResult(status=RolloutStatus.SUCCESS))

    def has_capacity(self) -> bool:
        return self.capacity

    def cancel_rollouts(
        self,
        ids: Sequence[str] | None = None,
        prefix: str | None = None,
        all: bool = False,
    ) -> dict[str, str]:
        self.cancel_args = (ids, prefix, all)
        return {"r1": "cancelled_queued"}

    def rollout_status(self, rollout_id: str) -> dict | None:
        if rollout_id == "known":
            return {"status": "success", "reward": 1.0, "err_message": None}
        return None

    def health(self) -> dict:
        return {"status": "ok", "backend": "stub", "max_queue_depth": 2}


def init_body() -> dict:
    return {
        "rollout_id": "r1",
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": "http://llm",
        "completion_callback_url": "http://controller/complete",
    }


def test_full_backend_rejects_with_429():
    backend = StubBackend(capacity=False)
    client = TestClient(create_rollout_server(backend=backend))
    response = client.post("/rollout", json=init_body())
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert backend.executed is False


def test_backend_with_capacity_accepts():
    backend = StubBackend(capacity=True)
    client = TestClient(create_rollout_server(backend=backend))
    response = client.post("/rollout", json=init_body())
    assert response.status_code == 202
    assert backend.executed is True


def test_cancel_requires_exactly_one_selector():
    client = TestClient(create_rollout_server(backend=StubBackend()))
    assert client.post("/rollout/cancel", json={}).status_code == 422
    assert (
        client.post("/rollout/cancel", json={"ids": ["a"], "all": True}).status_code
        == 422
    )


def test_cancel_forwards_selector_and_returns_dispositions():
    backend = StubBackend()
    client = TestClient(create_rollout_server(backend=backend))
    response = client.post("/rollout/cancel", json={"prefix": "job-1-"})
    assert response.status_code == 200
    assert response.json() == {"dispositions": {"r1": "cancelled_queued"}}
    assert backend.cancel_args == (None, "job-1-", False)


def test_cancel_default_backend_reports_nothing():
    class NoCancelBackend(StubBackend):
        cancel_rollouts = ExecutionBackend.cancel_rollouts

    client = TestClient(create_rollout_server(backend=NoCancelBackend()))
    response = client.post("/rollout/cancel", json={"all": True})
    assert response.status_code == 200
    assert response.json() == {"dispositions": {}}


def test_status_returns_backend_state():
    client = TestClient(create_rollout_server(backend=StubBackend()))
    response = client.get("/rollout/known/status")
    assert response.status_code == 200
    assert response.json() == {
        "rollout_id": "known",
        "status": "success",
        "reward": 1.0,
        "err_message": None,
    }


def test_status_unknown_rollout():
    client = TestClient(create_rollout_server(backend=StubBackend()))
    body = client.get("/rollout/nope/status").json()
    assert body["status"] == "unknown"
    assert body["reward"] is None


def test_accepted_rollout_status_is_backend_authoritative_unknown():
    client = TestClient(create_rollout_server(backend=StubBackend()))
    assert client.post("/rollout", json=init_body()).status_code == 202
    body = client.get("/rollout/r1/status").json()
    assert body["status"] == "unknown"
    assert body["rollout_id"] == "r1"


def test_local_backend_status_is_unknown_not_queued() -> None:
    from osmosis_ai.rollout.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout.backend.local.backend import LocalBackend
    from osmosis_ai.rollout.context import AgentWorkflowContext

    class _NoopWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> list[dict[str, str]]:
            return [{"role": "assistant", "content": "ok"}]

    backend = LocalBackend(workflow=_NoopWorkflow)

    async def _noop_execute(
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        return None

    backend.execute = _noop_execute  # type: ignore[method-assign]
    client = TestClient(create_rollout_server(backend=backend))
    assert client.post("/rollout", json=init_body()).status_code == 202
    body = client.get("/rollout/r1/status").json()
    assert body["status"] == "unknown"


async def _post_rollout_with_failing_send(app: Any, body: dict) -> None:
    """Drive one POST /rollout whose response send fails (client vanished)."""
    payload = json.dumps(body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/rollout",
        "raw_path": b"/rollout",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 80),
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict) -> None:
        raise ConnectionResetError("client disconnected before the response")

    with pytest.raises(ConnectionResetError):
        await app(scope, receive, send)


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_execution_survives_failed_response_send() -> None:
    backend = StubBackend()
    app = create_rollout_server(backend=backend)
    await _post_rollout_with_failing_send(app, init_body())
    await _settle()
    assert backend.execute_count == 1


class DrainStubBackend(StubBackend):
    """Backend whose execution outlives the shutdown signal."""

    def __init__(self, work_sec: float, events: list[str]) -> None:
        super().__init__()
        self.work_sec = work_sec
        self.events = events

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        self.executed = True
        self.execute_count += 1
        try:
            await asyncio.sleep(self.work_sec)
        except asyncio.CancelledError:
            self.events.append("rollout-cancelled")
            raise
        self.events.append("rollout-finished")
        await on_workflow_complete(ExecutionResult(status=RolloutStatus.SUCCESS))


async def test_lifespan_exit_drains_rollouts_before_caller_lifespan_closes() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        events.append("lifespan-closed")

    backend = DrainStubBackend(0.05, events)
    app = create_rollout_server(backend=backend, lifespan=lifespan)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://rollout"
    ) as client:
        async with app.router.lifespan_context(app):
            assert (await client.post("/rollout", json=init_body())).status_code == 202
            await _settle()
            assert events == []  # still running when shutdown begins
    assert events == ["rollout-finished", "lifespan-closed"]


async def test_lifespan_exit_cancels_rollouts_that_outlast_the_drain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server_app_module, "_SHUTDOWN_DRAIN_SEC", 0.01)
    events: list[str] = []
    backend = DrainStubBackend(30.0, events)
    app = create_rollout_server(backend=backend)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://rollout"
    ) as client:
        async with app.router.lifespan_context(app):
            assert (await client.post("/rollout", json=init_body())).status_code == 202
            await _settle()
    assert events == ["rollout-cancelled"]


def test_health_instance_id_is_captured_at_construction(monkeypatch) -> None:
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", "srv-initial")
    app = create_rollout_server(backend=StubBackend())
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", "srv-mutated")
    client = TestClient(app)
    assert client.get("/health").json()["instance_id"] == "srv-initial"


def test_health_omits_instance_id_when_env_absent(monkeypatch):
    monkeypatch.delenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", raising=False)
    client = TestClient(create_rollout_server(backend=StubBackend()))
    body = client.get("/health").json()
    assert body == {"status": "ok", "backend": "stub", "max_queue_depth": 2}
    assert "instance_id" not in body


def test_health_exposes_instance_id_and_preserves_backend_fields(monkeypatch):
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", "srv-abc123")
    client = TestClient(create_rollout_server(backend=StubBackend()))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"
    assert body["max_queue_depth"] == 2
    assert body["instance_id"] == "srv-abc123"

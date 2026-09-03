from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server import app as server_app_module
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.types import (
    POLLING_LEASE_HEADER,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    RolloutStatus,
)


@pytest.fixture(autouse=True)
def no_archive(monkeypatch):
    async def skip(**kwargs):
        return None

    monkeypatch.setattr(server_app_module, "save_trajectory", skip)


class StubBackend(ExecutionBackend):
    def __init__(self, capacity: bool = True) -> None:
        self.capacity = capacity
        self.executed = False
        self.execute_count = 0
        self.cancel_args: tuple | None = None

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        self.executed = True
        self.execute_count += 1
        result = ExecutionResult(status=RolloutStatus.SUCCESS)
        return ExecutionOutcome(workflow=result, grader=result)

    def has_capacity(self) -> bool:
        return self.capacity

    def cancel_rollouts(
        self,
        ids: Sequence[str] | None = None,
        prefix: str | None = None,
        all: bool = False,
    ) -> dict[str, str]:
        self.cancel_args = (ids, prefix, all)
        return {rollout_id: "cancelled_running" for rollout_id in ids or []}

    def health(self) -> dict:
        return {"status": "ok", "backend": "stub", "max_queue_depth": 2}


class BlockingBackend(StubBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        self.executed = True
        self.execute_count += 1
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def init_body() -> dict[str, Any]:
    return {
        "rollout_id": "r1",
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": "http://llm",
        "llm_api_key": "llm-key",
        "grade": True,
    }


def lease_headers(response: Any) -> dict[str, str]:
    return {POLLING_LEASE_HEADER: response.json()["polling_lease_token"]}


def test_full_backend_rejects_with_429() -> None:
    backend = StubBackend(capacity=False)
    client = TestClient(create_rollout_server(backend=backend))
    response = client.post("/rollout", json=init_body())
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert backend.executed is False


def test_admission_reports_server_poll_configuration() -> None:
    backend = StubBackend()
    with TestClient(create_rollout_server(backend=backend)) as client:
        response = client.post("/rollout", json=init_body())
    assert response.status_code == 202
    body = response.json()
    assert isinstance(body.pop("polling_lease_token"), str)
    assert body == {
        "rollout_id": "r1",
        "status": "queued",
        "result_wait_timeout_sec": 30.0,
        "polling_lease_timeout_sec": 120.0,
    }
    assert backend.executed is True


def test_result_returns_finished_outcome() -> None:
    with TestClient(create_rollout_server(backend=StubBackend())) as client:
        admission = client.post("/rollout", json=init_body())
        assert admission.status_code == 202
        response = client.get("/rollout/r1/result", headers=lease_headers(admission))
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_result_requires_a_polling_lease() -> None:
    with TestClient(create_rollout_server(backend=StubBackend())) as client:
        client.post("/rollout", json=init_body())
        response = client.get("/rollout/r1/result")
    assert response.status_code == 422


def test_result_rejects_an_invalid_lease() -> None:
    with TestClient(create_rollout_server(backend=StubBackend())) as client:
        client.post("/rollout", json=init_body())
        response = client.get(
            "/rollout/r1/result",
            headers={POLLING_LEASE_HEADER: "wrong"},
        )
    assert response.status_code == 403


def test_result_returns_404_for_unknown_rollout() -> None:
    client = TestClient(create_rollout_server(backend=StubBackend()))
    response = client.get(
        "/rollout/missing/result",
        headers={POLLING_LEASE_HEADER: "unknown-lease"},
    )
    assert response.status_code == 404


def test_duplicate_rollout_id_is_rejected() -> None:
    with TestClient(create_rollout_server(backend=BlockingBackend())) as client:
        assert client.post("/rollout", json=init_body()).status_code == 202
        duplicate = client.post("/rollout", json=init_body())
    assert duplicate.status_code == 409


async def test_result_wait_uses_server_configuration() -> None:
    backend = BlockingBackend()
    app = create_rollout_server(
        backend=backend,
        result_wait_timeout_sec=0.02,
        polling_lease_timeout_sec=0.2,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            admission = await client.post("/rollout", json=init_body())
            started = time.monotonic()
            response = await client.get(
                "/rollout/r1/result", headers=lease_headers(admission)
            )
            elapsed = time.monotonic() - started
    assert response.json()["status"] == "running"
    assert elapsed >= 0.015


async def test_expired_lease_fails_and_cancels_the_rollout() -> None:
    backend = BlockingBackend()
    app = create_rollout_server(
        backend=backend,
        result_wait_timeout_sec=0.01,
        polling_lease_timeout_sec=0.05,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            admission = await client.post("/rollout", json=init_body())
            await backend.started.wait()
            await asyncio.wait_for(backend.cancelled.wait(), timeout=1.0)
            response = await client.get(
                "/rollout/r1/result", headers=lease_headers(admission)
            )
    assert response.json()["status"] == "failure"
    assert response.json()["err_category"] == "lease_expired"
    assert backend.cancel_args == (["r1"], None, False)


def test_cancel_requires_exactly_one_selector() -> None:
    client = TestClient(create_rollout_server(backend=StubBackend()))
    assert client.post("/rollout/cancel", json={}).status_code == 422
    assert (
        client.post("/rollout/cancel", json={"ids": ["a"], "all": True}).status_code
        == 422
    )


def test_cancel_forwards_selector() -> None:
    backend = StubBackend()
    client = TestClient(create_rollout_server(backend=backend))
    response = client.post("/rollout/cancel", json={"ids": ["r1"]})
    assert response.json() == {"dispositions": {"r1": "cancelled_running"}}
    assert backend.cancel_args == (["r1"], None, False)


async def post_rollout_with_failing_send(app: Any, body: dict) -> None:
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


async def test_execution_survives_failed_response_send() -> None:
    backend = StubBackend()
    app = create_rollout_server(backend=backend)
    await post_rollout_with_failing_send(app, init_body())
    for _ in range(10):
        await asyncio.sleep(0)
    assert backend.execute_count == 1


class DrainStubBackend(StubBackend):
    def __init__(self, work_sec: float, events: list[str]) -> None:
        super().__init__()
        self.work_sec = work_sec
        self.events = events

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        self.executed = True
        self.execute_count += 1
        try:
            await asyncio.sleep(self.work_sec)
        except asyncio.CancelledError:
            self.events.append("rollout-cancelled")
            raise
        self.events.append("rollout-finished")
        result = ExecutionResult(status=RolloutStatus.SUCCESS)
        return ExecutionOutcome(workflow=result, grader=result)


async def test_lifespan_drains_before_caller_lifespan_closes() -> None:
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
            await client.post("/rollout", json=init_body())
            await asyncio.sleep(0)
    assert events == ["rollout-finished", "lifespan-closed"]


async def test_lifespan_cancels_work_past_drain_limit(monkeypatch) -> None:
    monkeypatch.setattr(server_app_module, "_SHUTDOWN_DRAIN_SEC", 0.01)
    events: list[str] = []
    backend = DrainStubBackend(30.0, events)
    app = create_rollout_server(backend=backend)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://rollout"
    ) as client:
        async with app.router.lifespan_context(app):
            await client.post("/rollout", json=init_body())
            await asyncio.sleep(0)
    assert events == ["rollout-cancelled"]


def test_health_preserves_backend_fields(monkeypatch) -> None:
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", "srv-abc123")
    client = TestClient(create_rollout_server(backend=StubBackend()))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"
    assert body["instance_id"] == "srv-abc123"

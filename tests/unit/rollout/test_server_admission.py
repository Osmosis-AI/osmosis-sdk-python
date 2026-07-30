"""Admission-control coverage for ``create_rollout_server``."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server import app as app_module
from osmosis_ai.rollout.server.app import create_rollout_server
from osmosis_ai.rollout.types import ExecutionRequest, RolloutInitRequest


class _CapacityBackend(ExecutionBackend):
    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        max_queue_depth: int | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._max_queue_depth = max_queue_depth

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrent

    @property
    def max_queue_depth(self) -> int | None:
        return self._max_queue_depth

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "capacity-test"}

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:  # pragma: no cover - _handle_rollout is replaced in these tests
        raise AssertionError("execute should not be called")


class _BlockingHandler:
    def __init__(self, expected_entries: int) -> None:
        self.expected_entries = expected_entries
        self.entries = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(
        self,
        backend: ExecutionBackend,
        request: RolloutInitRequest,
    ) -> None:
        del backend, request
        self.entries += 1
        if self.entries >= self.expected_entries:
            self.entered.set()
        await self.release.wait()


def _payload(rollout_id: str) -> dict[str, Any]:
    return {
        "rollout_id": rollout_id,
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": f"http://controller/sessions/{rollout_id}/v1",
        "completion_callback_url": (
            f"http://controller/v1/rollout/{rollout_id}/completed"
        ),
    }


async def _wait_for(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=5)


async def _health(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/health")
    assert response.status_code == 200
    return response.json()


class TestBoundedAdmission:
    async def test_full_queue_returns_429_and_health_tracks_release(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _BlockingHandler(expected_entries=2)
        monkeypatch.setattr(app_module, "_handle_rollout", handler)
        app = create_rollout_server(
            backend=_CapacityBackend(max_concurrent=1, max_queue_depth=1),
            configure_logging=False,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout.test"
        ) as client:
            first = asyncio.create_task(client.post("/rollout", json=_payload("r1")))
            second = asyncio.create_task(client.post("/rollout", json=_payload("r2")))
            try:
                await _wait_for(handler.entered)
                health = await _health(client)
                assert health["backend"] == "capacity-test"
                assert health["capacity"] == {
                    "max_concurrent": 1,
                    "max_queue_depth": 1,
                    "in_flight": 2,
                    "queue_depth": 1,
                    "available": 0,
                    "accepting": False,
                }

                rejected = await asyncio.wait_for(
                    client.post("/rollout", json=_payload("r3")), timeout=5
                )
                assert rejected.status_code == 429
                assert rejected.json() == {"detail": "Rollout queue is full"}
            finally:
                handler.release.set()
                responses = await asyncio.wait_for(
                    asyncio.gather(first, second), timeout=5
                )

            assert [response.status_code for response in responses] == [200, 200]
            assert (await _health(client))["capacity"] == {
                "max_concurrent": 1,
                "max_queue_depth": 1,
                "in_flight": 0,
                "queue_depth": 0,
                "available": 2,
                "accepting": True,
            }

    async def test_zero_queue_depth_rejects_while_execution_is_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _BlockingHandler(expected_entries=1)
        monkeypatch.setattr(app_module, "_handle_rollout", handler)
        app = create_rollout_server(
            backend=_CapacityBackend(max_concurrent=1, max_queue_depth=0),
            configure_logging=False,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout.test"
        ) as client:
            active = asyncio.create_task(client.post("/rollout", json=_payload("r1")))
            try:
                await _wait_for(handler.entered)
                rejected = await asyncio.wait_for(
                    client.post("/rollout", json=_payload("r2")), timeout=5
                )
                assert rejected.status_code == 429
                assert (await _health(client))["capacity"] == {
                    "max_concurrent": 1,
                    "max_queue_depth": 0,
                    "in_flight": 1,
                    "queue_depth": 0,
                    "available": 0,
                    "accepting": False,
                }
            finally:
                handler.release.set()
                response = await asyncio.wait_for(active, timeout=5)
            assert response.status_code == 200

    async def test_background_error_releases_reservation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail(
            backend: ExecutionBackend,
            request: RolloutInitRequest,
        ) -> None:
            del backend, request
            raise RuntimeError("background failed")

        monkeypatch.setattr(app_module, "_handle_rollout", fail)
        app = create_rollout_server(
            backend=_CapacityBackend(max_concurrent=1, max_queue_depth=0),
            configure_logging=False,
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout.test"
        ) as client:
            response = await client.post("/rollout", json=_payload("r1"))
            assert response.status_code == 200
            assert (await _health(client))["capacity"]["in_flight"] == 0
            assert (await _health(client))["capacity"]["accepting"] is True

    async def test_background_cancellation_releases_reservation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _BlockingHandler(expected_entries=1)
        monkeypatch.setattr(app_module, "_handle_rollout", handler)
        app = create_rollout_server(
            backend=_CapacityBackend(max_concurrent=1, max_queue_depth=0),
            configure_logging=False,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout.test"
        ) as client:
            active = asyncio.create_task(client.post("/rollout", json=_payload("r1")))
            await _wait_for(handler.entered)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active

            capacity = (await _health(client))["capacity"]
            assert capacity["in_flight"] == 0
            assert capacity["accepting"] is True


class TestUnboundedAdmission:
    async def test_backend_without_queue_bound_remains_unbounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _BlockingHandler(expected_entries=3)
        monkeypatch.setattr(app_module, "_handle_rollout", handler)
        app = create_rollout_server(
            backend=_CapacityBackend(max_concurrent=1, max_queue_depth=None),
            configure_logging=False,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout.test"
        ) as client:
            requests = [
                asyncio.create_task(client.post("/rollout", json=_payload(f"r{i}")))
                for i in range(3)
            ]
            try:
                await _wait_for(handler.entered)
                assert (await _health(client))["capacity"] == {
                    "max_concurrent": 1,
                    "max_queue_depth": None,
                    "in_flight": 3,
                    "queue_depth": 2,
                    "available": None,
                    "accepting": True,
                }
            finally:
                handler.release.set()
                responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=5)
            assert [response.status_code for response in responses] == [200, 200, 200]

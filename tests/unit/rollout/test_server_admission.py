"""Admission control and cancellation on the rollout server."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.types import ExecutionRequest


class StubBackend(ExecutionBackend):
    def __init__(self, capacity: bool = True) -> None:
        self.capacity = capacity
        self.executed = False
        self.cancel_args: tuple | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        self.executed = True

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
    assert response.status_code == 200
    assert backend.executed is True


def test_cancel_requires_exactly_one_selector():
    client = TestClient(create_rollout_server(backend=StubBackend()))
    assert client.post("/rollout/cancel", json={}).status_code == 422
    assert (
        client.post(
            "/rollout/cancel", json={"ids": ["a"], "all": True}
        ).status_code
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

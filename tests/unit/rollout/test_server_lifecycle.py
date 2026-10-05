from __future__ import annotations

import asyncio

import httpx
import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.client import RolloutClient, RolloutProtocolError
from osmosis_ai.rollout.context import get_rollout_context
from osmosis_ai.rollout.server import app as server_module
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.types import (
    ExecutionOutcome,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutStatus,
)


class Backend(ExecutionBackend):
    async def execute(self, request):
        self.process_id = get_rollout_context().process_id
        return ExecutionOutcome(workflow=ExecutionResult(status=RolloutStatus.SUCCESS))


BODY = {
    "rollout_id": "one",
    "initial_messages": [],
    "chat_completions_url": "https://model.example/v1",
}


@pytest.mark.parametrize("api_key", ["", " ", "\t", "\n", " \t\r\n "])
def test_whitespace_credentials_are_rejected_before_creating_client(
    api_key, monkeypatch
):
    def unexpected_client():
        pytest.fail("invalid credentials must not allocate an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        RolloutClient(url="http://test", api_key=api_key)
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        create_rollout_server(backend=Backend(), api_key=api_key)


@pytest.fixture(autouse=True)
def no_archive(monkeypatch):
    async def skip(**kwargs):
        pass

    monkeypatch.setattr(server_module, "save_trajectory", skip)


async def test_authenticated_health_and_rollout_share_process_identity():
    backend = Backend()
    app = create_rollout_server(backend=backend, api_key="secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        assert (await http.get("/health")).status_code == 401
        assert (await http.post("/rollout", json=BODY)).status_code == 401
        client = RolloutClient(url="http://test", http_client=http, api_key="secret")
        health = await client.health()
        assert health["lifecycle"] == {"accepting_rollouts": True, "active_rollouts": 0}
        result = await client.run_rollout(
            initial_messages=[],
            chat_completions_url="https://model.example/v1",
            rollout_id="one",
        )
        assert result.status == RolloutStatus.SUCCESS
        assert backend.process_id == health["process_id"]
        assert await client.wait_idle(timeout_sec=1) == await client.health()
    await app.state.rollout_futures.close()


async def test_wait_idle_waits_through_trajectory_finalization(monkeypatch):
    finalizing, finish = asyncio.Event(), asyncio.Event()

    async def archive(**kwargs):
        finalizing.set()
        await finish.wait()

    monkeypatch.setattr(server_module, "save_trajectory", archive)
    app = create_rollout_server(backend=Backend())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        assert (await http.post("/rollout", json=BODY)).status_code == 202
        await finalizing.wait()
        client = RolloutClient(url="http://test", http_client=http)
        assert (await client.health())["lifecycle"]["active_rollouts"] == 1
        with pytest.raises(TimeoutError):
            await client.wait_idle(timeout_sec=0.01, poll_interval_sec=0.001)
        waiting = asyncio.create_task(
            client.wait_idle(timeout_sec=1, poll_interval_sec=0.001)
        )
        await asyncio.sleep(0)
        assert not waiting.done()
        finish.set()
        assert (await waiting)["lifecycle"]["active_rollouts"] == 0
    await app.state.rollout_futures.close()


async def test_non_harbor_backends_keep_existing_colon_ids():
    app = create_rollout_server(backend=Backend())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        client = RolloutClient(url="http://test", http_client=http)
        result = await client.run_rollout(
            initial_messages=[],
            chat_completions_url="https://model.example/v1",
            rollout_id="run:1",
        )
        assert result.status == RolloutStatus.SUCCESS
        assert result.rollout_id == "run:1"
    await app.state.rollout_futures.close()


async def test_cancellation_during_binding_cannot_hide_scheduled_work(monkeypatch):
    app = create_rollout_server(backend=Backend())
    entered = asyncio.Event()
    registry = app.state.rollout_futures

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(registry, "bind_task", blocked)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        admission = asyncio.create_task(http.post("/rollout", json=BODY))
        await entered.wait()
        admission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await admission
        client = RolloutClient(url="http://test", http_client=http)
        await client.wait_idle(timeout_sec=1, poll_interval_sec=0.001)
        assert registry.entries == {}
    await registry.close()


async def test_wait_idle_rejects_legacy_shapes_and_process_replacement():
    responses = iter(
        [
            {"in_flight": 0, "running": 0, "queued": 0},
            {
                "process_id": "new",
                "lifecycle": {"accepting_rollouts": True, "active_rollouts": 0},
            },
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=next(responses))
        )
    ) as http:
        client = RolloutClient(url="https://rollout.example", http_client=http)
        with pytest.raises(RolloutProtocolError, match="identity"):
            await client.wait_idle()
        with pytest.raises(RolloutProtocolError, match="restarted"):
            await client.wait_idle(process_id="old")


async def test_health_auth_does_not_follow_redirects():
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.example"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as http:
        client = RolloutClient(
            url="https://rollout.example", http_client=http, api_key="secret"
        )
        with pytest.raises(RolloutProtocolError):
            await client.health()
    assert len(seen) == 1
    assert all(request.headers["authorization"] == "Bearer secret" for request in seen)


async def test_process_identity_changes_even_with_stable_instance(monkeypatch):
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_INSTANCE_ID", "deployment")
    states = []
    for _ in range(2):
        app = create_rollout_server(backend=Backend())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as http:
            states.append((await http.get("/health")).json())
        await app.state.rollout_futures.close()
    assert states[0]["instance_id"] == states[1]["instance_id"] == "deployment"
    assert states[0]["process_id"] != states[1]["process_id"]


async def test_wait_idle_retries_transient_health_but_not_auth_errors():
    statuses = iter([503, 200, 401])

    def respond(request):
        return httpx.Response(
            next(statuses),
            json={
                "process_id": "stable",
                "lifecycle": {"accepting_rollouts": True, "active_rollouts": 0},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = RolloutClient(url="https://rollout.example", http_client=http)
        assert (await client.wait_idle(timeout_sec=1, poll_interval_sec=0.001))[
            "process_id"
        ] == "stable"
        with pytest.raises(RolloutProtocolError) as error:
            await client.wait_idle(timeout_sec=1)
        assert error.value.status_code == 401


async def test_crashed_rollout_task_resolves_failure_result(monkeypatch):
    app = create_rollout_server(backend=Backend())
    registry = app.state.rollout_futures

    async def crashing_handle(*args, **kwargs):
        raise RuntimeError("catastrophic crash in server handling")

    monkeypatch.setattr(server_module, "_handle_rollout", crashing_handle)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        client = RolloutClient(url="http://test", http_client=http)
        result = await client.run_rollout(
            initial_messages=[],
            chat_completions_url="https://model.example/v1",
            rollout_id="crash-1",
        )
        assert result.status is RolloutStatus.FAILURE
        assert result.err_category is RolloutErrorCategory.INTERNAL_ERROR
        assert result.err_message == "rollout task crashed"
    await registry.close()

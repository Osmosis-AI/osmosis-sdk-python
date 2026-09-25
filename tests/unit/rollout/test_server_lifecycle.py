from __future__ import annotations

import asyncio

import httpx
import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.client import RolloutClient, RolloutProtocolError
from osmosis_ai.rollout.server import app as server_module
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.types import ExecutionOutcome, ExecutionResult, RolloutStatus


class Backend(ExecutionBackend):
    async def execute(self, request):
        return ExecutionOutcome(workflow=ExecutionResult(status=RolloutStatus.SUCCESS))


BODY = {
    "rollout_id": "one",
    "initial_messages": [],
    "chat_completions_url": "https://model.example/v1",
}


@pytest.fixture(autouse=True)
def no_archive(monkeypatch):
    async def skip(**kwargs):
        pass

    monkeypatch.setattr(server_module, "save_trajectory", skip)


async def test_authenticated_drain_fences_admission_and_is_idempotent():
    app = create_rollout_server(backend=Backend(), api_key="secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        assert (await http.get("/health")).status_code == 401
        assert (await http.post("/drain", json={})).status_code == 401
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
        drained = await client.drain(timeout_sec=0)
        assert drained.drained and not drained.accepting_rollouts
        assert drained.process_id == health["process_id"]
        assert drained == await client.drain(timeout_sec=0)
        response = await http.post(
            "/rollout", json=BODY, headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 503
        assert await client.wait_idle(timeout_sec=1) == await client.health()
    await app.state.rollout_futures.close()


async def test_drain_waits_through_trajectory_finalization(monkeypatch):
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
        result = await client.drain(timeout_sec=0.01)
        assert not result.drained and result.active_rollouts == 1
        assert result.rollout_ids == ["one"]
        with pytest.raises(TimeoutError):
            await client.wait_idle(timeout_sec=0.01)
        assert (
            await http.post("/rollout", json={**BODY, "rollout_id": "two"})
        ).status_code == 503
        waiting = asyncio.create_task(client.drain(timeout_sec=1))
        await asyncio.sleep(0)
        assert not waiting.done()
        finish.set()
        assert (await waiting).drained
    await app.state.rollout_futures.close()


async def test_admission_in_progress_cannot_escape_drain(monkeypatch):
    app = create_rollout_server(backend=Backend())
    entered, release = asyncio.Event(), asyncio.Event()
    registry = app.state.rollout_futures
    register = registry.register

    async def blocked(rollout_id):
        entered.set()
        await release.wait()
        return await register(rollout_id)

    monkeypatch.setattr(registry, "register", blocked)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as http:
        admission = asyncio.create_task(http.post("/rollout", json=BODY))
        await entered.wait()
        draining = asyncio.create_task(http.post("/drain", json={"timeout_sec": 1}))
        await asyncio.sleep(0)
        assert not draining.done()
        release.set()
        assert (await admission).status_code == 202
        result = (await draining).json()
        assert result["drained"] and result["rollout_ids"] == ["one"]
        assert (
            await http.post("/rollout", json={**BODY, "rollout_id": "two"})
        ).status_code == 503
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
        with pytest.raises(RolloutProtocolError):
            await client.drain()
    assert len(seen) == 2
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

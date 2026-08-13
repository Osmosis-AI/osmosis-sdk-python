"""Tests for the localhost callback listener."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport

from osmosis_ai.rollout.controller.listener import (
    CallbackListener,
    create_callback_app,
)
from osmosis_ai.rollout.controller.store import CallbackStore, TerminalCallbackResult
from osmosis_ai.rollout.types import GraderStatus, RolloutSample, RolloutStatus

ROLLOUT_ID = "b" * 32
TOKEN = "callback-secret"


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _completion_body(rollout_id: str = ROLLOUT_ID) -> dict[str, object]:
    return {"status": RolloutStatus.SUCCESS, "rollout_id": rollout_id}


def _grader_body(rollout_id: str = ROLLOUT_ID) -> dict[str, object]:
    return {
        "status": GraderStatus.SUCCESS,
        "rollout_id": rollout_id,
        "sample": {
            "messages": [{"role": "assistant", "content": "ok"}],
            "reward": 1.0,
        },
    }


async def _client(store: CallbackStore) -> httpx.AsyncClient:
    app = create_callback_app(store, auth_token=TOKEN)
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://127.0.0.1",
    )


async def test_completion_and_grader_routes_require_bearer_auth() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        completion_path = f"/v1/rollouts/{ROLLOUT_ID}/completion"
        grader_path = f"/v1/rollouts/{ROLLOUT_ID}/grader"
        assert (
            await client.post(completion_path, json=_completion_body())
        ).status_code == 401
        assert (
            await client.post(
                completion_path, json=_completion_body(), headers=_auth("nope")
            )
        ).status_code == 401
        assert (await client.post(grader_path, json=_grader_body())).status_code == 401
        assert (
            await client.post(grader_path, json=_grader_body(), headers=_auth("nope"))
        ).status_code == 401


async def test_valid_bearer_auth_accepts_callbacks() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        completion = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/completion",
            json=_completion_body(),
            headers=_auth(),
        )
        grader = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/grader",
            json=_grader_body(),
            headers=_auth(),
        )
    assert completion.status_code == 200
    assert grader.status_code == 200
    assert grader.json() == {"ok": True}


async def test_body_rollout_id_must_match_path() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        response = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/grader",
            json=_grader_body(rollout_id="c" * 32),
            headers=_auth(),
        )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_id", [".", "..", "has/slash", "has\\slash"])
async def test_malicious_or_invalid_rollout_ids_are_rejected(bad_id: str) -> None:
    store = CallbackStore()
    async with await _client(store) as client:
        response = await client.post(
            f"/v1/rollouts/{bad_id}/grader",
            json={
                "status": GraderStatus.SUCCESS,
                "sample": RolloutSample(messages=[]).model_dump(mode="json"),
            },
            headers=_auth(),
        )
    assert response.status_code in {400, 404, 422}


async def test_grader_response_waits_for_commit_hook() -> None:
    released = asyncio.Event()
    started = asyncio.Event()
    order: list[str] = []

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        order.append("commit")
        started.set()
        await released.wait()
        order.append("committed")
        return {"ok": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:

        async def post() -> None:
            response = await client.post(
                f"/v1/rollouts/{ROLLOUT_ID}/grader",
                json=_grader_body(),
                headers=_auth(),
            )
            assert response.status_code == 200
            order.append("http-returned")

        task = asyncio.create_task(post())
        await started.wait()
        assert order == ["commit"]
        released.set()
        await task
    assert order == ["commit", "committed", "http-returned"]


@pytest.mark.parametrize("status", ["queued", "running", "grading", "unknown"])
async def test_completion_rejects_non_terminal_statuses(status: str) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        response = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/completion",
            json={"status": status, "rollout_id": ROLLOUT_ID},
            headers=_auth(),
        )
    assert response.status_code == 422
    assert store.completion_for(ROLLOUT_ID) is None


async def test_grader_rejects_pending_status() -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        response = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/grader",
            json={"status": "pending", "rollout_id": ROLLOUT_ID, "sample": None},
            headers=_auth(),
        )
    assert response.status_code == 422
    assert store.terminal_for(ROLLOUT_ID) is None
    assert commits == 0


async def test_listener_binds_reserved_localhost_socket() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    listener = CallbackListener(store, auth_token=TOKEN, host="127.0.0.1", port=0)
    async with listener:
        assert listener.base_url.startswith("http://127.0.0.1:")
        async with httpx.AsyncClient() as http:
            response = await http.post(
                listener.completion_url(ROLLOUT_ID),
                json=_completion_body(),
                headers=_auth(),
            )
        assert response.status_code == 200
        assert listener.grader_url(ROLLOUT_ID).endswith(
            f"/v1/rollouts/{ROLLOUT_ID}/grader"
        )


def test_empty_auth_token_is_rejected() -> None:
    store = CallbackStore()
    with pytest.raises(ValueError, match="auth_token"):
        create_callback_app(store, auth_token="")
    with pytest.raises(ValueError, match="auth_token"):
        CallbackListener(store, auth_token="   ")


def test_non_loopback_bind_host_is_rejected() -> None:
    store = CallbackStore()
    with pytest.raises(ValueError, match="loopback"):
        CallbackListener(store, auth_token=TOKEN, host="0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        CallbackListener(store, auth_token=TOKEN, host="192.168.1.1")


@pytest.mark.parametrize("host", ["::1", "[::1]", "::ffff:127.0.0.1"])
def test_ipv6_hosts_are_rejected_at_construction(host: str) -> None:
    # The listener binds an AF_INET socket; claiming ::1 support would fail
    # later with an opaque bind error, so IPv6 is rejected explicitly.
    store = CallbackStore()
    with pytest.raises(ValueError, match="loopback"):
        CallbackListener(store, auth_token=TOKEN, host=host)


async def test_localhost_normalizes_to_ipv4_loopback() -> None:
    store = CallbackStore()
    listener = CallbackListener(store, auth_token=TOKEN, host="localhost")
    async with listener:
        assert listener.base_url.startswith("http://127.0.0.1:")


async def test_callback_urls_encode_rollout_ids() -> None:
    from urllib.parse import quote

    store = CallbackStore()
    listener = CallbackListener(store, auth_token=TOKEN)
    async with listener:
        special_id = "rid:1"
        encoded = quote(special_id, safe="")
        assert listener.completion_url(special_id).endswith(
            f"/v1/rollouts/{encoded}/completion"
        )
        assert listener.grader_url(special_id).endswith(
            f"/v1/rollouts/{encoded}/grader"
        )


async def _serve_until_exit(self, sockets=None):
    self.started = True
    while not self.should_exit:
        await asyncio.sleep(0)


async def test_failed_start_resets_and_is_retryable(monkeypatch) -> None:
    import uvicorn

    store = CallbackStore()
    listener = CallbackListener(store, auth_token=TOKEN)

    async def boom(self, sockets=None):
        raise RuntimeError("serve exploded")

    monkeypatch.setattr(uvicorn.Server, "serve", boom)
    with pytest.raises(RuntimeError, match="serve exploded"):
        await listener.start()
    assert listener._server._task is None
    assert listener._server._socket is None
    assert listener._server._base_url is None

    monkeypatch.setattr(uvicorn.Server, "serve", _serve_until_exit)
    url = await listener.start()
    assert url.startswith("http://127.0.0.1:")
    await listener.stop()
    assert listener._server._task is None
    assert listener._server._socket is None


async def test_start_timeout_resets_and_is_retryable(monkeypatch) -> None:
    import uvicorn

    from osmosis_ai.rollout.controller import listener as listener_module

    monkeypatch.setattr(listener_module, "_START_POLL_ATTEMPTS", 2)
    monkeypatch.setattr(listener_module, "_START_POLL_INTERVAL_SEC", 0)

    store = CallbackStore()
    listener = CallbackListener(store, auth_token=TOKEN)

    async def hang(self, sockets=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(uvicorn.Server, "serve", hang)
    with pytest.raises(RuntimeError, match="failed to start"):
        await listener.start()
    assert listener._server._task is None
    assert listener._server._socket is None

    monkeypatch.setattr(listener_module, "_START_POLL_ATTEMPTS", 500)
    monkeypatch.setattr(listener_module, "_START_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(uvicorn.Server, "serve", _serve_until_exit)
    url = await listener.start()
    assert url.startswith("http://127.0.0.1:")
    await listener.stop()


async def test_stop_failure_still_resets_and_is_retryable() -> None:
    store = CallbackStore()
    listener = CallbackListener(store, auth_token=TOKEN)
    await listener.start()
    real_task = listener._server._task
    assert real_task is not None
    listener._server._server.should_exit = True
    await real_task
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(RuntimeError("join failed"))
    listener._server._task = failed  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="join failed"):
        await listener.stop()
    assert listener._server._task is None
    assert listener._server._socket is None
    await listener.start()
    await listener.stop()

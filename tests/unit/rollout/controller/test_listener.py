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


async def _passthrough_commit(result: TerminalCallbackResult) -> None:
    """Trivial durable commit: accept the result and keep its own ack."""
    return None


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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    await store.register(ROLLOUT_ID)
    async with await _client(store) as client:
        response = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/completion",
            json={"status": status, "rollout_id": ROLLOUT_ID},
            headers=_auth(),
        )
    assert response.status_code == 422
    assert store._sessions[ROLLOUT_ID].completion is None


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
        assert commits == 0
        # The rejected body left the session live, so a terminal one still wins.
        accepted = await client.post(
            f"/v1/rollouts/{ROLLOUT_ID}/grader",
            json=_grader_body(),
            headers=_auth(),
        )
    assert accepted.status_code == 200
    assert commits == 1


async def test_listener_binds_reserved_localhost_socket() -> None:
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    await store.register(ROLLOUT_ID)
    listener = CallbackListener(store, auth_token=TOKEN)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    with pytest.raises(ValueError, match="auth_token"):
        create_callback_app(store, auth_token="")
    with pytest.raises(ValueError, match="auth_token"):
        CallbackListener(store, auth_token="   ")


async def test_callback_urls_encode_rollout_ids() -> None:
    from urllib.parse import quote

    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
        # Clients append "/chat/completions", so the api_base stops at the id.
        assert listener.chat_completions_url(special_id).endswith(
            f"/v1/rollouts/{encoded}"
        )


async def test_no_docs_or_openapi_surface_is_served() -> None:
    """A tunnel can expose this app to the internet; FastAPI's default
    docs/openapi routes would hand out the route list unauthenticated."""
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    async with await _client(store) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404, path


async def test_fixed_port_binds_exactly_and_conflicts_loudly() -> None:
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    listener = CallbackListener(store, auth_token=TOKEN)
    async with listener:
        port = int(listener.base_url.rsplit(":", 1)[1])
        conflicting = CallbackListener(store, auth_token=TOKEN, port=port)
        with pytest.raises(OSError):
            await conflicting.start()
    fixed = CallbackListener(store, auth_token=TOKEN, port=port)
    async with fixed:
        assert fixed.base_url == f"http://127.0.0.1:{port}"


async def test_advertised_base_url_moves_only_the_chat_url() -> None:
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    listener = CallbackListener(
        store,
        auth_token=TOKEN,
        advertised_base_url="https://fake-name.trycloudflare.com/",
    )
    async with listener:
        assert listener.chat_completions_url("rid") == (
            "https://fake-name.trycloudflare.com/v1/rollouts/rid"
        )
        # Host-process callbacks stay on loopback.
        assert listener.completion_url("rid").startswith(listener.base_url)
        assert listener.grader_url("rid").startswith(listener.base_url)


async def test_advertised_base_url_settable_after_start() -> None:
    """An auto-managed tunnel learns its public URL only after the bind."""
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
    listener = CallbackListener(store, auth_token=TOKEN)
    async with listener:
        assert listener.chat_completions_url("rid").startswith(listener.base_url)
        listener.advertised_base_url = "https://late.trycloudflare.com"
        assert listener.chat_completions_url("rid") == (
            "https://late.trycloudflare.com/v1/rollouts/rid"
        )


async def _serve_until_exit(self, sockets=None):
    self.started = True
    while not self.should_exit:
        await asyncio.sleep(0)


async def test_failed_start_resets_and_is_retryable(monkeypatch) -> None:
    import uvicorn

    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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

    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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
    store = CallbackStore(on_terminal_commit=_passthrough_commit)
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

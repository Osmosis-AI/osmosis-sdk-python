"""Tests for the eval-proxy session client and local contract stub."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from osmosis_ai.rollout.controller.listener import LocalhostUvicornServer
from osmosis_ai.rollout.controller.proxy_client import (
    EVAL_PROXY_INTEGRATION_MODEL,
    EVAL_PROXY_WIRE_MODEL,
    EvalProxyClient,
    EvalProxyError,
    EvalProxySession,
    create_eval_proxy_stub_app,
)

ROLLOUT_ID = "d" * 32
MODEL_PATH = "openai/gpt-4.1-mini"
PLATFORM_TOKEN = "platform-token"


def _platform() -> dict[str, str]:
    return {"Authorization": f"Bearer {PLATFORM_TOKEN}"}


async def _asgi_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://127.0.0.1",
    )


async def test_create_session_binds_rollout_and_model_without_synthetic_model() -> None:
    app = create_eval_proxy_stub_app()
    async with await _asgi_client(app) as client:
        response = await client.post(
            "/v1/eval-sessions",
            json={
                "rollout_id": ROLLOUT_ID,
                "model_path": MODEL_PATH,
                "row_index": 3,
                "run_index": 1,
            },
            headers=_platform(),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["api_base"] == f"/v1/eval-sessions/{ROLLOUT_ID}"
    assert not body["api_base"].endswith("/chat/completions")
    assert "/chat/completions" not in body["api_base"]
    assert body["integration_model"] == EVAL_PROXY_INTEGRATION_MODEL
    assert body["wire_model"] == EVAL_PROXY_WIRE_MODEL
    assert body.get("token")
    assert "model" not in body


async def test_create_session_rejects_client_selected_synthetic_model() -> None:
    app = create_eval_proxy_stub_app()
    async with await _asgi_client(app) as client:
        response = await client.post(
            "/v1/eval-sessions",
            json={
                "rollout_id": ROLLOUT_ID,
                "model_path": MODEL_PATH,
                "model": EVAL_PROXY_WIRE_MODEL,
            },
            headers=_platform(),
        )
    assert response.status_code == 400


async def test_chat_completions_requires_stream_and_wire_model() -> None:
    app = create_eval_proxy_stub_app()
    async with await _asgi_client(app) as client:
        created = await client.post(
            "/v1/eval-sessions",
            json={"rollout_id": ROLLOUT_ID, "model_path": MODEL_PATH},
            headers=_platform(),
        )
        token = created.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        path = f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions"
        missing_stream = await client.post(
            path,
            json={"model": EVAL_PROXY_WIRE_MODEL, "messages": []},
            headers=headers,
        )
        wrong_model = await client.post(
            path,
            json={
                "model": MODEL_PATH,
                "messages": [],
                "stream": True,
            },
            headers=headers,
        )
        bad_usage = await client.post(
            path,
            json={
                "model": EVAL_PROXY_WIRE_MODEL,
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": False},
            },
            headers=headers,
        )
    assert missing_stream.status_code == 400
    assert wrong_model.status_code == 400
    assert bad_usage.status_code == 400


async def test_chat_completions_sse_order_and_optional_stream_options() -> None:
    app = create_eval_proxy_stub_app()
    async with await _asgi_client(app) as client:
        created = await client.post(
            "/v1/eval-sessions",
            json={"rollout_id": ROLLOUT_ID, "model_path": MODEL_PATH},
            headers=_platform(),
        )
        token = created.json()["token"]
        response = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={
                "model": EVAL_PROXY_WIRE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    events = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(item) for item in events[:-1]]
    assert chunks[0]["choices"][0]["delta"].get("content")
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["choices"] == []
    assert "usage" in chunks[2]


async def test_client_create_session_posts_frozen_fields_only() -> None:
    app = create_eval_proxy_stub_app()
    async with LocalhostUvicornServer(app) as server:
        origin = server.base_url
        client = EvalProxyClient(base_url=origin, auth_token="platform-token")
        try:
            session = await client.create_session(
                rollout_id=ROLLOUT_ID,
                model_path=MODEL_PATH,
                row_index=2,
                run_index=0,
            )
        finally:
            await client.aclose()

    recorded = app.state.create_requests[-1]
    assert recorded["rollout_id"] == ROLLOUT_ID
    assert recorded["model_path"] == MODEL_PATH
    assert recorded["row_index"] == 2
    assert recorded["run_index"] == 0
    assert "model" not in recorded
    assert "integration_model" not in recorded
    assert "wire_model" not in recorded
    assert session.api_base == f"/v1/eval-sessions/{ROLLOUT_ID}"
    assert "/chat/completions" not in session.api_base
    assert session.api_base_url == f"{origin}/v1/eval-sessions/{ROLLOUT_ID}"
    assert session.integration_model == EVAL_PROXY_INTEGRATION_MODEL
    assert session.wire_model == EVAL_PROXY_WIRE_MODEL


async def test_client_create_session_surfaces_http_errors() -> None:
    app = create_eval_proxy_stub_app()
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            with pytest.raises(EvalProxyError):
                await client.create_session(rollout_id="", model_path=MODEL_PATH)
        finally:
            await client.aclose()


@pytest.mark.parametrize(
    "content,media_type",
    [
        ("not-json", "text/plain"),  # aiohttp.ContentTypeError
        ("{truncated", "application/json"),  # json.JSONDecodeError
    ],
)
async def test_malformed_2xx_body_raises_eval_proxy_error(
    content: str, media_type: str
) -> None:
    from fastapi import FastAPI
    from fastapi.responses import Response

    app = FastAPI()

    @app.post("/v1/eval-sessions")
    async def create() -> Response:
        return Response(content=content, media_type=media_type)

    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            with pytest.raises(EvalProxyError, match="invalid JSON"):
                await client.create_session(
                    rollout_id=ROLLOUT_ID, model_path=MODEL_PATH
                )
        finally:
            await client.aclose()


def _create_response_app(body: dict) -> object:
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/eval-sessions")
    async def create() -> dict:
        return body

    return app


async def _create_from_body(body: dict):
    app = _create_response_app(body)
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            return await client.create_session(
                rollout_id=ROLLOUT_ID, model_path=MODEL_PATH
            )
        finally:
            await client.aclose()


def _valid_create_body(**overrides: object) -> dict:
    body: dict[str, object] = {
        "rollout_id": ROLLOUT_ID,
        "api_base": f"/v1/eval-sessions/{ROLLOUT_ID}",
        "token": "session-token",
        "integration_model": EVAL_PROXY_INTEGRATION_MODEL,
        "wire_model": EVAL_PROXY_WIRE_MODEL,
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "body",
    [
        _valid_create_body(rollout_id="other-id"),
        _valid_create_body(
            api_base=f"https://evil.example/v1/eval-sessions/{ROLLOUT_ID}"
        ),
        _valid_create_body(api_base=f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions"),
        _valid_create_body(integration_model="openai/gpt-4.1-mini"),
        _valid_create_body(wire_model="gpt-4.1-mini"),
        _valid_create_body(token=""),
        {k: v for k, v in _valid_create_body().items() if k != "token"},
        {k: v for k, v in _valid_create_body().items() if k != "api_base"},
        {k: v for k, v in _valid_create_body().items() if k != "integration_model"},
        {k: v for k, v in _valid_create_body().items() if k != "wire_model"},
        {k: v for k, v in _valid_create_body().items() if k != "rollout_id"},
    ],
)
async def test_create_session_rejects_invalid_contract_fields(body: dict) -> None:
    with pytest.raises(EvalProxyError):
        await _create_from_body(body)


def test_session_repr_hides_token() -> None:
    session = EvalProxySession(
        rollout_id=ROLLOUT_ID,
        model_path=MODEL_PATH,
        api_base=f"/v1/eval-sessions/{ROLLOUT_ID}",
        api_base_url=f"http://proxy/v1/eval-sessions/{ROLLOUT_ID}",
        token="super-secret-session-token",
    )
    assert "super-secret-session-token" not in repr(session)
    assert "super-secret-session-token" not in str(session)


def test_client_requires_non_empty_platform_token() -> None:
    with pytest.raises(ValueError, match="auth_token"):
        EvalProxyClient(base_url="http://proxy", auth_token="")
    with pytest.raises(ValueError, match="auth_token"):
        EvalProxyClient(base_url="http://proxy", auth_token="   ")


def _close_recording_app(create_body: dict) -> tuple[Any, list[str]]:
    from fastapi import FastAPI

    closed: list[str] = []
    app = FastAPI()

    @app.post("/v1/eval-sessions")
    async def create() -> dict:
        return create_body

    @app.delete("/v1/eval-sessions/{rollout_id}")
    async def close(rollout_id: str) -> dict:
        closed.append(rollout_id)
        return {"ok": True}

    return app, closed


async def test_reflected_platform_token_is_rejected_and_session_closed() -> None:
    app, closed = _close_recording_app(_valid_create_body(token=PLATFORM_TOKEN))
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            with pytest.raises(EvalProxyError, match="platform"):
                await client.create_session(
                    rollout_id=ROLLOUT_ID, model_path=MODEL_PATH
                )
        finally:
            await client.aclose()
    assert closed == [ROLLOUT_ID]


async def test_malformed_create_response_triggers_best_effort_close() -> None:
    app, closed = _close_recording_app(
        _valid_create_body(api_base=f"/v1/eval-sessions/{'e' * 32}")
    )
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            with pytest.raises(EvalProxyError, match="api_base"):
                await client.create_session(
                    rollout_id=ROLLOUT_ID, model_path=MODEL_PATH
                )
        finally:
            await client.aclose()
    assert closed == [ROLLOUT_ID]


async def test_cancelled_create_attempts_cleanup_without_masking_cancellation() -> None:
    from fastapi import FastAPI

    closed: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    app = FastAPI()

    @app.post("/v1/eval-sessions")
    async def create() -> dict:
        entered.set()
        await release.wait()
        return _valid_create_body()

    @app.delete("/v1/eval-sessions/{rollout_id}")
    async def close(rollout_id: str) -> dict:
        closed.append(rollout_id)
        return {"ok": True}

    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            task = asyncio.create_task(
                client.create_session(rollout_id=ROLLOUT_ID, model_path=MODEL_PATH)
            )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            for _ in range(100):
                if closed:
                    break
                await asyncio.sleep(0.01)
        finally:
            await client.aclose()
    assert closed == [ROLLOUT_ID]


async def test_management_routes_use_platform_token_not_session_token() -> None:
    app = create_eval_proxy_stub_app()
    async with await _asgi_client(app) as client:
        created = await client.post(
            "/v1/eval-sessions",
            json={"rollout_id": ROLLOUT_ID, "model_path": MODEL_PATH},
            headers=_platform(),
        )
        session_token = created.json()["token"]
        session_headers = {"Authorization": f"Bearer {session_token}"}
        usage_session = await client.get(
            f"/v1/eval-sessions/{ROLLOUT_ID}/usage",
            headers=session_headers,
        )
        close_session = await client.delete(
            f"/v1/eval-sessions/{ROLLOUT_ID}",
            headers=session_headers,
        )
        usage_platform = await client.get(
            f"/v1/eval-sessions/{ROLLOUT_ID}/usage",
            headers=_platform(),
        )
        close_platform = await client.delete(
            f"/v1/eval-sessions/{ROLLOUT_ID}",
            headers=_platform(),
        )
    assert usage_session.status_code == 401
    assert close_session.status_code == 401
    assert usage_platform.status_code == 200
    assert usage_platform.json()["total_tokens"] == 2
    assert close_platform.status_code == 200


async def test_public_get_usage_and_close_session() -> None:
    app = create_eval_proxy_stub_app()
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            session = await client.create_session(
                rollout_id=ROLLOUT_ID, model_path=MODEL_PATH
            )
            usage = await client.get_usage(session.rollout_id)
            assert usage["total_tokens"] == 2
            await client.close_session(session.rollout_id)
            with pytest.raises(EvalProxyError) as exc:
                await client.get_usage(session.rollout_id)
            assert exc.value.status_code == 404
        finally:
            await client.aclose()


async def test_terminal_commit_hook_can_fetch_usage_before_ack() -> None:
    from osmosis_ai.rollout.controller.store import CallbackStore
    from osmosis_ai.rollout.types import (
        GraderCompleteRequest,
        GraderStatus,
        RolloutSample,
    )

    app = create_eval_proxy_stub_app()
    async with LocalhostUvicornServer(app) as server:
        proxy = EvalProxyClient(base_url=server.base_url, auth_token=PLATFORM_TOKEN)
        try:
            await proxy.create_session(rollout_id=ROLLOUT_ID, model_path=MODEL_PATH)

            async def commit(result):
                usage = await proxy.get_usage(result.rollout_id)
                return {"ok": True, "usage": usage}

            store = CallbackStore(on_terminal_commit=commit)
            await store.register(ROLLOUT_ID)
            ack = await store.handle_grader(
                ROLLOUT_ID,
                GraderCompleteRequest(
                    status=GraderStatus.SUCCESS,
                    rollout_id=ROLLOUT_ID,
                    sample=RolloutSample(
                        messages=[{"role": "assistant", "content": "ok"}],
                        reward=1.0,
                    ),
                ),
            )
            assert ack["ok"] is True
            assert ack["usage"]["total_tokens"] == 2
        finally:
            await proxy.aclose()

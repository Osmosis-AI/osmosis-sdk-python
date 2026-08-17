"""Local OpenAI-compatible eval-proxy contract stub. Test support only.

Not a production eval-proxy: it implements just enough of the frozen
create/chat wire contract for SDK tests to exercise it end to end. Lives
under ``tests/`` so the shipped package never carries a fake proxy.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from osmosis_ai.rollout.controller.proxy_client import (
    EVAL_PROXY_INTEGRATION_MODEL,
    EVAL_PROXY_WIRE_MODEL,
)
from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

DEFAULT_STUB_PLATFORM_TOKEN = "platform-token"

_CREATE_FORBIDDEN_FIELDS = frozenset(
    {"model", "integration_model", "wire_model", "synthetic_model"}
)


@dataclass(frozen=True)
class EvalProxyStubUpstream:
    """A real OpenAI-compatible provider behind the contract stub.

    The stub answers with a canned completion by default, which is enough to
    exercise the wire contract but scores zero against any real grader. Pointing
    it at a provider lets a local end-to-end run produce genuine rewards while
    the hosted eval-proxy service is still being built -- the frozen request and
    SSE contract is unchanged, only the response body becomes real.

    ``api_key`` is held in memory and never logged or persisted.
    """

    base_url: str
    api_key: str = field(repr=False)
    timeout_sec: float = 600.0

    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def _upstream_model(model_path: str) -> str:
    """Strip the LiteLLM provider prefix: ``openai/gpt-5-mini`` -> ``gpt-5-mini``."""
    _, separator, remainder = model_path.partition("/")
    return remainder if separator and remainder else model_path


async def _open_upstream_chat(
    *,
    upstream: EvalProxyStubUpstream,
    model_path: str,
    body: Mapping[str, Any],
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Start the upstream request and settle its status before streaming begins.

    The status check has to happen here, not inside the relay generator: once
    ``StreamingResponse`` has sent 200 and its headers, an exception can only
    truncate the body, which the client reports as a transport error instead of
    the provider's actual message.

    ``include_usage`` is forced on so the relayed stream still ends with the
    usage-only chunk the frozen SSE contract requires, whatever the client asked
    for.
    """
    forwarded = dict(body)
    forwarded["model"] = _upstream_model(model_path)
    forwarded["stream"] = True
    forwarded["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {upstream.api_key}"}
    client = httpx.AsyncClient(timeout=upstream.timeout_sec)
    try:
        request = client.build_request(
            "POST", upstream.chat_url(), json=forwarded, headers=headers
        )
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502, detail=f"upstream provider is unreachable: {exc}"
        ) from exc
    except BaseException:
        await client.aclose()
        raise
    if response.status_code >= 400:
        detail = (await response.aread()).decode(errors="replace")[:500]
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"upstream provider returned {response.status_code}: {detail}",
        )
    return client, response


async def _relay_upstream_chat(
    *,
    client: httpx.AsyncClient,
    response: httpx.Response,
) -> AsyncIterator[str]:
    """Relay an already-accepted upstream SSE stream verbatim."""
    try:
        async for line in response.aiter_lines():
            yield f"{line}\n"
    finally:
        await response.aclose()
        await client.aclose()


def create_eval_proxy_stub_app(
    *,
    platform_token: str = DEFAULT_STUB_PLATFORM_TOKEN,
    upstream: EvalProxyStubUpstream | None = None,
) -> FastAPI:
    """Build the contract stub app, isolated per test."""
    app = FastAPI()
    app.state.sessions = {}
    app.state.create_requests = []
    app.state.chat_requests = []
    app.state.platform_token = platform_token

    def _require_platform(authorization: str | None) -> None:
        expected = f"Bearer {platform_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _require_session(rollout_id: str, authorization: str | None) -> dict[str, Any]:
        session = app.state.sessions.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        expected = f"Bearer {session['token']}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return session

    @app.post("/v1/eval-sessions")
    async def create_session(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_platform(authorization)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        forbidden = _CREATE_FORBIDDEN_FIELDS & body.keys()
        if forbidden:
            raise HTTPException(
                status_code=400,
                detail="clients may not select the synthetic model",
            )
        rollout_id = body.get("rollout_id")
        model_path = body.get("model_path")
        if not isinstance(rollout_id, str) or not is_single_path_segment(rollout_id):
            raise HTTPException(status_code=400, detail="invalid rollout_id")
        if not isinstance(model_path, str) or not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")
        token = secrets.token_urlsafe(16)
        app.state.create_requests.append(dict(body))
        app.state.sessions[rollout_id] = {
            "token": token,
            "model_path": model_path,
        }
        return {
            "rollout_id": rollout_id,
            "api_base": f"/v1/eval-sessions/{rollout_id}",
            "token": token,
            "integration_model": EVAL_PROXY_INTEGRATION_MODEL,
            "wire_model": EVAL_PROXY_WIRE_MODEL,
        }

    @app.post("/v1/eval-sessions/{rollout_id}/chat/completions")
    async def chat_completions(
        rollout_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        session = _require_session(rollout_id, authorization)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        app.state.chat_requests.append(
            {"path": str(request.url.path), "body": dict(body)}
        )
        if body.get("stream") is not True:
            raise HTTPException(status_code=400, detail="stream=true is required")
        if body.get("model") != EVAL_PROXY_WIRE_MODEL:
            raise HTTPException(status_code=400, detail="model must be osmosis-rollout")
        stream_options = body.get("stream_options")
        if (
            isinstance(stream_options, dict)
            and "include_usage" in stream_options
            and stream_options.get("include_usage") is not True
        ):
            raise HTTPException(
                status_code=400,
                detail="include_usage must be true when present",
            )

        if upstream is not None:
            client, upstream_response = await _open_upstream_chat(
                upstream=upstream, model_path=session["model_path"], body=body
            )
            return StreamingResponse(
                _relay_upstream_chat(client=client, response=upstream_response),
                media_type="text/event-stream",
            )

        async def events() -> AsyncIterator[str]:
            completion_id = "chatcmpl-stub"
            chunks = [
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                },
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.delete("/v1/eval-sessions/{rollout_id}")
    async def close_session_provisional(
        rollout_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # Provisional path. Not part of the frozen chat/session-create contract.
        _require_platform(authorization)
        session = app.state.sessions.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        app.state.sessions.pop(rollout_id, None)
        return {"ok": True}

    return app

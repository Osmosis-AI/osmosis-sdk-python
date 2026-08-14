"""Eval-proxy session client and local OpenAI-compatible contract stub.

Frozen production contract (Phase 0):
- integration model ``openai/osmosis-rollout``
- wire body model ``osmosis-rollout``
- ``POST /v1/eval-sessions`` bound to ``rollout_id`` + ``model_path``
  (optional ``row_index`` / ``run_index``)
- clients must not select the synthetic model
- session ``api_base`` is ``/v1/eval-sessions/<rollout-id>`` and must not
  include ``/chat/completions``
- chat endpoint ``POST /v1/eval-sessions/<rollout-id>/chat/completions``
- ``stream=true`` is required; ``stream_options`` may be absent; if
  ``include_usage`` is present it must be ``true``
- SSE: content chunk, ``finish_reason="stop"`` chunk, ``choices=[]``
  usage-only chunk, then ``[DONE]``

Close and usage HTTP paths are **not** frozen. ``close_session`` and
``get_usage`` are public logical methods; path construction stays private.
Management-plane create/usage/close uses the platform bearer token. Chat
uses the session bearer token.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from osmosis_ai._imports import raise_optional_dependency_error

try:
    import aiohttp
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval-run",
        feature="Local evaluation",
    )

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import StreamingResponse
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval-run",
        feature="Local evaluation",
    )

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

EVAL_PROXY_INTEGRATION_MODEL = "openai/osmosis-rollout"
EVAL_PROXY_WIRE_MODEL = "osmosis-rollout"

_CREATE_FORBIDDEN_FIELDS = frozenset(
    {"model", "integration_model", "wire_model", "synthetic_model"}
)
_DEFAULT_STUB_PLATFORM_TOKEN = "platform-token"

# Bounded wait for the best-effort close after a failed create; the close
# keeps running in the background if it outlives this window.
_FAILED_CREATE_CLOSE_TIMEOUT_SEC = 10.0


class EvalProxyError(Exception):
    """HTTP or contract failure talking to the eval-proxy.

    This is not an error-taxonomy type. Platform auth/model/budget codes are
    not frozen and must not be inferred here.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EvalProxySession:
    """Session-scoped eval-proxy credentials for one rollout."""

    rollout_id: str
    model_path: str
    api_base: str
    api_base_url: str
    # Bearer credential; excluded from repr so sessions can be logged safely.
    token: str = field(repr=False)
    integration_model: str = EVAL_PROXY_INTEGRATION_MODEL
    wire_model: str = EVAL_PROXY_WIRE_MODEL
    row_index: int | None = None
    run_index: int | None = None


def _require_single_segment_id(rollout_id: str) -> str:
    # quote() cannot make "." or ".." safe: dots are unreserved characters,
    # and HTTP stacks normalize dot segments before the request leaves the
    # client, silently retargeting management calls outside the session path.
    if not is_single_path_segment(rollout_id):
        raise EvalProxyError(
            f"rollout_id must be a single path segment, got {rollout_id!r}"
        )
    return rollout_id


def _session_api_base(rollout_id: str) -> str:
    return f"/v1/eval-sessions/{quote(_require_single_segment_id(rollout_id), safe='')}"


def _consume_best_effort_close(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("best-effort eval-proxy session close failed", exc_info=exc)


class EvalProxyClient:
    """HTTP client for eval-proxy session create (and provisional close/usage)."""

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not auth_token or not auth_token.strip():
            raise ValueError("auth_token (platform management token) must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._session = session
        self._owns_session = session is None

    async def create_session(
        self,
        *,
        rollout_id: str,
        model_path: str,
        row_index: int | None = None,
        run_index: int | None = None,
    ) -> EvalProxySession:
        _require_single_segment_id(rollout_id)
        payload: dict[str, Any] = {
            "rollout_id": rollout_id,
            "model_path": model_path,
        }
        if row_index is not None:
            payload["row_index"] = row_index
        if run_index is not None:
            payload["run_index"] = run_index
        try:
            data = await self._request_json("POST", "/v1/eval-sessions", json=payload)
            return self._session_from_create_response(
                requested_id=rollout_id,
                model_path=model_path,
                data=data,
                row_index=row_index,
                run_index=run_index,
            )
        except BaseException:
            # The create may have reached the server even when the response
            # is invalid or this coroutine is cancelled; close by requested
            # id so a half-created session is not leaked. The original
            # exception always propagates.
            await self._close_after_failed_create(rollout_id)
            raise

    async def _close_after_failed_create(self, rollout_id: str) -> None:
        closer = asyncio.ensure_future(self.close_session(rollout_id))
        closer.add_done_callback(_consume_best_effort_close)
        try:
            await asyncio.wait_for(
                asyncio.shield(closer), _FAILED_CREATE_CLOSE_TIMEOUT_SEC
            )
        except BaseException:
            # Best effort only: the shielded close keeps running even if this
            # wait times out or is cancelled again, and close failures are
            # just logged.
            return

    def _session_from_create_response(
        self,
        *,
        requested_id: str,
        model_path: str,
        data: dict[str, Any],
        row_index: int | None,
        run_index: int | None,
    ) -> EvalProxySession:
        returned_id = data.get("rollout_id")
        if returned_id != requested_id:
            raise EvalProxyError(
                "eval-proxy create response rollout_id does not match the request"
            )
        api_base = data.get("api_base")
        expected_api_base = _session_api_base(requested_id)
        if api_base != expected_api_base:
            raise EvalProxyError(
                "eval-proxy api_base must be exactly "
                f"{expected_api_base} (path only, same origin)"
            )
        api_base = expected_api_base
        integration_model = data.get("integration_model")
        if integration_model != EVAL_PROXY_INTEGRATION_MODEL:
            raise EvalProxyError(
                "eval-proxy create response integration_model must be "
                f"{EVAL_PROXY_INTEGRATION_MODEL}"
            )
        wire_model = data.get("wire_model")
        if wire_model != EVAL_PROXY_WIRE_MODEL:
            raise EvalProxyError(
                f"eval-proxy create response wire_model must be {EVAL_PROXY_WIRE_MODEL}"
            )
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise EvalProxyError(
                "eval-proxy create response token must be a non-empty string"
            )
        if secrets.compare_digest(token.encode(), self._auth_token.encode()):
            raise EvalProxyError(
                "eval-proxy returned the platform management token as the "
                "session token; refusing to hand it to workload code"
            )
        return EvalProxySession(
            rollout_id=requested_id,
            model_path=model_path,
            api_base=api_base,
            api_base_url=f"{self._base_url}{api_base}",
            token=token,
            integration_model=EVAL_PROXY_INTEGRATION_MODEL,
            wire_model=EVAL_PROXY_WIRE_MODEL,
            row_index=row_index,
            run_index=run_index,
        )

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def close_session(self, rollout_id: str) -> None:
        """Close a proxy session.

        Path is provisional: production close is not a frozen contract.
        Uses the platform bearer token, not the session token.
        """
        await self._request_json("DELETE", self._close_path(rollout_id))

    async def get_usage(self, rollout_id: str) -> dict[str, Any]:
        """Fetch session usage.

        Path is provisional: production usage is not a frozen contract.
        Uses the platform bearer token, not the session token.
        """
        return await self._request_json("GET", self._usage_path(rollout_id))

    def _close_path(self, rollout_id: str) -> str:
        return _session_api_base(rollout_id)

    def _usage_path(self, rollout_id: str) -> str:
        return f"{_session_api_base(rollout_id)}/usage"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self._ensure_session()
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        url = f"{self._base_url}{path}"
        async with session.request(method, url, json=json, headers=headers) as response:
            if response.status >= 400:
                detail = await response.text()
                raise EvalProxyError(
                    f"eval-proxy {method} {path} failed: {response.status} {detail}",
                    status_code=response.status,
                )
            if response.status == 204:
                return {}
            try:
                payload = await response.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise EvalProxyError(
                    f"eval-proxy {method} {path} returned invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise EvalProxyError("eval-proxy returned a non-object JSON body")
            return payload


def create_eval_proxy_stub_app(
    *, platform_token: str = _DEFAULT_STUB_PLATFORM_TOKEN
) -> FastAPI:
    """Local contract stub for SDK tests. Not a production eval-proxy."""
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
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return {
            "rollout_id": rollout_id,
            "api_base": _session_api_base(rollout_id),
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
        _require_session(rollout_id, authorization)
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

    @app.get("/v1/eval-sessions/{rollout_id}/usage")
    async def usage_provisional(
        rollout_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # Provisional path. Not part of the frozen chat/session-create contract.
        _require_platform(authorization)
        session = app.state.sessions.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return dict(session["usage"])

    return app

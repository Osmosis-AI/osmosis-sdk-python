"""Local OpenAI-compatible eval-proxy contract stub. Test support only.

Not a production eval-proxy: it implements just enough of the frozen
create/chat wire contract for SDK tests to exercise it end to end. Lives
under ``tests/`` so the shipped package never carries a fake proxy.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from typing import Any

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


def create_eval_proxy_stub_app(
    *, platform_token: str = DEFAULT_STUB_PLATFORM_TOKEN
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

    return app

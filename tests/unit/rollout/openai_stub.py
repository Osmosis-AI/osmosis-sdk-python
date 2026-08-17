"""A minimal OpenAI-compatible chat-completions upstream for bridge tests.

Always replies ``"ok"`` (the echo dataset's ground truth), so an exact-match
grader passes every row. Served on ``LocalhostUvicornServer`` and used as the
LiteLLM bridge's ``api_base`` — tests then exercise the real
bridge -> litellm -> upstream path. A request bearing the
``FORCE_AUTH_ERROR_KEY`` credential is answered 401 so tests can drive the
preflight's fatal path.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

FORCE_AUTH_ERROR_KEY = "force-auth-error"


def create_openai_stub_app() -> FastAPI:
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if request.headers.get("Authorization") == f"Bearer {FORCE_AUTH_ERROR_KEY}":
            return JSONResponse(
                {
                    "error": {
                        "message": "Incorrect API key provided",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
                status_code=401,
            )
        body = await request.json()
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        }

    return app

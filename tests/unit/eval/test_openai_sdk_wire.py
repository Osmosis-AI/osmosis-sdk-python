"""Drive LiteLLM's openai/ provider through the real OpenAI SDK transport.

The other bridge and rubric tests stub LiteLLM's HTTP handler, which the
openai/ provider never uses: it sends through the OpenAI SDK's own client
(httpx on openai 2.x, httpx2 on 3.x). These tests answer from a real localhost
server instead, so a transport change shows up here. CI runs them on both
majors; see the "pytest (openai 3.x)" job.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

pytest.importorskip("litellm")

from osmosis_ai.eval.local.listener import LocalhostUvicornServer
from osmosis_ai.eval.local.llm_bridge import LiteLLMBridge, create_bridge_router
from osmosis_ai.eval.rubric import ProviderRequestError, evaluate_rubric

SECRET = "sk-wire-secret-0123456789"
BEARER = "wire-bearer-0123456789"
CONTENT = '{"score": 0.5, "explanation": "fine"}'
REJECTION = "upstream rejected the request"


def _upstream_app(hits: list[str]) -> FastAPI:
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        hits.append(body["model"])
        if body["model"] == "reject":
            return JSONResponse(
                {"error": {"message": REJECTION, "type": "invalid_request_error"}},
                status_code=400,
            )
        base = {"id": "chatcmpl-wire", "created": 1700000000, "model": body["model"]}
        if body.get("stream"):

            def events() -> Iterator[str]:
                for delta, finish in (
                    ({"role": "assistant", "content": CONTENT}, None),
                    ({}, "stop"),
                ):
                    chunk = {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            **base,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": CONTENT},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    return app


@pytest.fixture
async def upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[str, list[str]]]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    hits: list[str] = []
    async with LocalhostUvicornServer(_upstream_app(hits)) as server:
        # evaluate_rubric takes no api_base; LiteLLM reads it from here.
        monkeypatch.setenv("OPENAI_API_BASE", server.base_url)
        yield server.base_url, hits


async def _post_bridge(base_url: str, model: str, stream: bool) -> httpx.Response:
    bridge = LiteLLMBridge(model=f"openai/{model}", api_base=base_url, api_key=SECRET)
    app = FastAPI()
    app.include_router(create_bridge_router(bridge, auth_token=BEARER))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://bridge"
    ) as client:
        return await client.post(
            "/v1/rollouts/wire/chat/completions",
            headers={"Authorization": f"Bearer {BEARER}"},
            json={"messages": [{"role": "user", "content": "hi"}], "stream": stream},
        )


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_bridge_completes_through_openai_sdk(
    upstream: tuple[str, list[str]], stream: bool
) -> None:
    base_url, hits = upstream
    response = await _post_bridge(base_url, "ok", stream)

    assert response.status_code == 200
    if stream:
        assert response.text.rstrip().endswith("data: [DONE]")
        content = "".join(
            choice["delta"].get("content") or ""
            for payload in _sse_payloads(response.text)
            for choice in payload.get("choices", [])
        )
    else:
        content = response.json()["choices"][0]["message"]["content"]
    assert content == CONTENT
    assert hits == ["ok"]


@pytest.mark.parametrize("stream", [False, True])
async def test_bridge_keeps_provider_rejection_and_hides_credentials(
    upstream: tuple[str, list[str]],
    caplog: pytest.LogCaptureFixture,
    stream: bool,
) -> None:
    base_url, hits = upstream
    response = await _post_bridge(base_url, "reject", stream)

    if stream:
        assert response.status_code == 200
        assert "event: error\n" in response.text
        payload = _sse_payloads(response.text)[0]
    else:
        assert response.status_code == 400
        payload = response.json()
    assert payload["error"]["code"] == 400
    assert REJECTION in payload["error"]["message"]
    assert hits == ["reject"]
    for text in (response.text, caplog.text):
        assert SECRET not in text
        assert BEARER not in text


async def test_rubric_scores_through_openai_sdk(
    upstream: tuple[str, list[str]],
) -> None:
    _, hits = upstream
    result = await evaluate_rubric(
        solution_str="x", rubric="be good", model="openai/ok", api_key=SECRET
    )

    assert result.score == 0.5
    assert hits == ["ok"]


async def test_rubric_maps_provider_rejection(
    upstream: tuple[str, list[str]],
) -> None:
    _, hits = upstream
    with pytest.raises(ProviderRequestError, match=REJECTION) as excinfo:
        await evaluate_rubric(
            solution_str="x", rubric="be good", model="openai/reject", api_key=SECRET
        )

    assert SECRET not in str(excinfo.value)
    assert hits == ["reject"]

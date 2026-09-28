"""In-process LiteLLM bridge for local evaluation.

An OpenAI-compatible chat-completions surface served on a loopback listener, with
LiteLLM doing the provider conversion in-process. Provider credentials stay on
the host; the container only ever sees the loopback URL and a per-run bearer.

The bridge is non-streaming toward the provider. When a client requests
``stream=true`` it gets a valid SSE stream — heartbeat comments while the
completion is in flight, then a single chunk carrying the full delta,
``finish_reason`` and usage, then ``[DONE]``. Per-rollout token totals are
accumulated for the run index and read once with
:meth:`LiteLLMBridge.collect_tokens`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import httpx

from osmosis_ai._imports import raise_optional_dependency_error
from osmosis_ai.eval.local.openai_responses import (
    _compact_json,
    build_responses_kwargs,
    to_chat_response,
)
from osmosis_ai.eval.local.openai_responses import (
    _field as _getattr_or_key,
)
from osmosis_ai.eval.request_policy import (
    BridgeRequestError,
    build_chat_kwargs,
    is_official_openai_endpoint,
)

try:
    from fastapi import APIRouter, Depends, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval",
        feature="Local evaluation",
    )

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

_PREFLIGHT_FATAL_EXCEPTIONS = frozenset(
    {
        "AuthenticationError",
        "BudgetExceededError",
        "NotFoundError",
        "UnsupportedParamsError",
        "APIConnectionError",
        "BadRequestError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
    }
)

_SSE_HEARTBEAT_INTERVAL_SEC = 15.0

# Tunnel-mode keepalive for the non-stream path. Proxies in front of the
# bridge enforce idle-read timeouts (cloudflared quick tunnels: 125s between
# reads from origin), so a silently pending JSON response gets cut off at the
# edge. Within the grace window the response keeps its clean status code;
# past it the bridge commits 200 + application/json and trickles one
# whitespace byte (legal leading JSON) per interval until the real body — or
# an OpenAI-style error object, since the status line is already gone — is
# ready.
_NON_STREAM_GRACE_SEC = 90.0
_NON_STREAM_KEEPALIVE_INTERVAL_SEC = 30.0


def _get_litellm() -> Any:
    try:
        import litellm
    except ModuleNotFoundError as exc:
        raise_optional_dependency_error(
            exc,
            extra="eval",
            feature="Local evaluation",
        )
    litellm.suppress_debug_info = True
    return litellm


def _bridge_error(
    exc: Exception, *, secret_values: tuple[str | None, ...] = ()
) -> tuple[int, dict[str, Any]]:
    """Preserve actionable provider errors without echoing host credentials."""
    from litellm.litellm_core_utils.secret_redaction import redact_string

    status = (
        400
        if isinstance(exc, BridgeRequestError)
        else getattr(exc, "status_code", None)
    )
    message = str(getattr(exc, "message", None) or exc)
    link: BaseException | None = exc
    seen: set[int] = set()
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        if isinstance(link, httpx.HTTPStatusError):
            # LiteLLM can remap a provider 4xx to a different exception/status.
            status = link.response.status_code
            with suppress(httpx.ResponseNotRead):
                message = link.response.text or message
            break
        link = link.__cause__ or link.__context__
    values = {
        value
        for name, value in os.environ.items()
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", name) and len(value) >= 8
    }
    values.update(value for value in secret_values if value)
    for secret in sorted(values, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = " ".join(redact_string(message).split())[:1000]
    agent_status = status if isinstance(status, int) and 400 <= status < 500 else 502
    return agent_status, {
        "error": {
            "message": message,
            "type": "invalid_request_error" if agent_status == 400 else "bridge_error",
            "code": agent_status,
        },
        "detail": message,
    }


def _usage_payload(response: Any) -> dict[str, int] | None:
    usage = _getattr_or_key(response, "usage")
    if usage is None:
        return None
    prompt_tokens = _getattr_or_key(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _getattr_or_key(usage, "input_tokens", 0)
    completion_tokens = _getattr_or_key(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _getattr_or_key(usage, "output_tokens", 0)
    total_tokens = _getattr_or_key(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain_data(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {
            key: _plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _message_fields(choice: Any) -> tuple[str, Any, Any]:
    message = _getattr_or_key(choice, "message")
    if message is None:
        delta = _getattr_or_key(choice, "delta", {})
        return (
            str(_getattr_or_key(delta, "content", "") or ""),
            _getattr_or_key(delta, "tool_calls"),
            _getattr_or_key(delta, "refusal"),
        )
    return (
        str(_getattr_or_key(message, "content", "") or ""),
        _getattr_or_key(message, "tool_calls"),
        _getattr_or_key(message, "refusal"),
    )


def _reasoning_fields(choice: Any) -> dict[str, Any]:
    message = _getattr_or_key(choice, "message") or _getattr_or_key(choice, "delta")
    provider_fields = _getattr_or_key(message, "provider_specific_fields")
    fields: dict[str, Any] = {}
    for name in ("reasoning_content", "thinking_blocks", "reasoning_details"):
        value = _getattr_or_key(message, name)
        if value is None:
            value = _getattr_or_key(provider_fields, name)
        if value is not None:
            fields[name] = _plain_data(value)
    return fields


def _stream_tool_calls_payload(tool_calls: Any) -> list[Any]:
    plain = _plain_data(tool_calls)
    if not isinstance(plain, list):
        plain = [plain]
    normalized: list[Any] = []
    for index, item in enumerate(plain):
        if isinstance(item, dict):
            # Anthropic's content-block index includes earlier thinking blocks.
            normalized.append({**item, "index": index})
        else:
            normalized.append(item)
    return normalized


def _model_response_to_payload(
    response: Any, *, request_model: str, stream: bool
) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    for index, choice in enumerate(_getattr_or_key(response, "choices", []) or []):
        content, tool_calls, refusal = _message_fields(choice)
        finish_reason = _getattr_or_key(choice, "finish_reason", "stop")
        choice_index = int(_getattr_or_key(choice, "index", index) or index)
        if stream:
            delta: dict[str, Any] = {"content": content, **_reasoning_fields(choice)}
            if tool_calls:
                delta["tool_calls"] = _stream_tool_calls_payload(tool_calls)
            if refusal is not None:
                delta["refusal"] = str(refusal)
            choices.append(
                {
                    "index": choice_index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            )
        else:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                **_reasoning_fields(choice),
            }
            if tool_calls:
                message["tool_calls"] = _plain_data(tool_calls)
            if refusal is not None:
                message["refusal"] = str(refusal)
            choices.append(
                {
                    "index": choice_index,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            )

    payload: dict[str, Any] = {
        "id": _getattr_or_key(response, "id", "chatcmpl-eval"),
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(
            _getattr_or_key(response, "created", time.time()) or time.time()
        ),
        "model": request_model,
        "choices": choices,
    }
    usage = _usage_payload(response)
    if usage:
        payload["usage"] = usage
    return payload


class LiteLLMBridge:
    """Convert OpenAI-format chat requests to any litellm provider in-process."""

    def __init__(
        self, *, model: str, api_key: str | None = None, api_base: str | None = None
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._api_base = api_base
        self._tokens: dict[str, int] = {}
        self._reasoning: dict[str, dict[str, dict[str, Any]]] = {}

    def _build_kwargs(self, body: dict[str, Any]) -> dict[str, Any]:
        return build_chat_kwargs(
            body, model=self.model, api_key=self._api_key, api_base=self._api_base
        )

    def _uses_responses_api(self) -> bool:
        return self.model.startswith("openai/") and is_official_openai_endpoint(
            self._api_base
        )

    async def _provider_complete(self, body: dict[str, Any], *, litellm: Any) -> Any:
        if self._uses_responses_api():
            kwargs = build_responses_kwargs(
                body,
                model=self.model,
                api_key=self._api_key,
            )
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = await litellm.aresponses(**kwargs)
            return to_chat_response(response)
        return await litellm.acompletion(**self._build_kwargs(body))

    def _with_reasoning(self, body: dict[str, Any], rollout_id: str) -> dict[str, Any]:
        stored = self._reasoning.get(rollout_id)
        messages = body.get("messages")
        if not stored or not isinstance(messages, list):
            return body
        replayed = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    fields = stored.get(_getattr_or_key(call, "id") or "")
                    if fields:
                        message = {**fields, **message}
                        break
            replayed.append(message)
        return {**body, "messages": replayed}

    def _remember_reasoning(self, response: Any, rollout_id: str) -> None:
        for choice in _getattr_or_key(response, "choices", []) or []:
            fields = _reasoning_fields(choice)
            _, tool_calls, _ = _message_fields(choice)
            if not fields or not tool_calls:
                continue
            stored = self._reasoning.setdefault(rollout_id, {})
            for call in tool_calls:
                call_id = _getattr_or_key(call, "id")
                if call_id:
                    stored[call_id] = fields

    async def preflight_check(self) -> None:
        """One-shot completion probe; raises on persistent config problems."""
        litellm = _get_litellm()
        try:
            litellm.get_llm_provider(model=self.model, api_base=self._api_base)
        except Exception as exc:
            _, error = _bridge_error(exc, secret_values=(self._api_key,))
            raise RuntimeError(
                "Invalid LiteLLM model format. Use 'provider/model' "
                "(e.g. openai/gpt-5-mini, anthropic/claude-sonnet-4-6). "
                f"Received: {self.model!r}. Details: {error['detail']}"
            ) from exc
        try:
            await self._provider_complete(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 256,
                },
                litellm=litellm,
            )
        except Exception as exc:
            ename = type(exc).__name__
            if ename == "RateLimitError":
                return
            if ename in _PREFLIGHT_FATAL_EXCEPTIONS:
                raise
            _, error = _bridge_error(exc, secret_values=(self._api_key,))
            logger.warning("Preflight non-fatal error: %s", error["detail"])

    async def complete(self, body: dict[str, Any], *, rollout_id: str) -> Any:
        litellm = _get_litellm()
        response = await self._provider_complete(
            self._with_reasoning(body, rollout_id), litellm=litellm
        )
        self._remember_reasoning(response, rollout_id)
        usage = _usage_payload(response)
        if usage:
            self._tokens[rollout_id] = (
                self._tokens.get(rollout_id, 0) + usage["total_tokens"]
            )
        return response

    def collect_tokens(self, rollout_id: str) -> int | None:
        """Pop the rollout's token total; None if the bridge never served it."""
        self._reasoning.pop(rollout_id, None)
        return self._tokens.pop(rollout_id, None)

    def discard(self, rollout_id: str) -> None:
        self._tokens.pop(rollout_id, None)
        self._reasoning.pop(rollout_id, None)


async def _trickle_json_response(
    task: asyncio.Task[Any],
    *,
    request_model: str,
    secret_values: tuple[str | None, ...] = (),
) -> AsyncIterator[str]:
    """Body of an already-committed 200 while the provider is still in flight.

    A late upstream error can no longer change the status code, so it becomes
    an OpenAI-style ``{"error": ...}`` object — the client surfaces it as a
    parse/validation failure instead of a clean 502, which is the accepted
    tunnel-mode trade-off.
    """
    try:
        # The committed status line alone does not reset proxy idle-read
        # timers; only body bytes do. Send one immediately so the gap between
        # reads never exceeds the keepalive interval itself (the grace window
        # already spent most of the edge's budget).
        yield " "
        while True:
            # asyncio.wait, never wait_for: a provider that raises
            # TimeoutError (== asyncio.TimeoutError since 3.11) must land in
            # the terminal-error branch, not be mistaken for the keepalive
            # timer — wait_for would spin on the completed task forever.
            done, _ = await asyncio.wait(
                {task}, timeout=_NON_STREAM_KEEPALIVE_INTERVAL_SEC
            )
            if not done:
                yield " "
                continue
            try:
                response = task.result()
            except Exception as exc:
                _, error = _bridge_error(exc, secret_values=secret_values)
                logger.warning("LLM bridge error: %s", error["detail"])
                yield _compact_json(error)
                return
            yield _compact_json(
                _model_response_to_payload(
                    response, request_model=request_model, stream=False
                )
            )
            return
    finally:
        await _cancel_task(task)


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel and consume a provider task without masking response cleanup."""
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


def create_bridge_router(
    bridge: LiteLLMBridge, *, auth_token: str, non_stream_keepalive: bool = False
) -> APIRouter:
    """Chat-completions routes for :class:`LiteLLMBridge`.

    ``non_stream_keepalive`` turns on the tunnel-mode keepalive for
    non-streaming calls; pure-loopback runs leave it off and keep exact
    current behavior.
    """
    if not auth_token or not auth_token.strip():
        raise ValueError("bridge auth_token must be a non-empty string")

    router = APIRouter()
    secret_values = (bridge._api_key, auth_token)

    async def require_auth(request: Request) -> None:
        header = request.headers.get("Authorization")
        scheme, _, credentials = (header or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credentials.encode(), auth_token.encode()
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    async def _serve_completion(
        body: dict[str, Any], rollout_id: str
    ) -> dict[str, Any]:
        response = await bridge.complete(body, rollout_id=rollout_id)
        # The bridge is non-streaming, so a healthy response always has >=1
        # choice; an empty stream wrapper would silently emit an empty delta.
        if not (_getattr_or_key(response, "choices", []) or []):
            raise RuntimeError(
                "LLM response carried no choices; refusing to emit an empty completion"
            )
        return response

    @router.post(
        "/v1/rollouts/{rollout_id}/chat/completions",
        dependencies=[Depends(require_auth)],
    )
    async def chat_completions(rollout_id: str, request: Request) -> Any:
        if not is_single_path_segment(rollout_id):
            raise HTTPException(status_code=422, detail="invalid rollout_id")
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid body")

        is_stream = bool(body.get("stream", False))
        request_model = str(body.get("model") or bridge.model)

        if not is_stream:
            task = asyncio.create_task(_serve_completion(body, rollout_id))
            try:
                # asyncio.wait, never wait_for: a provider raising
                # TimeoutError must be a clean 502 below, not read as the
                # grace timer expiring. Without the keepalive there is no
                # grace timer at all, so the wait is unbounded.
                done, _ = await asyncio.wait(
                    {task},
                    timeout=_NON_STREAM_GRACE_SEC if non_stream_keepalive else None,
                )
            except asyncio.CancelledError:
                await _cancel_task(task)
                raise
            if not done:
                # Still in flight: commit 200 now and keep the connection warm.
                return StreamingResponse(
                    _trickle_json_response(
                        task, request_model=request_model, secret_values=secret_values
                    ),
                    media_type="application/json",
                    background=BackgroundTask(_cancel_task, task),
                )
            try:
                response = task.result()
            except Exception as exc:
                # No byte committed yet, so errors keep their clean status code.
                status, error = _bridge_error(exc, secret_values=secret_values)
                logger.warning("LLM bridge error: %s", error["detail"])
                return JSONResponse(error, status_code=status)
            return JSONResponse(
                _model_response_to_payload(
                    response, request_model=request_model, stream=False
                )
            )

        async def stream_events() -> AsyncIterator[str]:
            yield ": ping\n\n"
            task = asyncio.create_task(_serve_completion(body, rollout_id))
            try:
                while not task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=_SSE_HEARTBEAT_INTERVAL_SEC,
                        )
                    except TimeoutError:
                        yield ": ping\n\n"
                    except Exception:
                        break
                try:
                    response = await task
                except Exception as exc:
                    _, error = _bridge_error(exc, secret_values=secret_values)
                    logger.warning("LLM bridge error: %s", error["detail"])
                    yield f"event: error\ndata: {_compact_json(error)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                payload = _model_response_to_payload(
                    response, request_model=request_model, stream=True
                )
                yield f"data: {_compact_json(payload)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await _cancel_task(task)

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


__all__ = [
    "LiteLLMBridge",
    "create_bridge_router",
]

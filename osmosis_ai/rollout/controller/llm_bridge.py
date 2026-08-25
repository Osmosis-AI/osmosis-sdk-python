"""In-process LiteLLM bridge: the model-call half of the local eval controller.

Port of the monolith eval controller's ``LiteLLMBridge``: an OpenAI-compatible
chat-completions surface served on the controller's loopback listener, with
LiteLLM doing the provider conversion in-process. Provider credentials stay on
the host; the container only ever sees the loopback URL and a per-run bearer.

The bridge is non-streaming toward the provider. When a client requests
``stream=true`` it gets a valid SSE stream — heartbeat comments while the
completion is in flight, then a single chunk carrying the full delta,
``finish_reason`` and usage, then ``[DONE]`` — the same shape the hosted eval
controller serves. Per-rollout token totals are accumulated for the run index
and read once with :meth:`LiteLLMBridge.collect_tokens`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from osmosis_ai._imports import raise_optional_dependency_error
from osmosis_ai.rollout.controller.openai_responses import (
    build_responses_kwargs,
    to_chat_response,
)

try:
    from fastapi import APIRouter, Depends, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval",
        feature="Local evaluation",
    )

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

# Request fields forwarded to litellm; everything else the client sends is
# dropped. LiteLLM handles known provider incompatibilities, while the Responses
# path below learns from a provider rejection when its model metadata is stale.
_LITELLM_CONSUMED_FIELDS = frozenset(
    {"messages", "temperature", "top_p", "max_tokens", "tools"}
)
_OPENAI_SAMPLING_FIELDS = frozenset({"temperature", "top_p"})

# The preflight probes once, so a 4xx client error there is a persistent
# config/account problem, not a per-row fluke. Mid-run errors are not
# classified here: they fail their row and surface through the callbacks.
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


def _getattr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _unsupported_openai_sampling_param(exc: Exception, *, litellm: Any) -> str | None:
    """Return a rejected sampling parameter from a structured OpenAI 400."""
    bad_request_error = getattr(litellm, "BadRequestError", None)
    if not isinstance(bad_request_error, type) or not isinstance(
        exc, bad_request_error
    ):
        return None
    if (
        getattr(exc, "status_code", None) != 400
        or getattr(exc, "llm_provider", None) != "openai"
    ):
        return None

    body = getattr(exc, "body", None)
    error = _getattr_or_key(body, "error", body)
    if not isinstance(error, dict):
        # LiteLLM's async Responses wrapper can remap the provider exception and
        # retain the structured OpenAI error only as JSON after this marker.
        raw_message = getattr(exc, "message", str(exc))
        _, marker, encoded_error = raw_message.partition("OpenAIException - ")
        if marker:
            try:
                decoded_error = json.loads(encoded_error)
            except json.JSONDecodeError:
                pass
            else:
                error = _getattr_or_key(decoded_error, "error", decoded_error)
    message = _getattr_or_key(error, "message", "")
    error_type = getattr(exc, "type", None) or _getattr_or_key(error, "type")
    param = getattr(exc, "param", None) or _getattr_or_key(error, "param")
    if (
        not isinstance(message, str)
        or "unsupported parameter" not in message.lower()
        or error_type != "invalid_request_error"
        or param not in _OPENAI_SAMPLING_FIELDS
    ):
        return None
    return param


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


def _stream_tool_calls_payload(tool_calls: Any) -> list[Any]:
    plain = _plain_data(tool_calls)
    if not isinstance(plain, list):
        plain = [plain]
    normalized: list[Any] = []
    for index, item in enumerate(plain):
        if isinstance(item, dict) and "index" not in item:
            normalized.append({"index": index, **item})
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
            delta: dict[str, Any] = {"content": content}
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
            message: dict[str, Any] = {"role": "assistant", "content": content}
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


def _compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


class LiteLLMBridge:
    """Convert OpenAI-format chat requests to any litellm provider in-process."""

    def __init__(
        self, *, model: str, api_key: str | None = None, api_base: str | None = None
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._api_base = api_base
        self._tokens: dict[str, int] = {}
        self._rejected_openai_sampling_params: set[str] = set()

    def _build_kwargs(self, body: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": body.get("messages", []),
            "drop_params": True,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["base_url"] = self._api_base
        for key in _LITELLM_CONSUMED_FIELDS - {"messages"}:
            if key in body:
                kwargs[key] = body[key]
        return kwargs

    def _uses_responses_api(self, litellm: Any) -> bool:
        if not self.model.startswith("openai/"):
            return False
        return not (
            self._api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or getattr(litellm, "api_base", None)
        )

    def _filter_responses_sampling_params(
        self, body: dict[str, Any], *, litellm: Any
    ) -> dict[str, Any]:
        requested = {key: body[key] for key in _OPENAI_SAMPLING_FIELDS if key in body}
        if not requested:
            return body

        supported = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="openai",
            drop_params=True,
            **requested,
        )
        unsupported = (requested.keys() - supported.keys()) | (
            requested.keys() & self._rejected_openai_sampling_params
        )
        if not unsupported:
            return body

        filtered = dict(body)
        for key in unsupported:
            filtered.pop(key, None)
        return filtered

    async def _provider_complete(self, body: dict[str, Any], *, litellm: Any) -> Any:
        if self._uses_responses_api(litellm):
            kwargs = build_responses_kwargs(
                self._filter_responses_sampling_params(body, litellm=litellm),
                model=self.model,
                api_key=self._api_key,
            )
            # Each retry removes one of the two whitelisted sampling fields.
            # Repeated or unrelated bad requests are re-raised immediately.
            while True:
                try:
                    response = await litellm.aresponses(**kwargs)
                    return to_chat_response(response)
                except Exception as exc:
                    param = _unsupported_openai_sampling_param(exc, litellm=litellm)
                    if param is None or param not in kwargs:
                        raise
                    self._rejected_openai_sampling_params.add(param)
                    kwargs.pop(param)
                    logger.info(
                        "Retrying model %s without unsupported parameter %s",
                        self.model,
                        param,
                    )
        return await litellm.acompletion(**self._build_kwargs(body))

    async def preflight_check(self) -> None:
        """One-shot completion probe; raises on persistent config problems."""
        litellm = _get_litellm()
        try:
            litellm.get_llm_provider(model=self.model, api_base=self._api_base)
        except Exception as exc:
            msg = getattr(exc, "message", str(exc))
            raise RuntimeError(
                "Invalid LiteLLM model format. Use 'provider/model' "
                "(e.g. openai/gpt-5-mini, anthropic/claude-sonnet-4-6). "
                f"Received: {self.model!r}. Details: {msg}"
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
            logger.warning("Preflight non-fatal error: %s", exc)

    async def complete(self, body: dict[str, Any], *, rollout_id: str) -> Any:
        litellm = _get_litellm()
        response = await self._provider_complete(body, litellm=litellm)
        usage = _usage_payload(response)
        if usage:
            self._tokens[rollout_id] = (
                self._tokens.get(rollout_id, 0) + usage["total_tokens"]
            )
        return response

    def collect_tokens(self, rollout_id: str) -> int | None:
        """Pop the rollout's token total; None if the bridge never served it."""
        return self._tokens.pop(rollout_id, None)

    def discard(self, rollout_id: str) -> None:
        self._tokens.pop(rollout_id, None)


async def _trickle_json_response(
    task: asyncio.Task[Any], *, request_model: str
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
                logger.warning("LLM bridge error: %s", exc)
                yield _compact_json(
                    {
                        "error": {
                            "message": f"{type(exc).__name__}: {exc}",
                            "type": "bridge_error",
                        }
                    }
                )
                return
            yield _compact_json(
                _model_response_to_payload(
                    response, request_model=request_model, stream=False
                )
            )
            return
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def create_bridge_router(
    bridge: LiteLLMBridge, *, auth_token: str, non_stream_keepalive: bool = False
) -> APIRouter:
    """Chat-completions routes for :class:`LiteLLMBridge`.

    Mounted on the controller's callback app; guarded by its own bearer so the
    container-held credential cannot drive the callback surface.
    ``non_stream_keepalive`` turns on the tunnel-mode keepalive for
    non-streaming calls; pure-loopback runs leave it off and keep exact
    current behavior.
    """
    if not auth_token or not auth_token.strip():
        raise ValueError("bridge auth_token must be a non-empty string")

    router = APIRouter()

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
            if not non_stream_keepalive:
                try:
                    response = await _serve_completion(body, rollout_id)
                except Exception as exc:
                    logger.warning("LLM bridge error: %s", exc)
                    return JSONResponse({"detail": str(exc)}, status_code=502)
                return JSONResponse(
                    _model_response_to_payload(
                        response, request_model=request_model, stream=False
                    )
                )
            task = asyncio.create_task(_serve_completion(body, rollout_id))
            try:
                # asyncio.wait, never wait_for: a provider raising
                # TimeoutError must be a clean 502 below, not read as the
                # grace timer expiring.
                done, _ = await asyncio.wait({task}, timeout=_NON_STREAM_GRACE_SEC)
            except asyncio.CancelledError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise
            if not done:
                # Still in flight: commit 200 now and keep the connection warm.
                return StreamingResponse(
                    _trickle_json_response(task, request_model=request_model),
                    media_type="application/json",
                )
            try:
                response = task.result()
            except Exception as exc:
                # Within the grace window errors keep their clean status code.
                logger.warning("LLM bridge error: %s", exc)
                return JSONResponse({"detail": str(exc)}, status_code=502)
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
                    logger.warning("LLM bridge error: %s", exc)
                    yield (
                        f"event: error\ndata: {_compact_json({'error': str(exc)})}\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return
                payload = _model_response_to_payload(
                    response, request_model=request_model, stream=True
                )
                yield f"data: {_compact_json(payload)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

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

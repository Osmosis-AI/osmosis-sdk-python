"""FastAPI routes for the native Harbor protocol translation gateway."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from osmosis_ai.rollout.backend.native_harbor.gateway import (
    NativeHarborGatewayError,
    NativeHarborGatewayRoute,
    NativeHarborTranslationGateway,
)

logger: logging.Logger = logging.getLogger(__name__)

_Protocol = Literal["anthropic", "openai"]


def _error_status(exc: Exception) -> int:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code <= 599:
        return status_code
    return 502


def _error_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    return str(message if message is not None else exc) or "Gateway request failed"


def _error_payload(
    protocol: _Protocol,
    message: str,
    status_code: int,
) -> dict[str, Any]:
    if protocol == "anthropic":
        error_type = (
            "authentication_error"
            if status_code == 401
            else "invalid_request_error"
            if status_code < 500
            else "api_error"
        )
        return {
            "type": "error",
            "error": {"type": error_type, "message": message},
        }
    return {
        "error": {
            "message": message,
            "type": "authentication_error"
            if status_code == 401
            else "invalid_request_error"
            if status_code < 500
            else "api_error",
            "param": None,
            "code": str(status_code),
        }
    }


def _json_content(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Gateway translator returned unsupported response {type(value)!r}")


async def _request_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise NativeHarborGatewayError(
            "request body must be JSON",
            status_code=400,
        ) from exc
    if not isinstance(body, dict):
        raise NativeHarborGatewayError(
            "request body must be a JSON object",
            status_code=400,
        )
    return body


def _response_event_json(event: Any) -> str:
    if isinstance(event, BaseModel):
        return event.model_dump_json(exclude_none=True, exclude_unset=True)
    if isinstance(event, Mapping):
        return json.dumps(dict(event), separators=(",", ":"))
    if isinstance(event, bytes):
        return event.decode("utf-8")
    if isinstance(event, str):
        return event
    raise TypeError(f"Unsupported Responses stream event {type(event)!r}")


async def _anthropic_stream(response: Any) -> AsyncIterator[bytes | str]:
    try:
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                yield chunk
        else:
            for chunk in response:
                yield chunk
    except Exception as exc:
        logger.exception("Native Harbor Anthropic gateway stream failed")
        payload = _error_payload("anthropic", _error_message(exc), _error_status(exc))
        yield f"event: error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _responses_stream(response: Any) -> AsyncIterator[str]:
    try:
        if hasattr(response, "__aiter__"):
            async for event in response:
                yield f"data: {_response_event_json(event)}\n\n"
        else:
            for event in response:
                yield f"data: {_response_event_json(event)}\n\n"
    except Exception as exc:
        logger.exception("Native Harbor Responses gateway stream failed")
        payload = _error_payload("openai", _error_message(exc), _error_status(exc))
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _error_response(protocol: _Protocol, exc: Exception) -> JSONResponse:
    status_code = _error_status(exc)
    if status_code >= 500:
        logger.exception("Native Harbor %s gateway request failed", protocol)
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(protocol, _error_message(exc), status_code),
    )


def _resolve_route(
    gateway: NativeHarborTranslationGateway,
    request: Request,
) -> NativeHarborGatewayRoute:
    return gateway.resolve_headers(request.headers)


def install_native_harbor_gateway_routes(
    app: FastAPI,
    gateway: NativeHarborTranslationGateway,
) -> None:
    """Install the fixed Messages and Responses endpoints on a rollout app."""

    @app.post("/v1/messages", include_in_schema=False)
    async def anthropic_messages(request: Request) -> Any:
        try:
            route = _resolve_route(gateway, request)
            body = await _request_body(request)
            stream = body.get("stream") is True
            response = await gateway.anthropic_messages(body, route)
            if stream:
                return StreamingResponse(
                    _anthropic_stream(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return JSONResponse(content=_json_content(response))
        except Exception as exc:
            return _error_response("anthropic", exc)

    @app.post("/v1/responses", include_in_schema=False)
    async def openai_responses(request: Request) -> Any:
        try:
            route = _resolve_route(gateway, request)
            body = await _request_body(request)
            stream = body.get("stream") is True
            response = await gateway.openai_responses(body, route)
            if stream:
                return StreamingResponse(
                    _responses_stream(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return JSONResponse(content=_json_content(response))
        except Exception as exc:
            return _error_response("openai", exc)


__all__ = ["install_native_harbor_gateway_routes"]

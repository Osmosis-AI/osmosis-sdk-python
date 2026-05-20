from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from osmosis_ai.eval.controller.litellm_bridge import (
    build_sse_error_event,
    compact_json,
    model_response_to_payload,
)
from osmosis_ai.eval.controller.messages import preprocess_controller_messages

if TYPE_CHECKING:
    from osmosis_ai.eval.controller.state import ControllerRolloutState

VALID_MULTI_TURN_MODES = frozenset({"multi_sample", "single_sample"})
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0


def _heartbeat_interval() -> float:
    try:
        interval = float(os.environ.get("SSE_HEARTBEAT_INTERVAL_S", ""))
    except ValueError:
        return DEFAULT_HEARTBEAT_INTERVAL_S
    if interval <= 0:
        return DEFAULT_HEARTBEAT_INTERVAL_S
    return interval


class EvalControllerServer:
    def __init__(self, *, api_key: str, bridge: Any) -> None:
        self.api_key = api_key
        self.bridge = bridge
        self._states: dict[str, ControllerRolloutState] = {}
        self.app: FastAPI = self._build_app()

    def register_rollout_state(self, state: ControllerRolloutState) -> None:
        self._states[state.rollout_id] = state

    def pop_rollout_state(self, rollout_id: str) -> ControllerRolloutState | None:
        return self._states.pop(rollout_id, None)

    def get_rollout_state(self, rollout_id: str) -> ControllerRolloutState | None:
        return self._states.get(rollout_id)

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="EvalControllerServer", docs_url=None, redoc_url=None)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/chat/completions")
        async def chat_completions(request: Request) -> Any:
            body = await self._json_object_body(request)
            self._require_auth(request)
            rollout_id, sample_id = self._required_chat_headers(request)
            mode = self._multi_turn_mode(body)
            self._chat_messages(body)
            stream = bool(body.get("stream", True))
            if stream:
                return StreamingResponse(
                    self._stream_completion(rollout_id, sample_id, mode, body),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return await self._json_completion(rollout_id, sample_id, mode, body)

        @app.post("/v1/rollout/completed")
        async def rollout_completed(raw: Request) -> dict[str, str]:
            self._require_auth(raw)
            request = await self._rollout_callback_body(raw)
            state = self.get_rollout_state(request.rollout_id)
            if state is not None:
                state.mark_rollout_completed(request)
            return {"status": "ok"}

        @app.post("/v1/grader/completed")
        async def grader_completed(raw: Request) -> dict[str, str]:
            self._require_auth(raw)
            request = await self._grader_callback_body(raw)
            state = self.get_rollout_state(request.rollout_id)
            if state is not None:
                state.mark_grader_completed(request)
            return {"status": "ok"}

        return app

    async def _json_object_body(self, request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="request body must be a JSON object"
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail="request body must be a JSON object"
            )
        return body

    async def _rollout_callback_body(self, request: Request) -> Any:
        from osmosis_ai.rollout.types import RolloutCompleteRequest

        try:
            return RolloutCompleteRequest.model_validate(
                await self._json_object_body(request)
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    async def _grader_callback_body(self, request: Request) -> Any:
        from osmosis_ai.rollout.types import GraderCompleteRequest

        try:
            return GraderCompleteRequest.model_validate(
                await self._json_object_body(request)
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    def _require_auth(self, request: Request) -> None:
        expected = f"Bearer {self.api_key}"
        if request.headers.get("authorization") != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    def _required_chat_headers(self, request: Request) -> tuple[str, str]:
        missing = [
            name
            for name in ("x-rollout-id", "x-sample-id")
            if not request.headers.get(name)
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"missing required header(s): {', '.join(missing)}",
            )
        return (
            str(request.headers["x-rollout-id"]),
            str(request.headers["x-sample-id"]),
        )

    def _multi_turn_mode(self, body: dict[str, Any]) -> str:
        mode = str(body.get("multi_turn_mode", "multi_sample"))
        if mode not in VALID_MULTI_TURN_MODES:
            raise HTTPException(
                status_code=400,
                detail=("multi_turn_mode must be one of: multi_sample, single_sample"),
            )
        return mode

    def _chat_messages(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise HTTPException(
                status_code=400, detail="messages must be a list of objects"
            )
        return messages

    def _register_request(
        self,
        state: ControllerRolloutState,
        *,
        sample_id: str,
        mode: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        messages = preprocess_controller_messages(self._chat_messages(body))
        if mode == "single_sample" and sample_id in state.completed_sample_ids:
            tools = state.register_chat_reuse_branch(
                sample_id, mode, messages, body.get("tools")
            )
        else:
            tools = state.register_chat_create_branch(
                sample_id, mode, messages, body.get("tools")
            )

        prepared = dict(body)
        prepared["messages"] = messages
        if "tools" in prepared or tools:
            prepared["tools"] = tools
        return prepared

    async def _complete(
        self,
        rollout_id: str,
        sample_id: str,
        mode: str,
        body: dict[str, Any],
    ) -> tuple[ControllerRolloutState, dict[str, Any]]:
        state = self.get_rollout_state(rollout_id)
        if state is None:
            raise HTTPException(status_code=500, detail="rollout not found")

        prepared = self._register_request(
            state, sample_id=sample_id, mode=mode, body=body
        )
        response = await self.bridge.complete(rollout_id, sample_id, prepared)
        payload = model_response_to_payload(
            response,
            request_model=str(body.get("model", "osmosis-rollout")),
            stream=bool(body.get("stream", True)),
        )
        tokens = 0
        usage = payload.get("usage")
        if isinstance(usage, dict):
            tokens = int(usage.get("total_tokens", 0) or 0)
        state.mark_chat_completion(sample_id, tokens=tokens)
        return state, payload

    async def _json_completion(
        self, rollout_id: str, sample_id: str, mode: str, body: dict[str, Any]
    ) -> JSONResponse:
        try:
            _, payload = await self._complete(rollout_id, sample_id, mode, body)
        except HTTPException:
            raise
        except Exception as exc:
            self._mark_controller_error(rollout_id, str(exc))
            return JSONResponse(status_code=502, content={"error": str(exc)})
        return JSONResponse(content=payload)

    async def _stream_completion(
        self, rollout_id: str, sample_id: str, mode: str, body: dict[str, Any]
    ) -> AsyncIterator[str]:
        yield ": ping\n\n"
        task = asyncio.create_task(self._complete(rollout_id, sample_id, mode, body))
        try:
            interval = _heartbeat_interval()
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                except TimeoutError:
                    yield ": ping\n\n"
                except Exception:
                    break

            try:
                _, payload = await task
            except Exception as exc:
                self._mark_controller_error(rollout_id, str(exc))
                if isinstance(exc, HTTPException) and exc.detail == "rollout not found":
                    yield build_sse_error_event("rollout not found")
                else:
                    yield build_sse_error_event(str(exc))
                yield "data: [DONE]\n\n"
                return

            yield f"data: {compact_json(payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def _mark_controller_error(self, rollout_id: str, fallback_message: str) -> None:
        state = self.get_rollout_state(rollout_id)
        if state is None:
            return
        message = fallback_message
        collect = getattr(self.bridge, "collect_systemic_error", None)
        if callable(collect):
            collected = collect(rollout_id)
            if isinstance(collected, str) and collected:
                message = collected
        state.mark_controller_error(message)


__all__ = ["EvalControllerServer"]

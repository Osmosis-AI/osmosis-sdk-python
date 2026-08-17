"""Eval-proxy session client.

Frozen production contract:
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

The close HTTP path is **not** frozen: ``close_session`` is a public logical
method and path construction stays private. Management-plane create/close
uses the platform bearer token. Chat uses the session bearer token.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

EVAL_PROXY_INTEGRATION_MODEL = "openai/osmosis-rollout"
EVAL_PROXY_WIRE_MODEL = "osmosis-rollout"

# Bounded wait for the best-effort close after a failed create; the close
# keeps running in the background if it outlives this window.
_FAILED_CREATE_CLOSE_TIMEOUT_SEC = 10.0

# Total timeout for management calls: these are small JSON round-trips, so
# they must not hang a rollout, but the bound still tolerates a cold proxy.
_MANAGEMENT_REQUEST_TIMEOUT_SEC = 30.0


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
    """HTTP client for eval-proxy session create (and provisional close)."""

    def __init__(self, *, base_url: str, auth_token: str) -> None:
        if not auth_token or not auth_token.strip():
            raise ValueError("auth_token (platform management token) must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._http = httpx.AsyncClient(timeout=_MANAGEMENT_REQUEST_TIMEOUT_SEC)

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
        except (Exception, asyncio.CancelledError) as exc:
            # The create may have reached the server even when the response
            # is invalid or this coroutine is cancelled; close by requested
            # id so a half-created session is not leaked. The original
            # exception always propagates. Process-level exits
            # (KeyboardInterrupt, SystemExit, GeneratorExit) skip the network
            # cleanup: awaiting here would delay shutdown, and awaiting during
            # GeneratorExit is illegal; the eval-proxy session TTL reaps the
            # leftover. A definitive 4xx is the one rejection we must not
            # clean up after: nothing was created, and on a 409 the id belongs
            # to another run whose live session the DELETE would destroy.
            status = exc.status_code if isinstance(exc, EvalProxyError) else None
            if status is None or not 400 <= status < 500:
                await self._close_after_failed_create(rollout_id)
            raise

    async def _close_after_failed_create(self, rollout_id: str) -> None:
        closer = asyncio.ensure_future(self.close_session(rollout_id))
        closer.add_done_callback(_consume_best_effort_close)
        try:
            await asyncio.wait_for(
                asyncio.shield(closer), _FAILED_CREATE_CLOSE_TIMEOUT_SEC
            )
        except (Exception, asyncio.CancelledError):
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
        await self._http.aclose()

    async def close_session(self, rollout_id: str) -> None:
        """Close a proxy session.

        Path is provisional: production close is not a frozen contract.
        Uses the platform bearer token, not the session token.
        """
        await self._request_json("DELETE", self._close_path(rollout_id))

    def _close_path(self, rollout_id: str) -> str:
        return _session_api_base(rollout_id)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._http.request(
            method,
            f"{self._base_url}{path}",
            json=json,
            headers={"Authorization": f"Bearer {self._auth_token}"},
        )
        if not 200 <= response.status_code < 300:
            raise EvalProxyError(
                f"eval-proxy {method} {path} failed: "
                f"{response.status_code} {response.text}",
                status_code=response.status_code,
            )
        if response.status_code == 204:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise EvalProxyError(
                f"eval-proxy {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise EvalProxyError("eval-proxy returned a non-object JSON body")
        return payload

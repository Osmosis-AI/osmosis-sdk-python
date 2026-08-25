"""Concrete ``RolloutDriver`` over the v0.3 HTTP + callback protocol.

Owns admission (202 / 429), the callback rendezvous, and the best-effort
``/rollout/cancel`` teardown that follows a callback timeout or task
cancellation. A transport error on ``POST /rollout`` fails the item; the
supervisor retries it on its next invocation. Persistent dispatch/running
state and user-requested cancellation terminal semantics belong to a
supervisor, not this driver.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from contextlib import suppress

import httpx

from osmosis_ai.rollout.controller.store import CallbackStore, TerminalCallbackResult
from osmosis_ai.rollout.driver import RolloutDriver, RolloutOutcome, RolloutRunRequest
from osmosis_ai.rollout.types import (
    CancelRolloutsRequest,
    GraderStatus,
    RolloutInitRequest,
    RolloutStatus,
)

logger: logging.Logger = logging.getLogger(__name__)

# Bounds on a server-dictated Retry-After. The floor keeps a server that
# answers 429 with "0" from turning the admission loop into a busy spin; the
# ceiling keeps an outsized header from parking a rollout for hours when no
# admission budget is configured.
_RETRY_AFTER_FLOOR_SEC = 0.05
_MAX_RETRY_AFTER_SEC = 60.0
_CANCEL_REQUEST_TIMEOUT_SEC = 5.0


class RolloutAdmissionTimeoutError(TimeoutError):
    """The rollout server stayed backpressured beyond the admission budget."""


class RolloutProtocolError(RuntimeError):
    """The rollout server returned a response outside the v0.3 admission contract.

    ``status_code`` is part of the contract: a caller has to tell a refusal of
    *this* request apart from a server that is simply broken.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code: int = status_code


class _BorrowedTransport(httpx.AsyncBaseTransport):
    """Delegates to a caller-owned transport without ever closing it.

    Each cancel POST runs in its own short-lived client, and closing an
    ``httpx.AsyncClient`` also closes the transport it was handed — which
    would break every cancellation after the first through a shared
    ``cancel_transport``. The base class's ``aclose`` is already a no-op, so
    only request handling is delegated.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(request)


def _retry_after_seconds(response: httpx.Response, default: float = 1.0) -> float:
    """Parse Retry-After into a finite, positive, bounded wait; else the default."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(_RETRY_AFTER_FLOOR_SEC, value), _MAX_RETRY_AFTER_SEC)


class HttpRolloutDriver(RolloutDriver):
    """POST /rollout driver with in-memory callback rendezvous."""

    def __init__(
        self,
        *,
        rollout_base_url: str,
        callback_store: CallbackStore,
        completion_url_for: Callable[[str], str],
        grader_url_for: Callable[[str], str],
        chat_completions_url_for: Callable[[str], str],
        chat_api_key: str | None,
        controller_api_key: str,
        http_client: httpx.AsyncClient | None = None,
        cancel_transport: httpx.AsyncBaseTransport | None = None,
        admission_timeout_sec: float | None = None,
        callback_timeout_sec: float | None = None,
    ) -> None:
        # Cancellation deliberately does NOT reuse ``http_client``: its pool
        # may hold connection locks the just-cancelled request never released
        # (see _cancel_rollout), so cancel POSTs go through a fresh default
        # client per call. A caller whose ``http_client`` carries a
        # non-default channel — an ASGI/mock transport, proxy, or custom TLS —
        # must pass ``cancel_transport`` so cancels reach the same place.
        #
        # An empty key would admit rollouts whose callbacks the listener then
        # rejects as unauthenticated — a hang, not an error. Refuse it here.
        if not controller_api_key or not controller_api_key.strip():
            raise ValueError("controller_api_key must be a non-empty string")
        self._rollout_base_url = rollout_base_url.rstrip("/")
        self._store = callback_store
        self._completion_url_for = completion_url_for
        self._grader_url_for = grader_url_for
        self._chat_completions_url_for = chat_completions_url_for
        self._chat_api_key = chat_api_key
        self._controller_api_key = controller_api_key
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient()
        self._cancel_transport = cancel_transport
        self._admission_timeout_sec = admission_timeout_sec
        self._callback_timeout_sec = callback_timeout_sec

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def run(self, request: RolloutRunRequest) -> RolloutOutcome:
        rollout_id = request.rollout_id
        if not rollout_id:
            raise ValueError("RolloutRunRequest.rollout_id is required")

        registered = False
        try:
            await self._store.register(rollout_id)
            registered = True
            init = RolloutInitRequest(
                initial_messages=request.messages,
                label=request.label,
                metadata=request.metadata,
                rollout_id=rollout_id,
                chat_completions_url=self._chat_completions_url_for(rollout_id),
                controller_api_key=self._controller_api_key,
                llm_api_key=self._chat_api_key,
                completion_callback_url=self._completion_url_for(rollout_id),
                grader_callback_url=self._grader_url_for(rollout_id),
                agent_timeout_sec=request.agent_timeout_sec,
                grader_timeout_sec=request.grader_timeout_sec,
                extra_fields=request.extra_fields,
            )
            await self._admit(init)
            terminal = await self._wait_for_terminal(rollout_id)
            return _outcome_from_terminal(terminal)
        except asyncio.CancelledError:
            if registered:
                # A failed acknowledgment must not mask the cancellation.
                with suppress(Exception):
                    await self._store.acknowledge_without_commit(rollout_id)
            await self._cancel_rollout(rollout_id)
            raise
        finally:
            if registered:
                await self._store.discard(rollout_id)

    async def _admit(self, init: RolloutInitRequest) -> None:
        payload = init.model_dump(mode="json")
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        if self._admission_timeout_sec is not None:
            deadline = loop.time() + self._admission_timeout_sec
        while True:
            response = await self._http.post(
                f"{self._rollout_base_url}/rollout",
                json=payload,
            )
            if response.status_code == 202:
                return
            if response.status_code == 429:
                delay = _retry_after_seconds(response)
                # The budget is checked between attempts, so a timeout can only
                # follow a definitive 429: the server holds nothing for this
                # rollout and no cancel or tombstone is owed.
                if deadline is not None and loop.time() + delay > deadline:
                    raise RolloutAdmissionTimeoutError(
                        f"rollout {init.rollout_id} was not admitted within "
                        f"{self._admission_timeout_sec} seconds"
                    )
                await asyncio.sleep(delay)
                continue
            raise RolloutProtocolError(
                f"POST /rollout returned {response.status_code}; "
                "only 202 and 429 are accepted",
                status_code=response.status_code,
            )

    async def _wait_for_terminal(self, rollout_id: str) -> TerminalCallbackResult:
        timeout = self._callback_timeout_sec
        if timeout is None:
            return await self._store.wait_terminal(rollout_id)
        waiter = asyncio.create_task(self._store.wait_terminal(rollout_id))
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout)
        except TimeoutError:
            result = await self._store.finalize_timeout(
                rollout_id,
                acknowledgment={"ok": True, "error_type": "callback_timeout"},
            )
            if result.source == "timeout":
                await self._cancel_rollout(rollout_id)
            return result
        finally:
            if not waiter.done():
                waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await waiter

    async def _cancel_rollout(self, rollout_id: str) -> dict[str, str]:
        request = CancelRolloutsRequest(ids=[rollout_id])
        try:
            # A fresh client, never self._http: this usually runs on the
            # cancellation path, where the shared client's in-flight request
            # was just cancelled and httpcore may not have released that
            # connection's request lock. Reusing the pool then deadlocks on a
            # wait no httpx timeout covers (observed: every worker stuck in
            # AsyncHTTPConnection Lock.acquire).
            async with httpx.AsyncClient(
                timeout=_CANCEL_REQUEST_TIMEOUT_SEC,
                transport=_BorrowedTransport(self._cancel_transport)
                if self._cancel_transport is not None
                else None,
            ) as http:
                response = await http.post(
                    f"{self._rollout_base_url}/rollout/cancel",
                    json=request.model_dump(mode="json"),
                )
        except httpx.RequestError:
            logger.warning(
                "Failed to cancel rollout %s",
                rollout_id,
                exc_info=True,
            )
            return {}
        if response.status_code >= 400:
            logger.warning(
                "Cancel rollout %s returned HTTP %s",
                rollout_id,
                response.status_code,
            )
            return {}
        try:
            payload = response.json()
        except ValueError:
            logger.warning(
                "Cancel rollout %s returned malformed JSON",
                rollout_id,
                exc_info=True,
            )
            return {}
        if not isinstance(payload, dict):
            logger.warning(
                "Cancel rollout %s returned a non-object JSON body",
                rollout_id,
            )
            return {}
        dispositions = payload.get("dispositions", {})
        if not isinstance(dispositions, dict):
            logger.warning(
                "Cancel rollout %s returned malformed dispositions",
                rollout_id,
            )
            return {}
        return dispositions


def _outcome_from_terminal(terminal: TerminalCallbackResult) -> RolloutOutcome:
    if terminal.source == "timeout":
        return RolloutOutcome(
            status=RolloutStatus.FAILURE,
            error="callback_timeout",
            rollout_id=terminal.rollout_id,
        )
    if terminal.source == "cancelled":
        return RolloutOutcome(
            status=RolloutStatus.CANCELLED,
            error="cancelled",
            rollout_id=terminal.rollout_id,
        )
    grader = terminal.grader
    if grader is None:
        return RolloutOutcome(
            status=RolloutStatus.FAILURE,
            error="missing terminal grader callback",
            rollout_id=terminal.rollout_id,
        )
    status = (
        RolloutStatus.SUCCESS
        if grader.status == GraderStatus.SUCCESS
        else RolloutStatus.FAILURE
    )
    return RolloutOutcome(
        status=status,
        sample=grader.sample,
        error=grader.err_message,
        rollout_id=terminal.rollout_id,
    )


__all__ = [
    "HttpRolloutDriver",
    "RolloutAdmissionTimeoutError",
    "RolloutProtocolError",
]

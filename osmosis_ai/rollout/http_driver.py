"""Concrete ``RolloutDriver`` over the v0.3 HTTP + callback protocol.

Owns admission (202 / 429 / status recovery), callback rendezvous, and
``/rollout/cancel``. Persistent dispatch/running state and user-requested
cancellation terminal semantics belong to a supervisor, not this driver.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from osmosis_ai._imports import raise_optional_dependency_error
from osmosis_ai.rollout.controller.store import (
    CallbackStore,
    TerminalCallbackResult,
    UnknownRolloutIdError,
)
from osmosis_ai.rollout.driver import RolloutDriver, RolloutOutcome, RolloutRunRequest
from osmosis_ai.rollout.types import (
    CancelRolloutsRequest,
    GraderStatus,
    RolloutInitRequest,
    RolloutStatus,
    RolloutStatusResponse,
)

try:
    from osmosis_ai.rollout.controller.proxy_client import (
        EvalProxyClient,
        EvalProxySession,
    )
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval-run",
        feature="Local evaluation",
    )

logger: logging.Logger = logging.getLogger(__name__)


class _AdmissionProbe(StrEnum):
    ADMITTED = "admitted"
    UNKNOWN = "unknown"
    INDETERMINATE = "indeterminate"


#: Shortest wait the admission loop will honour between 429 retries.
_RETRY_AFTER_FLOOR_SEC = 0.05


class AdmissionUncertainError(RuntimeError):
    """Status recovery could not determine whether POST /rollout was admitted."""


class RolloutProtocolError(RuntimeError):
    """The rollout server returned a response outside the v0.3 admission contract.

    ``status_code`` is part of the contract: a caller has to tell a refusal of
    *this* request apart from a server that is simply broken.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code: int = status_code


def _retry_after_seconds(response: httpx.Response, default: float = 1.0) -> float:
    """Parse Retry-After into a finite, positive wait; else the default.

    Clamped to a floor rather than honoured exactly at zero: a server that keeps
    answering ``429`` with ``Retry-After: 0`` would otherwise turn the admission
    loop into a busy spin against itself.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(_RETRY_AFTER_FLOOR_SEC, value)


def _optional_index(request: RolloutRunRequest, name: str) -> int | None:
    for source in (request.extra_fields, request.metadata):
        if not source:
            continue
        value = source.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _status_path(rollout_id: str) -> str:
    return f"/rollout/{quote(rollout_id, safe='')}/status"


class HttpRolloutDriver(RolloutDriver):
    """POST /rollout driver with in-memory callback rendezvous."""

    def __init__(
        self,
        *,
        rollout_base_url: str,
        callback_store: CallbackStore,
        completion_url_for: Callable[[str], str],
        grader_url_for: Callable[[str], str],
        proxy_client: EvalProxyClient,
        controller_api_key: str,
        model_path: str,
        http_client: httpx.AsyncClient | None = None,
        callback_timeout_sec: float | None = None,
        status_poll_attempts: int = 5,
        status_poll_interval_sec: float = 0.05,
    ) -> None:
        self._rollout_base_url = rollout_base_url.rstrip("/")
        self._store = callback_store
        self._completion_url_for = completion_url_for
        self._grader_url_for = grader_url_for
        self._proxy_client = proxy_client
        self._controller_api_key = controller_api_key
        self._model_path = model_path
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient()
        self._callback_timeout_sec = callback_timeout_sec
        self._status_poll_attempts = status_poll_attempts
        self._status_poll_interval_sec = status_poll_interval_sec

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def run(self, request: RolloutRunRequest) -> RolloutOutcome:
        rollout_id = request.rollout_id
        if not rollout_id:
            raise ValueError("RolloutRunRequest.rollout_id is required")

        proxy_session: EvalProxySession | None = None
        registered = False
        try:
            await self._store.register(rollout_id)
            registered = True
            proxy_session = await self._proxy_client.create_session(
                rollout_id=rollout_id,
                model_path=self._model_path,
                row_index=_optional_index(request, "row_index"),
                run_index=_optional_index(request, "run_index"),
            )
            init = RolloutInitRequest(
                initial_messages=request.messages,
                label=request.label,
                metadata=request.metadata,
                rollout_id=rollout_id,
                chat_completions_url=proxy_session.api_base_url,
                controller_api_key=self._controller_api_key,
                llm_api_key=proxy_session.token,
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
                await self._store.acknowledge_without_commit(rollout_id)
            await self._cancel_rollout(rollout_id)
            raise
        finally:
            if registered:
                await self._store.discard(rollout_id)
            if proxy_session is not None:
                await self._close_proxy_session(proxy_session)

    async def cancel(self, rollout_id: str) -> dict[str, str]:
        """Tombstone a known active run and forward the remote cancel.

        Unknown ids never mutate the callback store (a tombstone would poison
        a later registration); the remote cancel is still forwarded because
        the server may know the rollout.
        """
        try:
            await self._store.acknowledge_without_commit(rollout_id)
        except UnknownRolloutIdError:
            logger.warning(
                "Cancel for unknown rollout %s; forwarding remote cancel only",
                rollout_id,
            )
        return await self._cancel_rollout(rollout_id)

    def _is_cancelled_locally(self, rollout_id: str) -> bool:
        terminal = self._store.terminal_for(rollout_id)
        return terminal is not None and terminal.source == "cancelled"

    async def _admit(self, init: RolloutInitRequest) -> None:
        rollout_id = init.rollout_id
        if rollout_id is None:
            raise ValueError("rollout_id is required for admission")
        payload = init.model_dump(mode="json")
        while True:
            # cancel() may land while we wait through 429 backpressure or
            # status recovery; never POST work that is already cancelled.
            if self._is_cancelled_locally(rollout_id):
                return
            try:
                response = await self._http.post(
                    f"{self._rollout_base_url}/rollout",
                    json=payload,
                )
            except httpx.RequestError:
                probe = await self._recover_after_ambiguous_post(rollout_id)
                if probe is _AdmissionProbe.ADMITTED:
                    await self._cancel_remotely_if_cancelled_locally(rollout_id)
                    return
                if probe is _AdmissionProbe.UNKNOWN:
                    continue
                # The POST may have been admitted even though recovery stayed
                # indeterminate; best-effort remote cancel so an unobserved
                # rollout cannot keep running. _cancel_rollout swallows its
                # own failures, so the admission error below always wins.
                await self._cancel_rollout(rollout_id)
                raise AdmissionUncertainError(
                    f"admission uncertain for rollout {rollout_id}: "
                    "status recovery stayed indeterminate"
                ) from None
            if response.status_code == 202:
                await self._cancel_remotely_if_cancelled_locally(rollout_id)
                return
            if response.status_code == 429:
                await asyncio.sleep(_retry_after_seconds(response))
                continue
            raise RolloutProtocolError(
                f"POST /rollout returned {response.status_code}; "
                "only 202 and 429 are accepted",
                status_code=response.status_code,
            )

    async def _cancel_remotely_if_cancelled_locally(self, rollout_id: str) -> None:
        """Re-issue the remote cancel when cancel() raced a successful POST."""
        if self._is_cancelled_locally(rollout_id):
            await self._cancel_rollout(rollout_id)

    async def _recover_after_ambiguous_post(self, rollout_id: str) -> _AdmissionProbe:
        last = _AdmissionProbe.INDETERMINATE
        attempts = max(1, self._status_poll_attempts)
        for attempt in range(attempts):
            last = await self._probe_status(rollout_id)
            if last is _AdmissionProbe.ADMITTED:
                return last
            if last is _AdmissionProbe.UNKNOWN:
                return last
            if attempt + 1 < attempts:
                await asyncio.sleep(self._status_poll_interval_sec)
        return last

    async def _probe_status(self, rollout_id: str) -> _AdmissionProbe:
        try:
            response = await self._http.get(
                f"{self._rollout_base_url}{_status_path(rollout_id)}"
            )
        except httpx.RequestError:
            return _AdmissionProbe.INDETERMINATE
        if response.status_code != 200:
            return _AdmissionProbe.INDETERMINATE
        try:
            body = RolloutStatusResponse.model_validate_json(response.content)
        except ValidationError:
            return _AdmissionProbe.INDETERMINATE
        if body.rollout_id != rollout_id:
            # An answer about a different rollout must never trigger a
            # re-POST; treat it as indeterminate.
            return _AdmissionProbe.INDETERMINATE
        if body.status is RolloutStatus.UNKNOWN:
            return _AdmissionProbe.UNKNOWN
        return _AdmissionProbe.ADMITTED

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
            response = await self._http.post(
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

    async def _close_proxy_session(self, session: EvalProxySession) -> None:
        try:
            await self._proxy_client.close_session(session.rollout_id)
        except Exception:
            logger.warning(
                "eval-proxy session close failed for %s",
                session.rollout_id,
                exc_info=True,
            )


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
    "AdmissionUncertainError",
    "HttpRolloutDriver",
    "RolloutProtocolError",
]

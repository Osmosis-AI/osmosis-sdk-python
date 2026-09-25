from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Generator
from time import monotonic
from typing import Any

import httpx

from osmosis_ai.rollout.types import (
    POLLING_LEASE_HEADER,
    CancelRolloutsRequest,
    CancelRolloutsResponse,
    DrainRolloutsRequest,
    DrainRolloutsResponse,
    MessageDict,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutLifecycle,
    RolloutResultResponse,
    RolloutStatus,
)

_RETRY_AFTER_FLOOR_SEC = 0.05
_MAX_RETRY_AFTER_SEC = 60.0
_CANCEL_REQUEST_TIMEOUT_SEC = 5.0
_RESULT_READ_GRACE_SEC = 10.0
_RESULT_RETRY_DELAYS_SEC = (0.1, 0.5)
_FINISHED_STATUSES = frozenset(
    {RolloutStatus.SUCCESS, RolloutStatus.FAILURE, RolloutStatus.CANCELLED}
)
_STATUS_ORDER = {
    RolloutStatus.QUEUED: 0,
    RolloutStatus.RUNNING: 1,
    RolloutStatus.GRADING: 2,
    RolloutStatus.SUCCESS: 3,
}

logger: logging.Logger = logging.getLogger(__name__)


class RolloutAdmissionTimeoutError(TimeoutError):
    pass


class RolloutProtocolError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code: int = status_code


class RolloutHandle:
    def __init__(
        self,
        client: RolloutClient,
        admission: RolloutInitResponse,
        *,
        lease_deadline: float | None = None,
    ) -> None:
        self.rollout_id: str = admission.rollout_id
        self.status: RolloutStatus = admission.status
        self.latest_result: RolloutResultResponse | None = None
        self.status_changed: asyncio.Condition = asyncio.Condition()
        self.polling_finished: bool = False
        self.result_task: asyncio.Task[RolloutResultResponse] = asyncio.create_task(
            self._wait_for_completion(
                client,
                admission,
                lease_deadline
                if lease_deadline is not None
                else monotonic() + admission.polling_lease_timeout_sec,
            )
        )

    def __await__(self) -> Generator[Any, None, RolloutResultResponse]:
        return self.result_task.__await__()

    def cancel(self) -> bool:
        return self.result_task.cancel()

    def done(self) -> bool:
        return self.result_task.done()

    async def wait_for_running(self) -> RolloutStatus:
        return await self._wait_for_status(RolloutStatus.RUNNING)

    async def wait_for_grading(self) -> RolloutStatus:
        return await self._wait_for_status(RolloutStatus.GRADING)

    async def wait_for_completion(self) -> RolloutResultResponse:
        return await self.result_task

    async def _wait_for_status(self, status: RolloutStatus) -> RolloutStatus:
        def reached() -> bool:
            return self.status in _FINISHED_STATUSES or (
                self.status in _STATUS_ORDER
                and _STATUS_ORDER[self.status] >= _STATUS_ORDER[status]
            )

        async with self.status_changed:
            await self.status_changed.wait_for(
                lambda: reached() or self.polling_finished
            )
            current = self.status

        if reached():
            return current
        return (await self.result_task).status

    async def _wait_for_completion(
        self,
        client: RolloutClient,
        admission: RolloutInitResponse,
        lease_deadline: float,
    ) -> RolloutResultResponse:
        try:
            while True:
                result, lease_deadline = await client._get_result(
                    admission, lease_deadline=lease_deadline
                )
                async with self.status_changed:
                    self.status = result.status
                    self.latest_result = result
                    self.status_changed.notify_all()
                if result.status in _FINISHED_STATUSES:
                    return result
        finally:
            async with self.status_changed:
                self.polling_finished = True
                self.status_changed.notify_all()


def _retry_after_seconds(response: httpx.Response, default: float = 1.0) -> float:
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


def _admission(response: httpx.Response, rollout_id: str) -> RolloutInitResponse:
    if response.status_code != 202:
        raise RolloutProtocolError(
            f"POST /rollout returned {response.status_code}; "
            "only 202 and 429 are accepted",
            status_code=response.status_code,
        )
    try:
        admission = RolloutInitResponse.model_validate(response.json())
    except ValueError as exc:
        raise RolloutProtocolError(
            "POST /rollout returned an invalid response",
            status_code=response.status_code,
        ) from exc
    if admission.rollout_id != rollout_id:
        raise RolloutProtocolError(
            "POST /rollout returned a different rollout_id",
            status_code=response.status_code,
        )
    return admission


def _result(response: httpx.Response, rollout_id: str) -> RolloutResultResponse:
    if response.status_code != 200:
        raise RolloutProtocolError(
            f"GET /rollout/{{id}}/result returned {response.status_code}",
            status_code=response.status_code,
        )
    try:
        result = RolloutResultResponse.model_validate(response.json())
    except ValueError as exc:
        raise RolloutProtocolError(
            "GET /rollout/{id}/result returned an invalid response",
            status_code=response.status_code,
        ) from exc
    if result.rollout_id != rollout_id:
        raise RolloutProtocolError(
            "GET /rollout/{id}/result returned a different rollout_id",
            status_code=response.status_code,
        )
    return result


def _cancelled(response: httpx.Response) -> CancelRolloutsResponse:
    if response.status_code != 200:
        raise RolloutProtocolError(
            f"POST /rollout/cancel returned {response.status_code}",
            status_code=response.status_code,
        )
    try:
        return CancelRolloutsResponse.model_validate(response.json())
    except ValueError as exc:
        raise RolloutProtocolError(
            "POST /rollout/cancel returned an invalid response",
            status_code=response.status_code,
        ) from exc


class RolloutClient:
    def __init__(
        self,
        *,
        url: str,
        http_client: httpx.AsyncClient | None = None,
        admission_timeout_sec: float | None = None,
        api_key: str | None = None,
    ) -> None:
        if admission_timeout_sec is not None and not math.isfinite(
            admission_timeout_sec
        ):
            raise ValueError(
                "admission_timeout_sec must be finite; omit it to wait unbounded"
            )
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key must be non-empty when provided")
        self.url: str = url.rstrip("/")
        self.owns_http_client: bool = http_client is None
        self.http_client: httpx.AsyncClient = http_client or httpx.AsyncClient()
        self.admission_timeout_sec: float | None = admission_timeout_sec
        self._auth_headers = {"Authorization": "Bearer " + api_key} if api_key else {}

    async def health(self) -> dict[str, Any]:
        """Read health without assuming any backend-specific counters."""
        response = await self.http_client.get(
            f"{self.url}/health", headers=self._auth_headers, follow_redirects=False
        )
        if response.status_code != 200:
            raise RolloutProtocolError(
                "GET /health failed", status_code=response.status_code
            )
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except ValueError as exc:
            raise RolloutProtocolError(
                "GET /health returned invalid JSON", status_code=200
            ) from exc

    async def wait_idle(
        self,
        *,
        timeout_sec: float = 120,
        poll_interval_sec: float = 1,
        process_id: str | None = None,
    ) -> dict[str, Any]:
        """Wait for finalization; this observation does not fence admissions.

        Older servers without lifecycle metadata are rejected rather than
        inferring idle from backend-specific counters. A changed process also
        invalidates the observation.
        """
        for value in (timeout_sec, poll_interval_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("wait timeouts must be positive and finite")
        async with asyncio.timeout(timeout_sec):
            while True:
                try:
                    health = await self.health()
                except RolloutProtocolError as error:
                    if error.status_code not in {408, 429, 500, 502, 503, 504}:
                        raise
                    await asyncio.sleep(poll_interval_sec)
                    continue
                except httpx.TransportError:
                    await asyncio.sleep(poll_interval_sec)
                    continue
                current_instance = health.get("process_id")
                if not isinstance(current_instance, str) or not current_instance:
                    raise RolloutProtocolError(
                        "Server has no lifecycle identity", status_code=200
                    )
                if process_id is not None and current_instance != process_id:
                    raise RolloutProtocolError(
                        "Server restarted while waiting", status_code=200
                    )
                process_id = current_instance
                try:
                    state = RolloutLifecycle.model_validate(health.get("lifecycle"))
                except ValueError as exc:
                    raise RolloutProtocolError(
                        "Server has no valid lifecycle state", status_code=200
                    ) from exc
                if state.active_rollouts == 0:
                    return health
                await asyncio.sleep(poll_interval_sec)

    async def drain(self, *, timeout_sec: float = 30) -> DrainRolloutsResponse:
        """Fence admissions and wait boundedly; a timeout leaves the fence set."""
        request = DrainRolloutsRequest(timeout_sec=timeout_sec)
        async with asyncio.timeout(timeout_sec + 5):
            response = await self.http_client.post(
                f"{self.url}/drain",
                json=request.model_dump(),
                headers=self._auth_headers,
                timeout=timeout_sec + 5,
                follow_redirects=False,
            )
        if response.status_code != 200:
            raise RolloutProtocolError(
                "POST /drain failed", status_code=response.status_code
            )
        try:
            result = DrainRolloutsResponse.model_validate(response.json())
        except ValueError as exc:
            raise RolloutProtocolError(
                "POST /drain returned invalid JSON", status_code=200
            ) from exc
        if result.accepting_rollouts or result.drained != (result.active_rollouts == 0):
            raise RolloutProtocolError(
                "POST /drain returned inconsistent state", status_code=200
            )
        return result

    async def aclose(self) -> None:
        if self.owns_http_client:
            await self.http_client.aclose()

    async def run_rollout(
        self,
        initial_messages: list[MessageDict],
        chat_completions_url: str,
        rollout_id: str,
        llm_api_key: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        grade: bool = True,
        agent_timeout_sec: float | None = None,
        grader_timeout_sec: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> RolloutResultResponse:
        rollout = await self.run_rollout_async(
            initial_messages=initial_messages,
            chat_completions_url=chat_completions_url,
            rollout_id=rollout_id,
            llm_api_key=llm_api_key,
            label=label,
            metadata=metadata,
            grade=grade,
            agent_timeout_sec=agent_timeout_sec,
            grader_timeout_sec=grader_timeout_sec,
            extra_fields=extra_fields,
        )
        return await rollout

    async def run_rollout_async(
        self,
        initial_messages: list[MessageDict],
        chat_completions_url: str,
        rollout_id: str,
        llm_api_key: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        grade: bool = True,
        agent_timeout_sec: float | None = None,
        grader_timeout_sec: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> RolloutHandle:
        request = RolloutInitRequest(
            initial_messages=initial_messages,
            label=label,
            metadata=metadata,
            rollout_id=rollout_id,
            chat_completions_url=chat_completions_url,
            llm_api_key=llm_api_key,
            grade=grade,
            agent_timeout_sec=agent_timeout_sec,
            grader_timeout_sec=grader_timeout_sec,
            extra_fields=extra_fields,
        )
        deadline = asyncio.timeout(self.admission_timeout_sec)
        response_pending = False
        try:
            async with deadline:
                while True:
                    response_pending = True
                    request_started = monotonic()
                    response = await self.http_client.post(
                        f"{self.url}/rollout",
                        json=request.model_dump(mode="json"),
                        headers=self._auth_headers,
                        follow_redirects=False,
                    )
                    response_pending = False
                    if response.status_code != 429:
                        admission = _admission(response, rollout_id)
                        return RolloutHandle(
                            self,
                            admission,
                            lease_deadline=request_started
                            + admission.polling_lease_timeout_sec,
                        )
                    await asyncio.sleep(_retry_after_seconds(response))
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            message = (
                f"admission for rollout {rollout_id} timed out after "
                f"{self.admission_timeout_sec} seconds"
            )
            if response_pending:
                # A lost response may be either an acceptance or a duplicate-ID
                # rejection. Cancelling by ID could terminate somebody else's work.
                message += (
                    "; admission may have succeeded; the server requests cancellation "
                    "when an unobserved rollout's polling lease expires"
                )
            raise RolloutAdmissionTimeoutError(message) from exc

    async def _get_result(
        self, admission: RolloutInitResponse, *, lease_deadline: float
    ) -> tuple[RolloutResultResponse, float]:
        timeout = admission.result_wait_timeout_sec + _RESULT_READ_GRACE_SEC
        delays = iter(_RESULT_RETRY_DELAYS_SEC)
        last_error: httpx.TransportError | None = None
        while True:
            # A successful response confirms renewal when its request reached
            # the server, not when the long-poll response arrived. A failed
            # request may never have reached it, so cannot extend this budget.
            retry_deadline = asyncio.timeout(
                max(0.0, lease_deadline - monotonic())
                if last_error is not None
                else None
            )
            try:
                async with retry_deadline:
                    request_started = monotonic()
                    response = await self.http_client.get(
                        f"{self.url}/rollout/{admission.rollout_id}/result",
                        headers={
                            **self._auth_headers,
                            POLLING_LEASE_HEADER: admission.polling_lease_token,
                        },
                        timeout=timeout,
                        follow_redirects=False,
                    )
            except TimeoutError as exc:
                if retry_deadline.expired() and last_error is not None:
                    raise last_error from exc
                raise
            except (
                httpx.RemoteProtocolError,
                httpx.NetworkError,
                httpx.ReadTimeout,
            ) as exc:
                delay = next(delays, None)
                if delay is None or monotonic() + delay + timeout >= lease_deadline:
                    raise
                logger.warning(
                    "Retrying result read for rollout %s after %s in %.1fs",
                    admission.rollout_id,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                if monotonic() + timeout >= lease_deadline:
                    raise
                last_error = exc
            else:
                return (
                    _result(response, admission.rollout_id),
                    request_started + admission.polling_lease_timeout_sec,
                )

    async def cancel_rollout(self, rollout_id: str) -> CancelRolloutsResponse:
        request = CancelRolloutsRequest(ids=[rollout_id])
        # HTTP timeouts do not bound connection-pool lock acquisition after a
        # cancelled request. Bound the entire operation as well.
        async with asyncio.timeout(_CANCEL_REQUEST_TIMEOUT_SEC):
            response = await self.http_client.post(
                f"{self.url}/rollout/cancel",
                json=request.model_dump(mode="json"),
                headers=self._auth_headers,
                follow_redirects=False,
                timeout=_CANCEL_REQUEST_TIMEOUT_SEC,
            )
        return _cancelled(response)


__all__ = [
    "RolloutAdmissionTimeoutError",
    "RolloutClient",
    "RolloutHandle",
    "RolloutProtocolError",
]

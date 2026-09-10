from __future__ import annotations

import asyncio
import math
from collections.abc import Generator
from typing import Any

import httpx

from osmosis_ai.rollout.types import (
    POLLING_LEASE_HEADER,
    CancelRolloutsRequest,
    CancelRolloutsResponse,
    MessageDict,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutResultResponse,
    RolloutStatus,
)

_RETRY_AFTER_FLOOR_SEC = 0.05
_MAX_RETRY_AFTER_SEC = 60.0
_CANCEL_REQUEST_TIMEOUT_SEC = 5.0
_RESULT_READ_GRACE_SEC = 10.0
_FINISHED_STATUSES = frozenset(
    {RolloutStatus.SUCCESS, RolloutStatus.FAILURE, RolloutStatus.CANCELLED}
)
_STATUS_ORDER = {
    RolloutStatus.QUEUED: 0,
    RolloutStatus.RUNNING: 1,
    RolloutStatus.GRADING: 2,
    RolloutStatus.SUCCESS: 3,
}


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
    ) -> None:
        self.rollout_id: str = admission.rollout_id
        self.status: RolloutStatus = admission.status
        self.latest_result: RolloutResultResponse | None = None
        self.status_changed: asyncio.Condition = asyncio.Condition()
        self.polling_finished: bool = False
        self.result_task: asyncio.Task[RolloutResultResponse] = asyncio.create_task(
            self._wait_for_completion(client, admission)
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
    ) -> RolloutResultResponse:
        try:
            while True:
                result = await client._get_result(admission)
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
    ) -> None:
        if admission_timeout_sec is not None and not math.isfinite(
            admission_timeout_sec
        ):
            raise ValueError(
                "admission_timeout_sec must be finite; omit it to wait unbounded"
            )
        self.url: str = url.rstrip("/")
        self.owns_http_client: bool = http_client is None
        self.http_client: httpx.AsyncClient = http_client or httpx.AsyncClient()
        self.admission_timeout_sec: float | None = admission_timeout_sec

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
                    response = await self.http_client.post(
                        f"{self.url}/rollout",
                        json=request.model_dump(mode="json"),
                    )
                    response_pending = False
                    if response.status_code != 429:
                        admission = _admission(response, rollout_id)
                        return RolloutHandle(self, admission)
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
        self, admission: RolloutInitResponse
    ) -> RolloutResultResponse:
        timeout = admission.result_wait_timeout_sec + _RESULT_READ_GRACE_SEC
        response = await self.http_client.get(
            f"{self.url}/rollout/{admission.rollout_id}/result",
            headers={POLLING_LEASE_HEADER: admission.polling_lease_token},
            timeout=timeout,
        )
        return _result(response, admission.rollout_id)

    async def cancel_rollout(self, rollout_id: str) -> CancelRolloutsResponse:
        request = CancelRolloutsRequest(ids=[rollout_id])
        # HTTP timeouts do not bound connection-pool lock acquisition after a
        # cancelled request. Bound the entire operation as well.
        async with asyncio.timeout(_CANCEL_REQUEST_TIMEOUT_SEC):
            response = await self.http_client.post(
                f"{self.url}/rollout/cancel",
                json=request.model_dump(mode="json"),
                timeout=_CANCEL_REQUEST_TIMEOUT_SEC,
            )
        return _cancelled(response)


__all__ = [
    "RolloutAdmissionTimeoutError",
    "RolloutClient",
    "RolloutHandle",
    "RolloutProtocolError",
]

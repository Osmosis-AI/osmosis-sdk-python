from __future__ import annotations

import asyncio
import math
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


class RolloutAdmissionTimeoutError(TimeoutError):
    pass


class RolloutProtocolError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code: int = status_code


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


def _admission(response: httpx.Response) -> RolloutInitResponse:
    if response.status_code != 202:
        raise RolloutProtocolError(
            f"POST /rollout returned {response.status_code}; "
            "only 202 and 429 are accepted",
            status_code=response.status_code,
        )
    try:
        return RolloutInitResponse.model_validate(response.json())
    except ValueError as exc:
        raise RolloutProtocolError(
            "POST /rollout returned an invalid response",
            status_code=response.status_code,
        ) from exc


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
        self.url = url.rstrip("/")
        self.owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient()
        self.admission_timeout_sec = admission_timeout_sec

    async def aclose(self) -> None:
        if self.owns_http_client:
            await self.http_client.aclose()

    async def request_rollout(
        self,
        initial_messages: list[MessageDict],
        chat_completions_url: str,
        rollout_id: str,
        chat_completions_api_key: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        grade: bool = True,
        agent_timeout_sec: float | None = None,
        grader_timeout_sec: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> RolloutResultResponse:
        future = await self.request_rollout_async(
            initial_messages=initial_messages,
            chat_completions_url=chat_completions_url,
            rollout_id=rollout_id,
            chat_completions_api_key=chat_completions_api_key,
            label=label,
            metadata=metadata,
            grade=grade,
            agent_timeout_sec=agent_timeout_sec,
            grader_timeout_sec=grader_timeout_sec,
            extra_fields=extra_fields,
        )
        return await future

    async def request_rollout_async(
        self,
        initial_messages: list[MessageDict],
        chat_completions_url: str,
        rollout_id: str,
        chat_completions_api_key: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        grade: bool = True,
        agent_timeout_sec: float | None = None,
        grader_timeout_sec: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> asyncio.Task[RolloutResultResponse]:
        request = RolloutInitRequest(
            initial_messages=initial_messages,
            label=label,
            metadata=metadata,
            rollout_id=rollout_id,
            chat_completions_url=chat_completions_url,
            chat_completions_api_key=chat_completions_api_key,
            grade=grade,
            agent_timeout_sec=agent_timeout_sec,
            grader_timeout_sec=grader_timeout_sec,
            extra_fields=extra_fields,
        )
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + self.admission_timeout_sec
            if self.admission_timeout_sec is not None
            else None
        )
        while True:
            response = await self.http_client.post(
                f"{self.url}/rollout",
                json=request.model_dump(mode="json"),
            )
            if response.status_code != 429:
                admission = _admission(response)
                return asyncio.create_task(self._wait_for_completion(admission))
            delay = _retry_after_seconds(response)
            if deadline is not None and loop.time() + delay > deadline:
                raise RolloutAdmissionTimeoutError(
                    f"rollout {rollout_id} was not admitted within "
                    f"{self.admission_timeout_sec} seconds"
                )
            await asyncio.sleep(delay)

    async def _wait_for_completion(
        self, admission: RolloutInitResponse
    ) -> RolloutResultResponse:
        timeout = admission.result_wait_timeout_sec + _RESULT_READ_GRACE_SEC
        while True:
            response = await self.http_client.get(
                f"{self.url}/rollout/{admission.rollout_id}/result",
                headers={POLLING_LEASE_HEADER: admission.polling_lease_token},
                timeout=timeout,
            )
            result = _result(response, admission.rollout_id)
            if result.status in _FINISHED_STATUSES:
                return result

    async def cancel_rollout(self, rollout_id: str) -> CancelRolloutsResponse:
        request = CancelRolloutsRequest(ids=[rollout_id])
        response = await self.http_client.post(
            f"{self.url}/rollout/cancel",
            json=request.model_dump(mode="json"),
            timeout=_CANCEL_REQUEST_TIMEOUT_SEC,
        )
        return _cancelled(response)


__all__ = [
    "RolloutAdmissionTimeoutError",
    "RolloutClient",
    "RolloutProtocolError",
]

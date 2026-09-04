from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from osmosis_ai.rollout.client import (
    RolloutAdmissionTimeoutError,
    RolloutClient,
    RolloutProtocolError,
)
from osmosis_ai.rollout.client.client import (
    _retry_after_seconds,
)
from osmosis_ai.rollout.types import POLLING_LEASE_HEADER, RolloutStatus

ROLLOUT_ID = "f" * 32


def request() -> dict[str, Any]:
    return {
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": f"http://bridge/{ROLLOUT_ID}",
        "rollout_id": ROLLOUT_ID,
        "llm_api_key": "bridge-key",
        "label": "yes",
        "metadata": {"split": "train"},
        "grade": False,
        "agent_timeout_sec": 30.0,
        "grader_timeout_sec": 10.0,
        "extra_fields": {"custom": "x"},
    }


def admission() -> dict[str, Any]:
    return {
        "rollout_id": ROLLOUT_ID,
        "status": "queued",
        "polling_lease_token": "test-lease",
        "result_wait_timeout_sec": 30.0,
        "polling_lease_timeout_sec": 120.0,
    }


def client(handler: Any, **kwargs: Any) -> RolloutClient:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://rollout"
    )
    return RolloutClient(
        url="http://rollout",
        http_client=http_client,
        **kwargs,
    )


async def completed(rollout_client: RolloutClient):
    return await rollout_client.run_rollout(**request())


def json_body(value: httpx.Request) -> dict[str, Any]:
    return json.loads(value.content)


async def test_run_rollout_async_returns_completion_task() -> None:
    requests: list[httpx.Request] = []
    polls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal polls
        requests.append(http_request)
        if http_request.method == "POST":
            return httpx.Response(202, json=admission())
        polls += 1
        if polls == 1:
            return httpx.Response(
                200, json={"rollout_id": ROLLOUT_ID, "status": "running"}
            )
        return httpx.Response(
            200,
            json={
                "rollout_id": ROLLOUT_ID,
                "status": "success",
                "sample": {"messages": [], "reward": 1.0},
            },
        )

    rollout_client = client(handler)
    future = await rollout_client.run_rollout_async(**request())

    assert polls == 0
    assert isinstance(future, asyncio.Task)

    outcome = await future

    assert outcome.status is RolloutStatus.SUCCESS
    assert outcome.sample is not None and outcome.sample.reward == 1.0
    assert polls == 2
    assert POLLING_LEASE_HEADER not in requests[0].headers
    lease_values = {item.headers[POLLING_LEASE_HEADER] for item in requests[1:]}
    assert lease_values == {"test-lease"}
    body = json_body(requests[0])
    assert body["chat_completions_url"] == f"http://bridge/{ROLLOUT_ID}"
    assert body["llm_api_key"] == "bridge-key"
    assert body["grade"] is False
    assert "completion_callback_url" not in body
    assert "grader_callback_url" not in body


async def test_failure_result_is_returned() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.method == "POST":
            return httpx.Response(202, json=admission())
        return httpx.Response(
            200,
            json={
                "rollout_id": ROLLOUT_ID,
                "status": "failure",
                "err_message": "polling lease expired",
                "err_category": "lease_expired",
            },
        )

    outcome = await completed(client(handler))
    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.err_message == "polling lease expired"
    assert outcome.err_category == "lease_expired"


async def test_429_retries_using_retry_after() -> None:
    posts = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if http_request.method == "POST":
            posts += 1
            if posts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(202, json=admission())
        return httpx.Response(200, json={"rollout_id": ROLLOUT_ID, "status": "success"})

    outcome = await completed(client(handler))
    assert outcome.status is RolloutStatus.SUCCESS
    assert posts == 2


async def test_admission_timeout_does_not_cancel_unaccepted_work() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/rollout"
        return httpx.Response(429, headers={"Retry-After": "60"})

    with pytest.raises(RolloutAdmissionTimeoutError):
        await client(handler, admission_timeout_sec=0.01).run_rollout(**request())


async def test_invalid_admission_response_is_a_protocol_error() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={})

    with pytest.raises(RolloutProtocolError, match="invalid response"):
        await client(handler).run_rollout(**request())


async def test_cancel_rollout_posts_cancel() -> None:
    cancelled: list[dict[str, Any]] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/rollout/cancel":
            cancelled.append(json_body(http_request))
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(http_request.url.path)

    await client(handler).cancel_rollout(ROLLOUT_ID)
    assert cancelled == [{"ids": [ROLLOUT_ID], "prefix": None, "all": False}]


async def test_run_rollout_waits_for_completion() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.method == "POST":
            return httpx.Response(202, json=admission())
        return httpx.Response(
            200,
            json={"rollout_id": ROLLOUT_ID, "status": "success"},
        )

    outcome = await client(handler).run_rollout(**request())
    assert outcome.status is RolloutStatus.SUCCESS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("inf", 1.0),
        ("junk", 1.0),
        ("0", 0.05),
        ("2.5", 2.5),
        ("86400", 60.0),
    ],
)
def test_retry_after_is_bounded(raw: str, expected: float) -> None:
    response = httpx.Response(429, headers={"Retry-After": raw})
    assert _retry_after_seconds(response) == expected

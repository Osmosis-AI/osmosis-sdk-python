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


async def test_cancel_bounds_waits_outside_http_timeouts(monkeypatch) -> None:
    from osmosis_ai.rollout.client import client as client_module

    monkeypatch.setattr(client_module, "_CANCEL_REQUEST_TIMEOUT_SEC", 0.01)
    cancelled = asyncio.Event()

    async def handler(http_request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    rollout_client = client(handler)
    async with asyncio.timeout(1.0):
        with pytest.raises(TimeoutError):
            await rollout_client.cancel_rollout(ROLLOUT_ID)
    assert cancelled.is_set()
    await rollout_client.http_client.aclose()


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
    paths: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        assert http_request.url.path == "/rollout"
        return httpx.Response(429, headers={"Retry-After": "60"})

    with pytest.raises(RolloutAdmissionTimeoutError):
        await client(handler, admission_timeout_sec=0.01).run_rollout(**request())
    assert paths == ["/rollout"]


async def test_admission_timeout_bounds_the_http_request() -> None:
    cancelled = asyncio.Event()
    paths: list[str] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    rollout_client = client(handler, admission_timeout_sec=0.01)
    try:
        async with asyncio.timeout(1.0):
            with pytest.raises(RolloutAdmissionTimeoutError) as error:
                await rollout_client.run_rollout(**request())
        assert cancelled.is_set()
        assert paths == ["/rollout"]
        assert isinstance(error.value.__cause__, TimeoutError)
        assert "admission may have succeeded" in str(error.value)
    finally:
        await rollout_client.http_client.aclose()


@pytest.mark.parametrize("existing_rollout", [False, True])
async def test_lost_admission_response_preserves_lease_ownership(
    existing_rollout: bool,
) -> None:
    from osmosis_ai.rollout.backend.base import ExecutionBackend
    from osmosis_ai.rollout.server.app import create_rollout_server

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingBackend(ExecutionBackend):
        async def execute(self, request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    app = create_rollout_server(
        backend=BlockingBackend(),
        result_wait_timeout_sec=0.01,
        polling_lease_timeout_sec=0.2,
    )
    transport = httpx.ASGITransport(app=app)
    paths: list[str] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        response = await transport.handle_async_request(http_request)
        if http_request.url.path == "/rollout":
            assert response.status_code == (409 if existing_rollout else 202)
            await started.wait()
            await asyncio.Event().wait()
        return response

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rollout"
        ) as owner:
            if existing_rollout:
                accepted = await owner.post("/rollout", json=request())
                assert accepted.status_code == 202
            rollout_client = client(handler, admission_timeout_sec=0.05)
            try:
                async with asyncio.timeout(1.0):
                    with pytest.raises(RolloutAdmissionTimeoutError):
                        await rollout_client.run_rollout(**request())
                    assert started.is_set()
                    assert not cancelled.is_set()
                    if existing_rollout:
                        result = await owner.get(
                            f"/rollout/{ROLLOUT_ID}/result",
                            headers={
                                POLLING_LEASE_HEADER: accepted.json()[
                                    "polling_lease_token"
                                ]
                            },
                        )
                        assert result.json()["status"] == "running"
                        await owner.post("/rollout/cancel", json={"ids": [ROLLOUT_ID]})
                    assert paths == ["/rollout"]
                    await cancelled.wait()
            finally:
                await rollout_client.http_client.aclose()


async def test_admission_rejects_a_different_rollout_id() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.method == "POST"
        return httpx.Response(202, json=admission() | {"rollout_id": "another-run"})

    rollout_client = client(handler)
    try:
        with pytest.raises(RolloutProtocolError, match="different rollout_id"):
            await rollout_client.run_rollout(**request())
    finally:
        await rollout_client.http_client.aclose()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_admission_timeout_must_be_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="admission_timeout_sec must be finite"):
        RolloutClient(url="http://rollout", admission_timeout_sec=timeout)


async def test_transport_timeout_is_not_reported_as_admission_deadline() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        raise TimeoutError("transport timed out")

    rollout_client = client(handler, admission_timeout_sec=1.0)
    try:
        with pytest.raises(TimeoutError, match="transport timed out") as error:
            await rollout_client.run_rollout(**request())
        assert not isinstance(error.value, RolloutAdmissionTimeoutError)
    finally:
        await rollout_client.http_client.aclose()


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

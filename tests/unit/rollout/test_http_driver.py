"""Tests for the concrete HTTP rollout driver."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from osmosis_ai.rollout.controller.store import CallbackStore, TerminalCallbackResult
from osmosis_ai.rollout.driver import RolloutRunRequest
from osmosis_ai.rollout.http_driver import (
    HttpRolloutDriver,
    RolloutAdmissionTimeoutError,
    RolloutProtocolError,
)
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutSample,
    RolloutStatus,
)

ROLLOUT_ID = "f" * 32
CHAT_BASE = "http://127.0.0.1:1/v1/rollouts"
TEST_TIMEOUT_SEC = 2.0


async def _commit(_result: TerminalCallbackResult) -> None:
    """Terminal-commit hook that is durable without doing anything."""
    return None


def _store() -> CallbackStore:
    return CallbackStore(on_terminal_commit=_commit)


def _grader() -> GraderCompleteRequest:
    return GraderCompleteRequest(
        status=GraderStatus.SUCCESS,
        rollout_id=ROLLOUT_ID,
        sample=RolloutSample(
            messages=[{"role": "assistant", "content": "ok"}],
            reward=1.0,
        ),
    )


def _request(**overrides: Any) -> RolloutRunRequest:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "label": "yes",
        "metadata": {"split": "train"},
        "rollout_id": ROLLOUT_ID,
        "agent_timeout_sec": 30.0,
        "grader_timeout_sec": 10.0,
        "extra_fields": {"row_index": 4, "run_index": 1, "custom": "x"},
    }
    payload.update(overrides)
    return RolloutRunRequest(**payload)


def _driver(
    store: CallbackStore,
    handler: Any,
    *,
    admission_timeout_sec: float | None = 5.0,
    callback_timeout_sec: float | None = 5.0,
    chat_api_key: str | None = "bridge-token",
) -> HttpRolloutDriver:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://rollout",
    )
    return HttpRolloutDriver(
        rollout_base_url="http://rollout",
        callback_store=store,
        completion_url_for=lambda rid: (
            f"http://cb/v1/rollouts/{quote(rid, safe='')}/completion"
        ),
        grader_url_for=lambda rid: (
            f"http://cb/v1/rollouts/{quote(rid, safe='')}/grader"
        ),
        chat_completions_url_for=lambda rid: f"{CHAT_BASE}/{quote(rid, safe='')}",
        chat_api_key=chat_api_key,
        controller_api_key="controller-key",
        http_client=http,
        admission_timeout_sec=admission_timeout_sec,
        callback_timeout_sec=callback_timeout_sec,
    )


async def test_202_admission_posts_init_and_returns_grader_outcome() -> None:
    store = _store()
    posted: list[dict[str, Any]] = []
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/rollout":
            posted.append(json_body(request))
            admitted.set()
            return httpx.Response(202, json={})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    driver = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await asyncio.wait_for(admitted.wait(), timeout=TEST_TIMEOUT_SEC)
    await store.handle_grader(ROLLOUT_ID, _grader())
    outcome = await asyncio.wait_for(run_task, timeout=TEST_TIMEOUT_SEC)

    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.sample is not None
    assert outcome.sample.reward == 1.0
    assert outcome.rollout_id == ROLLOUT_ID
    body = posted[0]
    assert body["rollout_id"] == ROLLOUT_ID
    # The bridge endpoint and its token are what the agent talks to.
    assert body["chat_completions_url"] == f"{CHAT_BASE}/{ROLLOUT_ID}"
    assert body["llm_api_key"] == "bridge-token"
    assert body["completion_callback_url"].endswith(f"/{ROLLOUT_ID}/completion")
    assert body["grader_callback_url"].endswith(f"/{ROLLOUT_ID}/grader")
    assert body["controller_api_key"] == "controller-key"
    assert body["controller_api_key"] != body["llm_api_key"]
    assert body["metadata"] == {"split": "train"}
    assert body["agent_timeout_sec"] == 30.0
    assert body["grader_timeout_sec"] == 10.0
    assert body["extra_fields"]["custom"] == "x"
    late_completion = await store.handle_completion(
        ROLLOUT_ID,
        RolloutCompleteRequest(status=RolloutStatus.SUCCESS),
    )
    assert late_completion == {"ok": True}


async def test_absent_chat_api_key_is_sent_as_null() -> None:
    store = _store()
    posted: list[dict[str, Any]] = []
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/rollout":
            posted.append(json_body(request))
            admitted.set()
            return httpx.Response(202, json={})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    driver = _driver(store, handler, chat_api_key=None)
    run_task = asyncio.create_task(driver.run(_request()))
    await asyncio.wait_for(admitted.wait(), timeout=TEST_TIMEOUT_SEC)
    await store.handle_grader(ROLLOUT_ID, _grader())
    await asyncio.wait_for(run_task, timeout=TEST_TIMEOUT_SEC)

    assert posted[0]["llm_api_key"] is None


async def test_429_retries_after_header_without_reregistering() -> None:
    store = _store()
    posts = 0
    registrations = {"n": 0}
    admitted = asyncio.Event()
    real_register = store.register

    async def tracked_register(rollout_id: str) -> None:
        registrations["n"] += 1
        await real_register(rollout_id)

    store.register = tracked_register  # type: ignore[method-assign]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout":
            posts += 1
            if posts == 1:
                return httpx.Response(
                    429, json={"detail": "full"}, headers={"Retry-After": "0"}
                )
            admitted.set()
            return httpx.Response(202, json={})
        raise AssertionError(request.url.path)

    driver = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await asyncio.wait_for(admitted.wait(), timeout=TEST_TIMEOUT_SEC)
    await store.handle_grader(ROLLOUT_ID, _grader())
    outcome = await asyncio.wait_for(run_task, timeout=TEST_TIMEOUT_SEC)
    assert outcome.status == RolloutStatus.SUCCESS
    assert posts == 2
    assert registrations["n"] == 1


async def test_persistent_429_hits_admission_timeout_and_cancels() -> None:
    store = _store()
    posts = 0
    cancelled: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout":
            posts += 1
            return httpx.Response(429, headers={"Retry-After": "0"})
        if request.url.path == "/rollout/cancel":
            cancelled.append(json_body(request))
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver = _driver(store, handler, admission_timeout_sec=0.02)
    with pytest.raises(RolloutAdmissionTimeoutError, match="not admitted within"):
        await driver.run(_request())

    assert posts == 1
    assert cancelled and cancelled[0]["ids"] == [ROLLOUT_ID]


async def test_post_200_is_protocol_error_without_looping() -> None:
    store = _store()
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(request.url.path)

    driver = _driver(store, handler)
    with pytest.raises(RolloutProtocolError, match="200") as excinfo:
        await driver.run(_request())
    assert posts == 1
    # The supervisor needs the status to tell a refusal of this request apart
    # from a server that is simply broken.
    assert excinfo.value.status_code == 200


async def test_transport_error_on_post_propagates_without_retry() -> None:
    store = _store()
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            raise httpx.ConnectError("reset", request=request)
        raise AssertionError(request.url.path)

    driver = _driver(store, handler)
    with pytest.raises(httpx.ConnectError):
        await driver.run(_request())
    assert posts == 1


async def test_callback_timeout_wins_over_late_grader() -> None:
    store = _store()
    cancelled: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            return httpx.Response(202, json={})
        if request.url.path == "/rollout/cancel":
            cancelled.append(json_body(request))
            return httpx.Response(
                200, json={"dispositions": {ROLLOUT_ID: "cancelled_running"}}
            )
        raise AssertionError(request.url.path)

    driver = _driver(store, handler, callback_timeout_sec=0.05)
    outcome = await driver.run(_request())
    late = await store.handle_grader(ROLLOUT_ID, _grader())

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "callback_timeout"
    assert late.get("error_type") == "callback_timeout"
    assert cancelled and cancelled[0]["ids"] == [ROLLOUT_ID]


async def test_timeout_plus_failed_cancel_keeps_timeout_result() -> None:
    store = _store()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            return httpx.Response(202, json={})
        if request.url.path == "/rollout/cancel":
            return httpx.Response(503, json={"detail": "unavailable"})
        raise AssertionError(request.url.path)

    driver = _driver(store, handler, callback_timeout_sec=0.05)
    outcome = await driver.run(_request())
    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "callback_timeout"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("inf", 1.0),
        ("-inf", 1.0),
        ("nan", 1.0),
        ("1e999", 1.0),
        ("junk", 1.0),
        # Negative and zero waits clamp to the floor so a 429 loop cannot spin.
        ("-5", 0.05),
        ("0", 0.05),
        ("2.5", 2.5),
        ("60", 60.0),
        ("86400", 60.0),
    ],
)
def test_retry_after_seconds_is_finite_non_negative_and_capped(
    raw: str, expected: float
) -> None:
    from osmosis_ai.rollout.http_driver import _retry_after_seconds

    response = httpx.Response(429, headers={"Retry-After": raw})
    assert _retry_after_seconds(response) == expected


async def test_task_cancellation_reraises_cancelled_error() -> None:
    store = _store()
    cancelled: list[dict[str, Any]] = []
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            admitted.set()
            return httpx.Response(202, json={})
        if request.url.path == "/rollout/cancel":
            cancelled.append(json_body(request))
            return httpx.Response(
                200, json={"dispositions": {ROLLOUT_ID: "cancelled_running"}}
            )
        raise AssertionError(request.url.path)

    driver = _driver(store, handler)
    task = asyncio.create_task(driver.run(_request()))
    await asyncio.wait_for(admitted.wait(), timeout=TEST_TIMEOUT_SEC)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TEST_TIMEOUT_SEC)
    assert cancelled and cancelled[0]["ids"] == [ROLLOUT_ID]
    late_completion = await store.handle_completion(
        ROLLOUT_ID,
        RolloutCompleteRequest(status=RolloutStatus.SUCCESS),
    )
    late_grader = await store.handle_grader(ROLLOUT_ID, _grader())
    assert late_completion == {"ok": True}
    assert late_grader == late_completion


def json_body(request: httpx.Request) -> dict[str, Any]:
    import json

    return json.loads(request.content)


def test_an_empty_controller_api_key_is_rejected_at_construction() -> None:
    # An empty key would admit rollouts whose callbacks the listener then
    # rejects as unauthenticated — a hang, not an error.
    with pytest.raises(ValueError, match="controller_api_key"):
        HttpRolloutDriver(
            rollout_base_url="http://127.0.0.1:1",
            callback_store=_store(),
            completion_url_for=lambda rid: f"http://127.0.0.1:1/c/{rid}",
            grader_url_for=lambda rid: f"http://127.0.0.1:1/g/{rid}",
            chat_completions_url_for=lambda rid: (
                f"http://127.0.0.1:1/v1/rollouts/{rid}"
            ),
            chat_api_key=None,
            controller_api_key="  ",
        )

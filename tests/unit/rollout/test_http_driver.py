"""Tests for the concrete HTTP rollout driver."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from osmosis_ai.rollout.controller.proxy_client import EvalProxySession
from osmosis_ai.rollout.controller.store import CallbackStore
from osmosis_ai.rollout.driver import RolloutRunRequest
from osmosis_ai.rollout.http_driver import (
    AdmissionUncertainError,
    HttpRolloutDriver,
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


class FakeProxyClient:
    def __init__(self, store: CallbackStore | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self.closed: list[str] = []
        self._store = store
        self.registered_at_create: bool | None = None

    async def create_session(
        self,
        *,
        rollout_id: str,
        model_path: str,
        row_index: int | None = None,
        run_index: int | None = None,
    ) -> EvalProxySession:
        self.registered_at_create = (
            self._store is not None and rollout_id in self._store._sessions
        )
        self.created.append(
            {
                "rollout_id": rollout_id,
                "model_path": model_path,
                "row_index": row_index,
                "run_index": run_index,
            }
        )
        return EvalProxySession(
            rollout_id=rollout_id,
            model_path=model_path,
            api_base=f"/v1/eval-sessions/{rollout_id}",
            api_base_url=f"http://proxy/v1/eval-sessions/{rollout_id}",
            token="session-token",
            row_index=row_index,
            run_index=run_index,
        )

    async def aclose(self) -> None:
        return None

    async def close_session(self, rollout_id: str) -> None:
        self.closed.append(rollout_id)


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
    callback_timeout_sec: float | None = 5.0,
    status_poll_attempts: int = 5,
    status_poll_interval_sec: float = 0.0,
) -> tuple[HttpRolloutDriver, FakeProxyClient]:
    proxy = FakeProxyClient(store)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://rollout",
    )
    driver = HttpRolloutDriver(
        rollout_base_url="http://rollout",
        callback_store=store,
        completion_url_for=lambda rid: (
            f"http://cb/v1/rollouts/{quote(rid, safe='')}/completion"
        ),
        grader_url_for=lambda rid: (
            f"http://cb/v1/rollouts/{quote(rid, safe='')}/grader"
        ),
        proxy_client=proxy,  # type: ignore[arg-type]
        controller_api_key="controller-key",
        model_path="openai/gpt-4.1-mini",
        http_client=http,
        callback_timeout_sec=callback_timeout_sec,
        status_poll_attempts=status_poll_attempts,
        status_poll_interval_sec=status_poll_interval_sec,
    )
    return driver, proxy


async def test_202_admission_posts_init_and_returns_grader_outcome() -> None:
    store = CallbackStore()
    posted: list[dict[str, Any]] = []
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/rollout":
            posted.append(json_body(request))
            admitted.set()
            return httpx.Response(202, json={})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    driver, proxy = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await admitted.wait()
    await store.handle_grader(ROLLOUT_ID, _grader())
    outcome = await run_task

    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.sample is not None
    assert outcome.sample.reward == 1.0
    assert outcome.rollout_id == ROLLOUT_ID
    body = posted[0]
    assert body["rollout_id"] == ROLLOUT_ID
    assert body["chat_completions_url"] == f"http://proxy/v1/eval-sessions/{ROLLOUT_ID}"
    assert body["completion_callback_url"].endswith(f"/{ROLLOUT_ID}/completion")
    assert body["grader_callback_url"].endswith(f"/{ROLLOUT_ID}/grader")
    assert body["controller_api_key"] == "controller-key"
    assert body["llm_api_key"] == "session-token"
    assert body["controller_api_key"] != body["llm_api_key"]
    assert body["metadata"] == {"split": "train"}
    assert body["agent_timeout_sec"] == 30.0
    assert body["grader_timeout_sec"] == 10.0
    assert body["extra_fields"]["custom"] == "x"
    assert proxy.created[0]["row_index"] == 4
    assert proxy.created[0]["run_index"] == 1
    assert proxy.registered_at_create is True
    late_completion = await store.handle_completion(
        ROLLOUT_ID,
        RolloutCompleteRequest(status=RolloutStatus.SUCCESS),
    )
    assert late_completion == {"ok": True}


async def test_row_and_run_indices_are_read_from_extra_fields_only() -> None:
    store = CallbackStore()
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            admitted.set()
            return httpx.Response(202, json={})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    driver, proxy = _driver(store, handler)
    run_task = asyncio.create_task(
        driver.run(
            _request(metadata={"row_index": 7, "run_index": 8}, extra_fields=None)
        )
    )
    await admitted.wait()
    await store.handle_grader(ROLLOUT_ID, _grader())
    await run_task

    assert proxy.created[0]["row_index"] is None
    assert proxy.created[0]["run_index"] is None


async def test_429_retries_after_header_without_reregistering() -> None:
    store = CallbackStore()
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

    driver, _proxy = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await admitted.wait()
    await store.handle_grader(ROLLOUT_ID, _grader())
    outcome = await run_task
    assert outcome.status == RolloutStatus.SUCCESS
    assert posts == 2
    assert registrations["n"] == 1


async def test_ambiguous_post_waits_when_status_shows_admitted() -> None:
    store = CallbackStore()
    posts = 0
    statuses = 0
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts, statuses
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            statuses += 1
            admitted.set()
            return httpx.Response(
                200, json={"rollout_id": ROLLOUT_ID, "status": "running"}
            )
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await admitted.wait()
    await store.handle_grader(ROLLOUT_ID, _grader())
    outcome = await run_task
    assert outcome.status == RolloutStatus.SUCCESS
    assert posts == 1
    assert statuses == 1


async def test_status_500_never_causes_second_post() -> None:
    store = CallbackStore()
    posts = 0
    statuses = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts, statuses
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            statuses += 1
            return httpx.Response(500, json={"detail": "not ready"})
        if request.url.path == "/rollout/cancel":
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=3)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert posts == 1
    assert statuses == 3


async def test_status_network_failure_never_causes_second_post() -> None:
    store = CallbackStore()
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            raise httpx.ConnectError("status down", request=request)
        if request.url.path == "/rollout/cancel":
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=3)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert posts == 1


async def test_admission_uncertain_issues_best_effort_remote_cancel() -> None:
    store = CallbackStore()
    cancel_posts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            return httpx.Response(500, json={"detail": "not ready"})
        if request.url.path == "/rollout/cancel":
            cancel_posts.append(json_body(request))
            return httpx.Response(
                200, json={"dispositions": {ROLLOUT_ID: "cancelled_queued"}}
            )
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=2)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert cancel_posts == [{"ids": [ROLLOUT_ID], "prefix": None, "all": False}]


async def test_admission_uncertain_survives_failed_best_effort_cancel() -> None:
    store = CallbackStore()
    cancel_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancel_attempts
        if request.url.path == "/rollout" and request.method == "POST":
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            return httpx.Response(500, json={"detail": "not ready"})
        if request.url.path == "/rollout/cancel":
            cancel_attempts += 1
            raise httpx.ConnectError("cancel down", request=request)
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=2)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert cancel_attempts == 1


async def test_explicit_unknown_safely_retries_post() -> None:
    store = CallbackStore()
    posts = 0
    admitted = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            if posts == 1:
                raise httpx.ConnectError("reset", request=request)
            admitted.set()
            return httpx.Response(202, json={})
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            return httpx.Response(
                200, json={"rollout_id": ROLLOUT_ID, "status": "unknown"}
            )
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler)
    run_task = asyncio.create_task(driver.run(_request()))
    await admitted.wait()
    await store.handle_grader(ROLLOUT_ID, _grader())
    await run_task
    assert posts == 2


async def test_malformed_status_is_indeterminate_and_does_not_retry_post() -> None:
    store = CallbackStore()
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            raise httpx.ConnectError("reset", request=request)
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            return httpx.Response(200, text="not-json")
        if request.url.path == "/rollout/cancel":
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=2)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert posts == 1


async def test_post_200_is_protocol_error_without_looping() -> None:
    store = CallbackStore()
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/rollout" and request.method == "POST":
            posts += 1
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler)
    with pytest.raises(RolloutProtocolError, match="200") as excinfo:
        await driver.run(_request())
    assert posts == 1
    # The supervisor needs the status to tell a refusal of this request apart
    # from a server that is simply broken.
    assert excinfo.value.status_code == 200


async def test_status_url_encodes_rollout_id() -> None:
    store = CallbackStore()
    special_id = "rid:1"
    status_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            raise httpx.ConnectError("reset", request=request)
        if request.method == "GET" and "/status" in request.url.path:
            status_urls.append(str(request.url))
            return httpx.Response(500, json={"detail": "nope"})
        if request.url.path == "/rollout/cancel":
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, status_poll_attempts=1)
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request(rollout_id=special_id))
    assert status_urls
    assert f"/rollout/{quote(special_id, safe='')}/status" in status_urls[0]


async def test_callback_timeout_wins_over_late_grader() -> None:
    store = CallbackStore()
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

    driver, _proxy = _driver(store, handler, callback_timeout_sec=0.05)
    outcome = await driver.run(_request())
    late = await store.handle_grader(ROLLOUT_ID, _grader())

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "callback_timeout"
    assert late.get("error_type") == "callback_timeout"
    assert cancelled and cancelled[0]["ids"] == [ROLLOUT_ID]


async def test_timeout_plus_failed_cancel_keeps_timeout_result() -> None:
    store = CallbackStore()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rollout" and request.method == "POST":
            return httpx.Response(202, json={})
        if request.url.path == "/rollout/cancel":
            return httpx.Response(503, json={"detail": "unavailable"})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(store, handler, callback_timeout_sec=0.05)
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


async def test_status_rollout_id_mismatch_never_causes_second_post() -> None:
    store = CallbackStore()
    rollout_posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal rollout_posts
        if request.url.path == "/rollout" and request.method == "POST":
            rollout_posts += 1
            if rollout_posts == 1:
                raise httpx.ConnectError("reset", request=request)
            return httpx.Response(202, json={})
        if request.url.path == f"/rollout/{ROLLOUT_ID}/status":
            return httpx.Response(
                200, json={"rollout_id": "e" * 32, "status": "unknown"}
            )
        if request.url.path == "/rollout/cancel":
            return httpx.Response(200, json={"dispositions": {}})
        raise AssertionError(request.url.path)

    driver, _proxy = _driver(
        store, handler, status_poll_attempts=2, callback_timeout_sec=0.05
    )
    with pytest.raises(AdmissionUncertainError):
        await driver.run(_request())
    assert rollout_posts == 1


async def test_task_cancellation_reraises_cancelled_error() -> None:
    store = CallbackStore()
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

    driver, proxy = _driver(store, handler)
    task = asyncio.create_task(driver.run(_request()))
    await admitted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled and cancelled[0]["ids"] == [ROLLOUT_ID]
    assert proxy.closed == [ROLLOUT_ID]
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

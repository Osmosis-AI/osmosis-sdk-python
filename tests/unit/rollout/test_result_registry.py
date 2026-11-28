from __future__ import annotations

import asyncio
import gc
import weakref
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from osmosis_ai.rollout.server.lease import InvalidLeaseError
from osmosis_ai.rollout.server.result_registry import (
    DuplicateRolloutError,
    RolloutFutureRegistry,
    RolloutPendingError,
    UnknownRolloutError,
)
from osmosis_ai.rollout.types import (
    RolloutErrorCategory,
    RolloutResultResponse,
    RolloutSample,
    RolloutStatus,
)

AGENT_STARTED_AT = datetime(2026, 10, 22, 17, 30, tzinfo=UTC)


def registry(
    cancelled: list[str], *, wait: float = 0.01, lease: float = 0.1
) -> RolloutFutureRegistry:
    return RolloutFutureRegistry(
        result_wait_timeout_sec=wait,
        polling_lease_timeout_sec=lease,
        result_retention_sec=60.0,
        cancel_rollout=cancelled.append,
    )


@pytest.mark.parametrize(
    ("wait", "lease", "retention"),
    [
        (0.0, 1.0, 1.0),
        (float("inf"), 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 1.0),
        (0.1, 1.0, -1.0),
        (0.1, 1.0, float("nan")),
    ],
)
def test_invalid_registry_timeouts_are_rejected(
    wait: float, lease: float, retention: float
) -> None:
    with pytest.raises(ValueError):
        RolloutFutureRegistry(
            result_wait_timeout_sec=wait,
            polling_lease_timeout_sec=lease,
            result_retention_sec=retention,
            cancel_rollout=lambda _rollout_id: None,
        )


async def test_wait_returns_finished_result_early() -> None:
    cancelled: list[str] = []
    store = registry(cancelled)
    lease = await store.register("r1")
    waiter = asyncio.create_task(store.wait_for_result("r1", lease))
    await asyncio.sleep(0)
    expected = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    assert await store.complete("r1", expected)
    assert await waiter == expected
    await store.close()


async def test_completed_sample_is_released_and_retry_returns_the_full_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    store = registry([], wait=1, lease=2)
    lease = await store.register("r1")
    sample = RolloutSample(
        messages=[{"role": "assistant", "content": "large result" * 8192}],
        trajectory_messages=None,
        reward=0.0,
        metrics={"turns": 3},
    )
    reference = weakref.ref(sample)
    result = RolloutResultResponse(
        rollout_id="r1",
        status=RolloutStatus.SUCCESS,
        sample=sample,
        agent_started_at=AGENT_STARTED_AT,
    )
    expected = result.model_dump_json()
    try:
        assert await store.complete("r1", result)
        del result, sample
        gc.collect()
        assert reference() is None
        assert any(tmp_path.iterdir())
        with pytest.raises(InvalidLeaseError):
            await store.wait_for_result("r1", "wrong")
        polls = await asyncio.gather(
            *(store.wait_for_result("r1", lease) for _ in range(2))
        )
        assert [p.model_dump_json() for p in polls] == [expected, expected]
    finally:
        await store.close()
    assert not list(tmp_path.iterdir())


async def test_unwritable_result_storage_returns_a_terminal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "missing"))
    store = registry([], wait=1, lease=2)
    lease = await store.register("r1")
    try:
        await store.complete(
            "r1", RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
        )
        result = await store.wait_for_result("r1", lease)
        assert result.status is RolloutStatus.FAILURE
        assert result.err_category is RolloutErrorCategory.INTERNAL_ERROR
        assert await store.wait_for_result("r1", lease) == result
    finally:
        await store.close()


async def test_poll_in_flight_survives_retention_and_id_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: tmp_path
    )
    store = registry([], wait=1, lease=2)
    store.result_retention_sec = 0
    lease = await store.register("r1")
    first = RolloutResultResponse(
        rollout_id="r1", status=RolloutStatus.FAILURE, err_message="first run"
    )
    started = asyncio.Event()
    resume = Event()
    loop = asyncio.get_running_loop()
    read_bytes = Path.read_bytes

    def slow_read(path: Path) -> bytes:
        loop.call_soon_threadsafe(started.set)
        assert resume.wait(5)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", slow_read)

    async def finish_execution_after_poll_starts() -> None:
        await started.wait()

    await store.bind_task(
        "r1", asyncio.create_task(finish_execution_after_poll_starts())
    )
    try:
        await store.complete("r1", first)
        cleanup = store.entry("r1").cleanup_task
        assert cleanup is not None
        poll = asyncio.create_task(store.wait_for_result("r1", lease))
        await asyncio.wait_for(cleanup, timeout=5)
        next_lease = await store.register("r1")
        with pytest.raises(InvalidLeaseError):
            await store.wait_for_result("r1", lease)
        resume.set()
        assert await poll == first
        assert not list(tmp_path.iterdir())
        second = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
        await store.complete("r1", second)
        assert await store.wait_for_result("r1", next_lease) == second
    finally:
        resume.set()
        await store.close()


async def test_wait_returns_on_status_change() -> None:
    store = registry([], wait=1.0, lease=2.0)
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    waiter = asyncio.create_task(store.wait_for_result("r1", lease))
    await asyncio.sleep(0)
    await progress.set_status(RolloutStatus.GRADING)
    result = await asyncio.wait_for(waiter, timeout=0.1)
    assert result.status is RolloutStatus.GRADING
    await store.close()


async def test_wait_returns_when_the_agent_starts() -> None:
    store = registry([], wait=1.0, lease=2.0)
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.RUNNING)
    waiter = asyncio.create_task(store.wait_for_result("r1", lease))
    await asyncio.sleep(0)
    await progress.mark_agent_started(AGENT_STARTED_AT)
    result = await asyncio.wait_for(waiter, timeout=0.1)
    assert result.status is RolloutStatus.RUNNING
    assert result.agent_started_at == AGENT_STARTED_AT
    await store.close()


async def test_wait_timeout_does_not_cancel_shared_future() -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.GRADING)
    pending = await store.wait_for_result("r1", lease)
    assert pending.status is RolloutStatus.GRADING
    expected = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    await store.complete("r1", expected)
    assert await store.wait_for_result("r1", lease) == expected
    await store.close()


async def test_poll_renews_the_lease() -> None:
    cancelled: list[str] = []
    store = registry(cancelled, wait=0.005, lease=0.04)
    lease_token = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.RUNNING)
    for _ in range(3):
        await asyncio.sleep(0.02)
        result = await store.wait_for_result("r1", lease_token)
        assert result.status is RolloutStatus.RUNNING
    assert cancelled == []
    await store.close()


async def test_in_progress_response_carries_agent_started_at() -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.RUNNING)
    await progress.mark_agent_started(AGENT_STARTED_AT)
    response = await store.wait_for_result("r1", lease)
    assert response.status is RolloutStatus.RUNNING
    assert response.agent_started_at == AGENT_STARTED_AT
    await store.close()


async def test_in_progress_response_without_agent_start_is_null() -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.RUNNING)
    response = await store.wait_for_result("r1", lease)
    assert response.status is RolloutStatus.RUNNING
    assert response.agent_started_at is None
    await store.close()


async def test_mark_agent_started_keeps_first_timestamp() -> None:
    store = registry([])
    await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.mark_agent_started(AGENT_STARTED_AT)
    await progress.mark_agent_started(AGENT_STARTED_AT + timedelta(minutes=1))
    assert progress.agent_started_at == AGENT_STARTED_AT
    await store.close()


async def test_expiry_publishes_failure_and_cancels() -> None:
    cancelled: list[str] = []
    store = registry(cancelled, lease=0.02)
    lease = await store.register("r1")
    await asyncio.sleep(0.04)
    result = await store.wait_for_result("r1", lease)
    assert result.status is RolloutStatus.FAILURE
    assert result.err_category == "lease_expired"
    assert cancelled == ["r1"]
    await store.close()


@pytest.mark.parametrize(
    "started_at",
    [None, AGENT_STARTED_AT],
    ids=["before-agent-start", "after-agent-start"],
)
async def test_expiry_failure_reports_agent_start(started_at: datetime | None) -> None:
    store = registry([], lease=0.02)
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    if started_at is not None:
        await progress.mark_agent_started(started_at)
    await asyncio.sleep(0.04)
    result = await store.wait_for_result("r1", lease)
    assert result.err_category is RolloutErrorCategory.LEASE_EXPIRED
    assert result.agent_started_at == started_at
    await store.close()


async def test_poll_after_lease_deadline_reports_agent_start(monkeypatch) -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.mark_agent_started(AGENT_STARTED_AT)
    monkeypatch.setattr(store.leases, "renew", lambda *_args: False)
    result = await store.wait_for_result("r1", lease)
    assert result.err_category is RolloutErrorCategory.LEASE_EXPIRED
    assert result.agent_started_at == AGENT_STARTED_AT
    await store.close()


async def test_completion_after_lease_deadline_reports_agent_start(
    monkeypatch,
) -> None:
    store = registry([])
    await store.register("r1")
    progress = await store.get_progress("r1")
    await progress.mark_agent_started(AGENT_STARTED_AT)
    monkeypatch.setattr(store.leases, "expired", lambda _rollout_id: True)
    finished = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    assert not await store.complete("r1", finished)
    result = store.entry("r1").result.result()
    assert result.err_category is RolloutErrorCategory.LEASE_EXPIRED
    assert result.agent_started_at == AGENT_STARTED_AT
    await store.close()


@pytest.mark.parametrize("expired", [False, True])
async def test_published_by_run_excludes_lease_expiry(monkeypatch, expired) -> None:
    store = registry([])
    lease = await store.register("r1")
    with pytest.raises(RolloutPendingError):
        await store.published_by_run("r1", lease)
    with pytest.raises(InvalidLeaseError):
        await store.published_by_run("r1", "wrong")
    monkeypatch.setattr(store.leases, "expired", lambda _rollout_id: expired)
    finished = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    await store.complete("r1", finished)
    assert await store.published_by_run("r1", lease) is not expired
    await store.close()


async def test_unknown_invalid_and_duplicate_are_distinct() -> None:
    store = registry([])
    with pytest.raises(UnknownRolloutError):
        await store.wait_for_result("missing", "lease")
    await store.register("r1")
    with pytest.raises(InvalidLeaseError):
        await store.wait_for_result("r1", "wrong")
    with pytest.raises(DuplicateRolloutError):
        await store.register("r1")
    await store.close()


async def test_cancel_before_execution_publishes_a_terminal_result() -> None:
    store = registry([])
    lease = await store.register("r1")
    task = asyncio.create_task(asyncio.sleep(30))
    await store.bind_task("r1", task)
    assert await store.cancel(ids=["r1"]) == {"r1": "cancelled_queued"}
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await store.wait_for_result("r1", lease)
    assert result.status is RolloutStatus.CANCELLED
    await store.close()


@pytest.mark.parametrize(
    "started_at",
    [None, AGENT_STARTED_AT],
    ids=["before-agent-start", "after-agent-start"],
)
async def test_task_cancellation_reports_agent_start(
    started_at: datetime | None,
) -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    if started_at is not None:
        await progress.mark_agent_started(started_at)
    task = asyncio.create_task(asyncio.sleep(30))
    await store.bind_task("r1", task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await store.wait_for_result("r1", lease)
    assert result.status is RolloutStatus.CANCELLED
    assert result.agent_started_at == started_at
    await store.close()


async def test_cancel_before_binding_is_replayed_without_executing() -> None:
    cancelled: list[str] = []
    executed: list[str] = []
    store = registry(cancelled)
    lease = await store.register("r1")
    await store.cancel(ids=["r1"])
    await store.cancel(ids=["r1"])

    async def execute() -> None:
        executed.append("r1")

    task = asyncio.create_task(execute())
    await store.bind_task("r1", task)
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await store.wait_for_result("r1", lease)
    assert result.status is RolloutStatus.CANCELLED
    assert executed == []
    assert cancelled == ["r1"]
    await store.close()


async def test_backend_child_cancellation_does_not_interrupt_cleanup() -> None:
    started = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def execute() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish_cleanup.wait()
            cleaned.set()

    child = asyncio.create_task(execute())

    async def parent() -> None:
        await child

    task = asyncio.create_task(parent())
    store = registry([])

    def cancel_child(_rollout_id: str) -> str:
        child.cancel()
        return "cancelled_queued"

    store.cancel_rollout = cancel_child
    await store.register("r1")
    await store.bind_task("r1", task)
    progress = await store.get_progress("r1")
    await progress.set_status(RolloutStatus.RUNNING)
    await started.wait()
    assert await store.cancel(ids=["r1"]) == {"r1": "cancelled_queued"}
    await cleaning.wait()
    assert await store.cancel(ids=["r1"]) == {"r1": "cancelled_queued"}
    finish_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child.cancelling() == 1
    assert cleaned.is_set()
    await store.close()


async def test_expired_rollout_id_is_retained_until_execution_finishes() -> None:
    store = registry([], wait=0.005, lease=0.01)
    store.result_retention_sec = 0.0
    started = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    lease = await store.register("r1")

    async def execute() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish_cleanup.wait()
            assert not await store.complete(
                "r1",
                RolloutResultResponse(rollout_id="r1", status=RolloutStatus.CANCELLED),
            )

    task = asyncio.create_task(execute())
    await store.bind_task("r1", task)
    try:
        await started.wait()
        async with asyncio.timeout(1.0):
            await cleaning.wait()
        # Let the zero-retention cleanup run while execution is still unwinding.
        await asyncio.sleep(0.01)
        with pytest.raises(DuplicateRolloutError):
            await store.register("r1")
        result = await store.wait_for_result("r1", lease)
        assert result.err_category == "lease_expired"
        cleanup = store.entry("r1").cleanup_task
        assert cleanup is not None
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await cleanup
        await store.register("r1")
    finally:
        finish_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


async def test_expired_poll_before_binding_cancels_execution(monkeypatch) -> None:
    cancelled: list[str] = []
    executed: list[str] = []
    store = registry(cancelled)
    lease = await store.register("r1")
    monkeypatch.setattr(store.leases, "renew", lambda *_args: False)
    result = await store.wait_for_result("r1", lease)
    assert result.err_category == "lease_expired"

    async def execute() -> None:
        executed.append("r1")

    task = asyncio.create_task(execute())
    await store.bind_task("r1", task)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == ["r1"]
    assert executed == []
    await store.close()


async def test_task_crash_publishes_failure_result() -> None:
    store = registry([])
    lease = await store.register("r1")

    async def execute() -> None:
        raise RuntimeError("backend worker crashed")

    task = asyncio.create_task(execute())
    await store.bind_task("r1", task)
    with pytest.raises(RuntimeError, match="backend worker crashed"):
        await task
    result = await store.wait_for_result("r1", lease)
    assert result.status is RolloutStatus.FAILURE
    assert result.err_category is RolloutErrorCategory.INTERNAL_ERROR
    assert result.err_message == "rollout task crashed"
    await store.close()


@pytest.mark.parametrize(
    "started_at",
    [None, AGENT_STARTED_AT],
    ids=["before-agent-start", "after-agent-start"],
)
async def test_task_crash_failure_reports_agent_start(
    started_at: datetime | None,
) -> None:
    store = registry([])
    lease = await store.register("r1")
    progress = await store.get_progress("r1")
    if started_at is not None:
        await progress.mark_agent_started(started_at)

    async def execute() -> None:
        raise RuntimeError("backend worker crashed")

    task = asyncio.create_task(execute())
    await store.bind_task("r1", task)
    with pytest.raises(RuntimeError, match="backend worker crashed"):
        await task
    result = await store.wait_for_result("r1", lease)
    assert result.err_category is RolloutErrorCategory.INTERNAL_ERROR
    assert result.agent_started_at == started_at
    await store.close()


def test_crash_failure_response() -> None:
    response = RolloutFutureRegistry.crash_failure("r1", None)
    assert response.rollout_id == "r1"
    assert response.status is RolloutStatus.FAILURE
    assert response.err_category is RolloutErrorCategory.INTERNAL_ERROR
    assert response.err_message == "rollout task crashed"


async def test_retention_cleanup_never_marks_an_id_reused_after_discard(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: tmp_path
    )
    (tmp_path / "r1").mkdir()
    store = registry([])
    store.result_retention_sec = 0.01
    await store.register("r1")
    await store.complete(
        "r1", RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    )
    cleanup = store.entry("r1").cleanup_task
    assert cleanup is not None
    await store.discard("r1")
    await store.register("r1")

    await cleanup

    assert not store.entry("r1").result.done()
    assert not (tmp_path / "r1" / ".osmosis-released").exists()
    await store.close()

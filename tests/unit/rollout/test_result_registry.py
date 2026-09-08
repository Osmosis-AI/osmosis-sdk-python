from __future__ import annotations

import asyncio

import pytest

from osmosis_ai.rollout.server.lease import InvalidLeaseError
from osmosis_ai.rollout.server.result_registry import (
    DuplicateRolloutError,
    RolloutFutureRegistry,
    UnknownRolloutError,
)
from osmosis_ai.rollout.types import RolloutResultResponse, RolloutStatus


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
    waiter = asyncio.create_task(
        store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
    )
    await asyncio.sleep(0)
    expected = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    assert await store.complete("r1", expected)
    assert await waiter == expected
    await store.close()


async def test_wait_timeout_does_not_cancel_shared_future() -> None:
    store = registry([])
    lease = await store.register("r1")
    pending = await store.wait_for_result("r1", lease, lambda: RolloutStatus.GRADING)
    assert pending.status is RolloutStatus.GRADING
    expected = RolloutResultResponse(rollout_id="r1", status=RolloutStatus.SUCCESS)
    await store.complete("r1", expected)
    assert (
        await store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
        == expected
    )
    await store.close()


async def test_poll_renews_the_lease() -> None:
    cancelled: list[str] = []
    store = registry(cancelled, wait=0.005, lease=0.04)
    lease_token = await store.register("r1")
    for _ in range(3):
        await asyncio.sleep(0.02)
        result = await store.wait_for_result(
            "r1", lease_token, lambda: RolloutStatus.RUNNING
        )
        assert result.status is RolloutStatus.RUNNING
    assert cancelled == []
    await store.close()


async def test_expiry_publishes_failure_and_cancels() -> None:
    cancelled: list[str] = []
    store = registry(cancelled, lease=0.02)
    lease = await store.register("r1")
    await asyncio.sleep(0.04)
    result = await store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
    assert result.status is RolloutStatus.FAILURE
    assert result.err_category == "lease_expired"
    assert cancelled == ["r1"]
    await store.close()


async def test_unknown_invalid_and_duplicate_are_distinct() -> None:
    store = registry([])
    with pytest.raises(UnknownRolloutError):
        await store.wait_for_result("missing", "lease", lambda: RolloutStatus.RUNNING)
    await store.register("r1")
    with pytest.raises(InvalidLeaseError):
        await store.wait_for_result("r1", "wrong", lambda: RolloutStatus.RUNNING)
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
    result = await store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
    assert result.status is RolloutStatus.CANCELLED
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
    result = await store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
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
    await store.set_status("r1", RolloutStatus.RUNNING)
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
        result = await store.wait_for_result("r1", lease, lambda: RolloutStatus.RUNNING)
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
    result = await store.wait_for_result("r1", lease, lambda: RolloutStatus.QUEUED)
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

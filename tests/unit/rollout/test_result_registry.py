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

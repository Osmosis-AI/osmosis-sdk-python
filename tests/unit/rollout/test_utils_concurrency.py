"""Tests for osmosis_ai.rollout.utils.concurrency."""

import asyncio

import pytest

from osmosis_ai.rollout.utils.concurrency import ConcurrencyLimiter


class TestConcurrencyLimiter:
    async def test_unlimited_concurrency(self):
        limiter = ConcurrencyLimiter(max_concurrent=None)
        assert limiter.snapshot() == {
            "max_concurrent": None,
            "queued": 0,
            "running": 0,
        }

        async with limiter.acquire():
            assert limiter.running == 1
        assert limiter.running == 0

    async def test_limited_concurrency_serializes(self):
        limiter = ConcurrencyLimiter(max_concurrent=1)
        order: list[int] = []

        async def task(n: int) -> None:
            async with limiter.acquire():
                order.append(n)
                await asyncio.sleep(0.01)

        await asyncio.gather(task(1), task(2))
        assert sorted(order) == [1, 2]

    async def test_snapshot_reflects_state(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        assert limiter.snapshot()["running"] == 0

        async with limiter.acquire():
            assert limiter.snapshot()["running"] == 1
        assert limiter.snapshot()["running"] == 0

    async def test_running_decremented_on_exception(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        with pytest.raises(RuntimeError):
            async with limiter.acquire():
                raise RuntimeError("boom")
        assert limiter.running == 0

    async def test_unlimited_running_decremented_on_exception(self):
        limiter = ConcurrencyLimiter(max_concurrent=None)
        with pytest.raises(RuntimeError):
            async with limiter.acquire():
                raise RuntimeError("boom")
        assert limiter.running == 0

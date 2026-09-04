from __future__ import annotations

import asyncio

import pytest

from osmosis_ai.rollout.server.lease import (
    InvalidLeaseError,
    LeaseManager,
    UnknownLeaseError,
)


async def test_renew_extends_the_deadline() -> None:
    expired = asyncio.Event()

    async def on_expired(_rollout_id: str) -> None:
        expired.set()

    leases = LeaseManager(timeout_sec=0.04, on_expired=on_expired)
    token = leases.register("r1")
    await asyncio.sleep(0.02)
    assert leases.renew("r1", token)
    await asyncio.sleep(0.03)
    assert not expired.is_set()
    await asyncio.wait_for(expired.wait(), timeout=0.1)
    await leases.close()


async def test_finished_lease_stays_authenticatable_without_expiring() -> None:
    expired: list[str] = []

    async def on_expired(rollout_id: str) -> None:
        expired.append(rollout_id)

    leases = LeaseManager(timeout_sec=0.01, on_expired=on_expired)
    token = leases.register("r1")
    leases.finish("r1")
    await asyncio.sleep(0.02)
    leases.authenticate("r1", token)
    assert expired == []
    await leases.close()


async def test_invalid_and_unknown_leases_are_distinct() -> None:
    async def on_expired(_rollout_id: str) -> None:
        return None

    leases = LeaseManager(timeout_sec=1.0, on_expired=on_expired)
    leases.register("r1")
    with pytest.raises(InvalidLeaseError):
        leases.authenticate("r1", "wrong")
    with pytest.raises(UnknownLeaseError):
        leases.authenticate("missing", "token")
    await leases.close()


@pytest.mark.parametrize("timeout_sec", [0.0, -1.0, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(timeout_sec: float) -> None:
    async def on_expired(_rollout_id: str) -> None:
        return None

    with pytest.raises(ValueError):
        LeaseManager(timeout_sec=timeout_sec, on_expired=on_expired)

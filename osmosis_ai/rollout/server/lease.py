from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class UnknownLeaseError(KeyError):
    pass


class InvalidLeaseError(PermissionError):
    pass


@dataclass
class PollingLease:
    digest: bytes
    deadline: float
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    watcher: asyncio.Task[None] | None = None
    active: bool = True


class LeaseManager:
    def __init__(
        self,
        *,
        timeout_sec: float,
        on_expired: Callable[[str], Awaitable[None]],
    ) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and greater than zero")
        self._timeout_sec = timeout_sec
        self._on_expired = on_expired
        self._leases: dict[str, PollingLease] = {}

    def register(self, rollout_id: str, token: str) -> None:
        if rollout_id in self._leases:
            raise ValueError(f"lease already registered for {rollout_id}")
        lease = PollingLease(
            digest=self._digest(token),
            deadline=asyncio.get_running_loop().time() + self._timeout_sec,
        )
        self._leases[rollout_id] = lease
        lease.watcher = asyncio.create_task(self._watch(rollout_id, lease))

    def authenticate(self, rollout_id: str, token: str) -> None:
        self._authenticated_lease(rollout_id, token)

    def renew(self, rollout_id: str, token: str) -> bool:
        lease = self._authenticated_lease(rollout_id, token)
        now = asyncio.get_running_loop().time()
        if now >= lease.deadline:
            return False
        lease.deadline = now + self._timeout_sec
        lease.changed.set()
        return True

    def expired(self, rollout_id: str) -> bool:
        lease = self._lease(rollout_id)
        return asyncio.get_running_loop().time() >= lease.deadline

    def finish(self, rollout_id: str) -> None:
        lease = self._leases.get(rollout_id)
        if lease is not None:
            self._stop(lease)

    def remove(self, rollout_id: str) -> None:
        lease = self._leases.pop(rollout_id, None)
        if lease is not None:
            self._stop(lease)

    async def close(self) -> None:
        tasks = [
            lease.watcher
            for lease in self._leases.values()
            if lease.watcher is not None and not lease.watcher.done()
        ]
        for lease in self._leases.values():
            self._stop(lease)
        self._leases.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch(self, rollout_id: str, lease: PollingLease) -> None:
        try:
            while lease.active:
                delay = max(
                    0.0,
                    lease.deadline - asyncio.get_running_loop().time(),
                )
                lease.changed.clear()
                try:
                    await asyncio.wait_for(lease.changed.wait(), delay)
                except TimeoutError:
                    await self._on_expired(rollout_id)
        except asyncio.CancelledError:
            return

    def _authenticated_lease(self, rollout_id: str, token: str) -> PollingLease:
        lease = self._lease(rollout_id)
        if not secrets.compare_digest(lease.digest, self._digest(token)):
            raise InvalidLeaseError(rollout_id)
        return lease

    def _lease(self, rollout_id: str) -> PollingLease:
        try:
            return self._leases[rollout_id]
        except KeyError as exc:
            raise UnknownLeaseError(rollout_id) from exc

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode()).digest()

    @staticmethod
    def _stop(lease: PollingLease) -> None:
        lease.active = False
        lease.changed.set()
        current = asyncio.current_task()
        if lease.watcher is not None and lease.watcher is not current:
            lease.watcher.cancel()


__all__ = ["InvalidLeaseError", "LeaseManager", "UnknownLeaseError"]

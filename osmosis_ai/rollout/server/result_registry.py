from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from osmosis_ai.rollout.server.lease import LeaseManager
from osmosis_ai.rollout.types import (
    RolloutErrorCategory,
    RolloutResultResponse,
    RolloutStatus,
)

logger = logging.getLogger(__name__)


class UnknownRolloutError(KeyError):
    pass


class DuplicateRolloutError(ValueError):
    pass


@dataclass
class RolloutFuture:
    rollout_id: str
    result: asyncio.Future[RolloutResultResponse]
    status: RolloutStatus = RolloutStatus.QUEUED
    task: asyncio.Task[None] | None = None
    cleanup_task: asyncio.Task[None] | None = None


class RolloutFutureRegistry:
    def __init__(
        self,
        *,
        result_wait_timeout_sec: float,
        polling_lease_timeout_sec: float,
        result_retention_sec: float,
        cancel_rollout: Callable[[str], None],
    ) -> None:
        if not math.isfinite(result_wait_timeout_sec) or result_wait_timeout_sec <= 0:
            raise ValueError(
                "result_wait_timeout_sec must be finite and greater than zero"
            )
        if not math.isfinite(result_retention_sec) or result_retention_sec < 0:
            raise ValueError("result_retention_sec must be finite and non-negative")
        if polling_lease_timeout_sec <= result_wait_timeout_sec:
            raise ValueError(
                "polling_lease_timeout_sec must exceed result_wait_timeout_sec"
            )
        self.result_wait_timeout_sec = result_wait_timeout_sec
        self.result_retention_sec = result_retention_sec
        self.cancel_rollout = cancel_rollout
        self.entries: dict[str, RolloutFuture] = {}
        self.lock = asyncio.Lock()
        self.leases = LeaseManager(
            timeout_sec=polling_lease_timeout_sec,
            on_expired=self.expire,
        )

    async def register(self, rollout_id: str, lease_token: str) -> None:
        async with self.lock:
            if rollout_id in self.entries:
                raise DuplicateRolloutError(rollout_id)
            entry = RolloutFuture(
                rollout_id=rollout_id,
                result=asyncio.get_running_loop().create_future(),
            )
            self.entries[rollout_id] = entry
            self.leases.register(rollout_id, lease_token)

    async def bind_task(self, rollout_id: str, task: asyncio.Task[None]) -> None:
        async with self.lock:
            entry = self.entry(rollout_id)
            entry.task = task

    async def discard(self, rollout_id: str) -> None:
        async with self.lock:
            entry = self.entries.pop(rollout_id, None)
            if entry is not None:
                self.leases.remove(rollout_id)

    async def set_status(self, rollout_id: str, status: RolloutStatus) -> None:
        async with self.lock:
            entry = self.entry(rollout_id)
            if not entry.result.done():
                entry.status = status

    async def complete(self, rollout_id: str, response: RolloutResultResponse) -> bool:
        task_to_cancel: asyncio.Task[None] | None = None
        expired = False
        async with self.lock:
            entry = self.entry(rollout_id)
            if entry.result.done():
                return False
            if self.leases.expired(rollout_id):
                response = self.lease_failure(rollout_id)
                task_to_cancel = entry.task
                expired = True
            self.finish(entry, response)
        if expired:
            self.cancel_execution(rollout_id, task_to_cancel)
        return not expired

    async def wait_for_result(
        self,
        rollout_id: str,
        lease_token: str,
        current_status: Callable[[], RolloutStatus],
    ) -> RolloutResultResponse:
        task_to_cancel: asyncio.Task[None] | None = None
        async with self.lock:
            entry = self.entry(rollout_id)
            if not entry.result.done():
                if not self.leases.renew(rollout_id, lease_token):
                    self.finish(entry, self.lease_failure(rollout_id))
                    task_to_cancel = entry.task
            else:
                self.leases.authenticate(rollout_id, lease_token)
            future = entry.result
        if task_to_cancel is not None:
            self.cancel_execution(rollout_id, task_to_cancel)
        if future.done():
            return future.result()
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), self.result_wait_timeout_sec
            )
        except TimeoutError:
            async with self.lock:
                entry = self.entry(rollout_id)
                self.leases.authenticate(rollout_id, lease_token)
                if entry.result.done():
                    return entry.result.result()
                status = current_status()
                entry.status = status
                return RolloutResultResponse(
                    rollout_id=rollout_id,
                    status=status,
                )

    async def expire(self, rollout_id: str) -> None:
        task_to_cancel: asyncio.Task[None] | None = None
        async with self.lock:
            entry = self.entries.get(rollout_id)
            if entry is None or entry.result.done():
                return
            if not self.leases.expired(rollout_id):
                return
            self.finish(entry, self.lease_failure(rollout_id))
            task_to_cancel = entry.task
        self.cancel_execution(rollout_id, task_to_cancel)

    async def close(self) -> None:
        async with self.lock:
            tasks = [
                entry.cleanup_task
                for entry in self.entries.values()
                if entry.cleanup_task is not None and not entry.cleanup_task.done()
            ]
        await self.leases.close()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def finish(self, entry: RolloutFuture, response: RolloutResultResponse) -> None:
        entry.status = response.status
        entry.result.set_result(response)
        self.leases.finish(entry.rollout_id)
        entry.cleanup_task = asyncio.create_task(self.remove_after_retention(entry))

    async def remove_after_retention(self, entry: RolloutFuture) -> None:
        try:
            await asyncio.sleep(self.result_retention_sec)
            async with self.lock:
                if self.entries.get(entry.rollout_id) is entry:
                    self.entries.pop(entry.rollout_id, None)
                    self.leases.remove(entry.rollout_id)
        except asyncio.CancelledError:
            return

    def cancel_execution(
        self, rollout_id: str, task: asyncio.Task[None] | None
    ) -> None:
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        try:
            self.cancel_rollout(rollout_id)
        except Exception:
            logger.exception("Backend cancellation failed for rollout %s", rollout_id)

    def entry(self, rollout_id: str) -> RolloutFuture:
        entry = self.entries.get(rollout_id)
        if entry is None:
            raise UnknownRolloutError(rollout_id)
        return entry

    @staticmethod
    def lease_failure(rollout_id: str) -> RolloutResultResponse:
        return RolloutResultResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.FAILURE,
            err_message="polling lease expired",
            err_category=RolloutErrorCategory.LEASE_EXPIRED,
        )


__all__ = [
    "DuplicateRolloutError",
    "RolloutFuture",
    "RolloutFutureRegistry",
    "UnknownRolloutError",
]

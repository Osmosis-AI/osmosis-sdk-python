from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from osmosis_ai.rollout.context import RolloutProgress
from osmosis_ai.rollout.server.lease import LeaseManager
from osmosis_ai.rollout.types import (
    RolloutErrorCategory,
    RolloutResultResponse,
    RolloutStatus,
)

logger: logging.Logger = logging.getLogger(__name__)


class UnknownRolloutError(KeyError):
    pass


class DuplicateRolloutError(ValueError):
    pass


@dataclass
class RolloutFuture:
    rollout_id: str
    result: asyncio.Future[RolloutResultResponse]
    progress: RolloutProgress
    task: asyncio.Task[None] | None = None
    cleanup_task: asyncio.Task[None] | None = None
    cancel_result_task: asyncio.Task[None] | None = None
    cancel_disposition: str | None = None
    cancel_task_on_bind: bool = False


class RolloutFutureRegistry:
    def __init__(
        self,
        *,
        result_wait_timeout_sec: float,
        polling_lease_timeout_sec: float,
        result_retention_sec: float,
        cancel_rollout: Callable[[str], str | None],
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
        self.cancel_rollout: Callable[[str], str | None] = cancel_rollout
        self.entries: dict[str, RolloutFuture] = {}
        self.lock: asyncio.Lock = asyncio.Lock()
        self.leases: LeaseManager = LeaseManager(
            timeout_sec=polling_lease_timeout_sec,
            on_expired=self.expire,
        )

    async def register(self, rollout_id: str) -> str:
        async with self.lock:
            if rollout_id in self.entries:
                raise DuplicateRolloutError(rollout_id)
            entry = RolloutFuture(
                rollout_id=rollout_id,
                result=asyncio.get_running_loop().create_future(),
                progress=RolloutProgress(changed=asyncio.Condition(self.lock)),
            )
            self.entries[rollout_id] = entry
            return self.leases.register(rollout_id)

    async def bind_task(self, rollout_id: str, task: asyncio.Task[None]) -> None:
        async with self.lock:
            entry = self.entry(rollout_id)
            entry.task = task
            if entry.cancel_task_on_bind and not task.cancelling():
                task.cancel()

        async def publish_cancelled() -> None:
            async with self.lock:
                if self.entries.get(rollout_id) is entry and not entry.result.done():
                    self.finish(
                        entry,
                        RolloutResultResponse(
                            rollout_id=rollout_id, status=RolloutStatus.CANCELLED
                        ),
                    )

        def complete_cancelled(done: asyncio.Task[None]) -> None:
            if done.cancelled():
                entry.cancel_result_task = asyncio.create_task(publish_cancelled())

        task.add_done_callback(complete_cancelled)

    async def discard(self, rollout_id: str) -> None:
        async with self.lock:
            entry = self.entries.pop(rollout_id, None)
            if entry is not None:
                self.leases.remove(rollout_id)

    async def get_progress(self, rollout_id: str) -> RolloutProgress:
        async with self.lock:
            return self.entry(rollout_id).progress

    async def complete(self, rollout_id: str, response: RolloutResultResponse) -> bool:
        expired = False
        async with self.lock:
            entry = self.entry(rollout_id)
            if entry.result.done():
                return False
            if self.leases.expired(rollout_id):
                response = self.lease_failure(rollout_id)
                expired = True
            self.finish(entry, response)
        if expired:
            self.cancel_execution(entry)
        return not expired

    async def wait_for_result(
        self,
        rollout_id: str,
        lease_token: str,
    ) -> RolloutResultResponse:
        expired = False
        async with self.lock:
            entry = self.entry(rollout_id)
            if not entry.result.done():
                if not self.leases.renew(rollout_id, lease_token):
                    self.finish(entry, self.lease_failure(rollout_id))
                    expired = True
            else:
                self.leases.authenticate(rollout_id, lease_token)
            future = entry.result
            progress = entry.progress
        if expired:
            self.cancel_execution(entry)
        if future.done():
            return future.result()
        try:
            async with asyncio.timeout(self.result_wait_timeout_sec):
                await progress.wait_for_status_change()
        except TimeoutError:
            pass
        async with self.lock:
            entry = self.entry(rollout_id)
            self.leases.authenticate(rollout_id, lease_token)
            if entry.result.done():
                return entry.result.result()
            return RolloutResultResponse(
                rollout_id=rollout_id,
                status=entry.progress.status,
            )

    async def expire(self, rollout_id: str) -> None:
        async with self.lock:
            entry = self.entries.get(rollout_id)
            if entry is None or entry.result.done():
                return
            if not self.leases.expired(rollout_id):
                return
            self.finish(entry, self.lease_failure(rollout_id))
        self.cancel_execution(entry)

    async def close(self) -> None:
        async with self.lock:
            tasks = [
                task
                for entry in self.entries.values()
                for task in (entry.cleanup_task, entry.cancel_result_task)
                if task is not None and not task.done()
            ]
        await self.leases.close()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def finish(self, entry: RolloutFuture, response: RolloutResultResponse) -> None:
        entry.result.set_result(response)
        entry.progress.status = response.status
        entry.progress.changed.notify_all()
        self.leases.finish(entry.rollout_id)
        entry.cleanup_task = asyncio.create_task(self.remove_after_retention(entry))

    async def remove_after_retention(self, entry: RolloutFuture) -> None:
        try:
            await asyncio.sleep(self.result_retention_sec)
            # Lease expiry publishes before execution cleanup finishes. Keep
            # the id reserved so late completion cannot affect a new rollout.
            if entry.task is not None:
                await asyncio.wait({entry.task})
            async with self.lock:
                if self.entries.get(entry.rollout_id) is entry:
                    self.entries.pop(entry.rollout_id, None)
                    self.leases.remove(entry.rollout_id)
        except asyncio.CancelledError:
            return

    async def cancel(
        self,
        *,
        ids: Sequence[str] | None = None,
        prefix: str | None = None,
        all: bool = False,
    ) -> dict[str, str]:
        async with self.lock:
            selected = (
                list(self.entries)
                if all
                else [rid for rid in self.entries if rid.startswith(prefix)]
                if prefix is not None
                else list(ids or [])
            )
            dispositions: dict[str, str] = {}
            for rollout_id in selected:
                entry = self.entries.get(rollout_id)
                if entry is None or entry.result.done():
                    dispositions[rollout_id] = "not_found"
                    continue
                dispositions[rollout_id] = self.cancel_execution(entry)
            return dispositions

    def cancel_execution(self, entry: RolloutFuture) -> str:
        if entry.cancel_disposition is not None:
            return entry.cancel_disposition
        entry.cancel_disposition = (
            "cancelled_queued"
            if entry.progress.status is RolloutStatus.QUEUED
            else "cancelled_running"
        )
        try:
            # A backend may cancel a child task that execute() is awaiting.
            # Cancelling its parent too would interrupt the child's cleanup.
            disposition = self.cancel_rollout(entry.rollout_id)
            if disposition in {"cancelled_queued", "cancelled_running", "not_found"}:
                entry.cancel_disposition = disposition
                return disposition
        except Exception:
            logger.exception(
                "Backend cancellation failed for rollout %s", entry.rollout_id
            )
        task = entry.task
        if task is None:
            entry.cancel_task_on_bind = True
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
            and not task.cancelling()
        ):
            task.cancel()
        return entry.cancel_disposition

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

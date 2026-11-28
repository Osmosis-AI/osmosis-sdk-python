from __future__ import annotations

import asyncio
import logging
import math
import tempfile
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from osmosis_ai.rollout.context import RolloutProgress
from osmosis_ai.rollout.server.lease import LeaseManager
from osmosis_ai.rollout.trajectory.save import mark_rollout_released
from osmosis_ai.rollout.types import (
    RolloutErrorCategory,
    RolloutResultResponse,
    RolloutStatus,
)

logger: logging.Logger = logging.getLogger(__name__)
_RESULT_JSON = TypeAdapter(RolloutResultResponse)


class UnknownRolloutError(KeyError):
    pass


class DuplicateRolloutError(ValueError):
    pass


class RolloutPendingError(RuntimeError):
    pass


class _ResultFile:
    """Private, immutable result shared by the registry and any in-flight polls."""

    def __init__(self, response: RolloutResultResponse) -> None:
        with tempfile.NamedTemporaryFile(
            prefix="osmosis-result-", delete=False
        ) as file:
            self._path = Path(file.name)
            # A poll may still be reading when retention expires. Unlink after
            # the registry and the last reader have both released this file.
            weakref.finalize(self, self._path.unlink, missing_ok=True)
            file.write(_RESULT_JSON.dump_json(response))

    def read(self) -> RolloutResultResponse:
        return _RESULT_JSON.validate_json(self._path.read_bytes())


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
    # Set only by complete(), after the run saved its files; stand-in results
    # (lease expiry, crash) leave it False.
    published_by_run: bool = False
    result_file: _ResultFile | None = None
    result_write_started: bool = False
    read_failure: RolloutResultResponse | None = None


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
                            rollout_id=rollout_id,
                            status=RolloutStatus.CANCELLED,
                            agent_started_at=entry.progress.agent_started_at,
                        ),
                    )

        async def publish_failure() -> None:
            async with self.lock:
                if self.entries.get(rollout_id) is entry and not entry.result.done():
                    self.finish(
                        entry,
                        self.crash_failure(rollout_id, entry.progress.agent_started_at),
                    )

        def complete_task(done: asyncio.Task[None]) -> None:
            if done.cancelled():
                entry.cancel_result_task = asyncio.create_task(publish_cancelled())
            elif done.exception() is not None:
                entry.cancel_result_task = asyncio.create_task(publish_failure())

        task.add_done_callback(complete_task)

    async def discard(self, rollout_id: str) -> None:
        async with self.lock:
            entry = self.entries.pop(rollout_id, None)
            if entry is not None:
                entry.result_file = None
                self.leases.remove(rollout_id)

    async def published_by_run(self, rollout_id: str, lease_token: str) -> bool:
        async with self.lock:
            entry = self.entry(rollout_id)
            self.leases.authenticate(rollout_id, lease_token)
            if not entry.result.done():
                raise RolloutPendingError(rollout_id)
            return entry.published_by_run

    async def get_progress(self, rollout_id: str) -> RolloutProgress:
        async with self.lock:
            return self.entry(rollout_id).progress

    async def complete(self, rollout_id: str, response: RolloutResultResponse) -> bool:
        entry = self.entry(rollout_id)
        if entry.result.done() or entry.result_write_started:
            return False
        entry.result_write_started = True
        publication = asyncio.create_task(self._store_and_finish(entry, response))
        # Keep execution alive until the file and its terminal state are published.
        # Otherwise its cancellation callback can replace the backend's result.
        cancelled = False
        while not publication.done():
            try:
                await asyncio.shield(publication)
            except asyncio.CancelledError:
                cancelled = True
        published = publication.result()
        if cancelled:
            raise asyncio.CancelledError
        return published

    async def _store_and_finish(
        self, entry: RolloutFuture, response: RolloutResultResponse
    ) -> bool:
        rollout_id = entry.rollout_id
        result_file = None
        try:
            result_file = await asyncio.to_thread(_ResultFile, response)
        except (OSError, ValueError):
            logger.exception("Could not store result for rollout %s", rollout_id)
            terminal = RolloutResultResponse(
                rollout_id=rollout_id,
                status=RolloutStatus.FAILURE,
                err_message="could not store rollout result",
                err_category=RolloutErrorCategory.INTERNAL_ERROR,
                agent_started_at=response.agent_started_at,
            )
        else:
            # Previews and observability only need terminal metadata.
            terminal = RolloutResultResponse(
                rollout_id=rollout_id,
                status=response.status,
                err_category=response.err_category,
                agent_started_at=response.agent_started_at,
            )
        async with self.lock:
            if self.entries.get(rollout_id) is not entry or entry.result.done():
                return False
            expired = self.leases.expired(rollout_id)
            if expired:
                terminal = self.lease_failure(
                    rollout_id, entry.progress.agent_started_at
                )
            else:
                entry.result_file = result_file
            entry.published_by_run = not expired
            self.finish(entry, terminal)
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
                    self.finish(
                        entry,
                        self.lease_failure(rollout_id, entry.progress.agent_started_at),
                    )
                    expired = True
            else:
                self.leases.authenticate(rollout_id, lease_token)
        if expired:
            self.cancel_execution(entry)
        if not entry.result.done():
            try:
                async with asyncio.timeout(self.result_wait_timeout_sec):
                    await entry.progress.wait_for_change()
                    if entry.result_write_started:
                        # A phase change may have woken this poll just as the
                        # backend finished. Give the write the rest of this poll.
                        await asyncio.shield(entry.result)
            except TimeoutError:
                pass
            async with self.lock:
                entry = self.entry(rollout_id)
                self.leases.authenticate(rollout_id, lease_token)
                if not entry.result.done():
                    return RolloutResultResponse(
                        rollout_id=rollout_id,
                        status=entry.progress.status,
                        agent_started_at=entry.progress.agent_started_at,
                    )
        if entry.result_file is not None:
            try:
                return await asyncio.to_thread(entry.result_file.read)
            except ValidationError:
                # The published file is immutable. Retrying these bytes cannot
                # recover the sample. Keep the failure available to later polls.
                logger.error("Invalid stored result for rollout %s", rollout_id)
                entry.read_failure = RolloutResultResponse(
                    rollout_id=rollout_id,
                    status=RolloutStatus.FAILURE,
                    err_message="stored rollout result is invalid",
                    err_category=RolloutErrorCategory.INTERNAL_ERROR,
                    agent_started_at=entry.result.result().agent_started_at,
                )
                entry.result_file = None
        return entry.read_failure or entry.result.result()

    async def expire(self, rollout_id: str) -> None:
        async with self.lock:
            entry = self.entries.get(rollout_id)
            if entry is None or entry.result.done():
                return
            if not self.leases.expired(rollout_id):
                return
            self.finish(
                entry, self.lease_failure(rollout_id, entry.progress.agent_started_at)
            )
        self.cancel_execution(entry)

    async def close(self) -> None:
        async with self.lock:
            tasks = [
                task
                for entry in self.entries.values()
                for task in (entry.cleanup_task, entry.cancel_result_task)
                if task is not None and not task.done()
            ]
            for entry in self.entries.values():
                entry.result_file = None
            self.entries.clear()
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
                if self.entries.get(entry.rollout_id) is not entry:
                    return
            # Mark before freeing the id, so a rollout reusing it starts after the
            # marker and removes it. Unlocked: the artifact root may be an S3 mount.
            await mark_rollout_released(entry.rollout_id)
            async with self.lock:
                if self.entries.get(entry.rollout_id) is entry:
                    entry.result_file = None
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
        if entry.result_write_started:
            return "not_found"
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
    def lease_failure(
        rollout_id: str, agent_started_at: datetime | None
    ) -> RolloutResultResponse:
        return RolloutResultResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.FAILURE,
            err_message="polling lease expired",
            err_category=RolloutErrorCategory.LEASE_EXPIRED,
            agent_started_at=agent_started_at,
        )

    @staticmethod
    def crash_failure(
        rollout_id: str, agent_started_at: datetime | None
    ) -> RolloutResultResponse:
        return RolloutResultResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.FAILURE,
            err_message="rollout task crashed",
            err_category=RolloutErrorCategory.INTERNAL_ERROR,
            agent_started_at=agent_started_at,
        )


__all__ = [
    "DuplicateRolloutError",
    "RolloutFuture",
    "RolloutFutureRegistry",
    "RolloutPendingError",
    "UnknownRolloutError",
]

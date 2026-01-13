"""Application state management for RolloutServer.

This module provides the AppState class that manages concurrent rollout
tasks, idempotency checking, and cleanup of completed records.

Example:
    state = AppState(max_concurrent=100, record_ttl_seconds=3600)
    state.start_cleanup_task()

    # Check idempotency
    if state.is_duplicate(rollout_id):
        return cached_response

    # Track task
    state.mark_started(rollout_id, task)
    # ... task runs ...
    state.mark_completed(rollout_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

from osmosis_ai.rollout.config.settings import RolloutServerSettings, get_settings
from osmosis_ai.rollout.core.schemas import InitResponse

logger = logging.getLogger(__name__)


class AppState:
    """Shared state for the FastAPI application.

    Manages concurrent rollout tasks, idempotency checking, and
    automatic cleanup of completed rollout records.

    Features:
        - Concurrency control via semaphore
        - Idempotency checking for duplicate requests
        - Automatic cleanup of old completion records

    Attributes:
        settings: Server settings (RolloutServerSettings instance).
        rollout_tasks: Dictionary of active rollout tasks by idempotency key.
        completed_rollouts: Dictionary of completion times by idempotency key.
        semaphore: Concurrency control semaphore.
        record_ttl: Time to live for completed records in seconds.

    Example:
        state = AppState(max_concurrent=100)
        state.start_cleanup_task()

        # In request handler
        if not state.is_duplicate(rollout_id):
            task = asyncio.create_task(run_rollout())
            state.mark_started(rollout_id, task)
    """

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        record_ttl_seconds: Optional[float] = None,
        cleanup_interval_seconds: Optional[float] = None,
        settings: Optional[RolloutServerSettings] = None,
        agent_loop_name: str = "default",
    ):
        """Initialize application state.

        Args:
            max_concurrent: Maximum concurrent rollouts. Defaults to settings.
            record_ttl_seconds: TTL for completed records. Defaults to settings.
            cleanup_interval_seconds: Cleanup check interval. Defaults to settings.
            settings: Server settings. Defaults to global settings.
            agent_loop_name: Name of the agent loop (for logging context).
        """
        if settings is None:
            settings = get_settings().server

        self.settings = settings
        self._max_concurrent = max_concurrent or settings.max_concurrent_rollouts
        self.record_ttl = record_ttl_seconds or settings.record_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds or settings.cleanup_interval_seconds
        self._agent_loop_name = agent_loop_name

        # NOTE: The "key" used throughout is the idempotency key for init requests:
        # - Prefer request.idempotency_key when provided
        # - Fallback to request.rollout_id when idempotency_key is missing
        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.completed_rollouts: Dict[str, float] = {}  # key -> completion_time
        # Cached init responses for idempotency (duplicate /v1/rollout/init requests)
        self._init_futures: Dict[str, asyncio.Future[InitResponse]] = {}
        self.semaphore = asyncio.Semaphore(self._max_concurrent)
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_or_create_init_future(
        self, key: str
    ) -> Tuple[asyncio.Future[InitResponse], bool]:
        """Get or create the init future for a given idempotency key.

        This is used to provide true idempotency for /v1/rollout/init:
        - The first request creates a future and becomes the "leader"
        - Duplicate requests await the same future and return the same InitResponse

        Returns:
            (future, created) where created=True means caller should compute tools
            and resolve the future.
        """
        existing = self._init_futures.get(key)
        if existing is not None:
            return existing, False

        fut: asyncio.Future[InitResponse] = asyncio.get_running_loop().create_future()
        self._init_futures[key] = fut
        return fut, True

    def clear_init_record(self, key: str) -> None:
        """Remove any cached init future/response for a given idempotency key."""
        self._init_futures.pop(key, None)

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task.

        Should be called during application startup.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cleanup task started: interval=%.1fs", self._cleanup_interval)

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task.

        Should be called during application shutdown.
        """
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up completed rollout records."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self._prune_completed_records()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error: %s", str(e))

    def _prune_completed_records(self) -> None:
        """Remove completed rollout records older than TTL."""
        now = time.monotonic()
        expired = [
            key
            for key, completed_at in self.completed_rollouts.items()
            if now - completed_at > self.record_ttl
        ]
        for key in expired:
            self.completed_rollouts.pop(key, None)
            self._init_futures.pop(key, None)
        if expired:
            logger.debug("Pruned %d completed records", len(expired))

    def is_duplicate(self, key: str) -> bool:
        """Check if this idempotency key is already running or recently completed.

        Used for idempotency - duplicate requests get the same response
        without starting a new rollout.

        Args:
            key: The idempotency key to check (or rollout_id if no idempotency_key).

        Returns:
            True if the rollout is running or recently completed.
        """
        return (
            key in self.rollout_tasks
            or key in self.completed_rollouts
            or key in self._init_futures
        )

    def mark_started(self, key: str, task: asyncio.Task) -> None:
        """Mark a rollout as started.

        Args:
            key: The idempotency key (or rollout_id if no idempotency_key).
            task: The asyncio task running the rollout.
        """
        self.rollout_tasks[key] = task

    def mark_completed(self, key: str) -> None:
        """Mark a rollout as completed.

        Removes from active tasks and adds to completed records
        for idempotency checking.

        Args:
            key: The idempotency key (or rollout_id if no idempotency_key).
        """
        self.rollout_tasks.pop(key, None)
        self.completed_rollouts[key] = time.monotonic()

    async def cancel_all(self) -> None:
        """Cancel all running rollout tasks.

        Should be called during application shutdown to gracefully
        stop all in-progress rollouts.
        """
        if not self.rollout_tasks:
            return

        logger.info("Cancelling all rollouts: count=%d", len(self.rollout_tasks))
        for task in self.rollout_tasks.values():
            task.cancel()

        results = await asyncio.gather(
            *self.rollout_tasks.values(),
            return_exceptions=True,
        )

        cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
        errors = sum(1 for r in results if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError))
        logger.info("All rollouts cancelled: cancelled=%d, errors=%d", cancelled, errors)

        # Best-effort cleanup to avoid leaving stale tasks around.
        self.rollout_tasks.clear()

    @property
    def active_count(self) -> int:
        """Get the number of active rollouts."""
        return len(self.rollout_tasks)

    @property
    def completed_count(self) -> int:
        """Get the number of recently completed rollouts."""
        return len(self.completed_rollouts)

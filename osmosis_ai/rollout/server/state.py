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
import time
from typing import Dict, Optional, Tuple

from osmosis_ai.rollout.config.settings import RolloutServerSettings, get_settings
from osmosis_ai.rollout.core.schemas import InitResponse
from osmosis_ai.rollout.observability.logging import get_logger
from osmosis_ai.rollout.observability.metrics import get_metrics

logger = get_logger(__name__)


class AppState:
    """Shared state for the FastAPI application.

    Manages concurrent rollout tasks, idempotency checking, and
    automatic cleanup of completed rollout records.

    Features:
        - Concurrency control via semaphore
        - Idempotency checking for duplicate requests
        - Automatic cleanup of old completion records
        - Metrics integration

    Attributes:
        rollout_tasks: Dictionary of active rollout tasks by ID.
        completed_rollouts: Dictionary of completion times by ID.
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
            agent_loop_name: Name of the agent loop for metrics labels.
        """
        if settings is None:
            settings = get_settings().server

        self._max_concurrent = max_concurrent or settings.max_concurrent_rollouts
        self.record_ttl = record_ttl_seconds or settings.record_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds or settings.cleanup_interval_seconds
        self._agent_loop_name = agent_loop_name

        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.completed_rollouts: Dict[str, float] = {}  # rollout_id -> completion_time
        # Cached init responses for idempotency (duplicate /v1/rollout/init requests)
        self._init_futures: Dict[str, asyncio.Future[InitResponse]] = {}
        self.semaphore = asyncio.Semaphore(self._max_concurrent)
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_or_create_init_future(
        self, rollout_id: str
    ) -> Tuple[asyncio.Future[InitResponse], bool]:
        """Get or create the init future for a rollout_id.

        This is used to provide true idempotency for /v1/rollout/init:
        - The first request creates a future and becomes the "leader"
        - Duplicate requests await the same future and return the same InitResponse

        Returns:
            (future, created) where created=True means caller should compute tools
            and resolve the future.
        """
        existing = self._init_futures.get(rollout_id)
        if existing is not None:
            return existing, False

        fut: asyncio.Future[InitResponse] = asyncio.get_running_loop().create_future()
        self._init_futures[rollout_id] = fut
        return fut, True

    def clear_init_record(self, rollout_id: str) -> None:
        """Remove any cached init future/response for a rollout_id."""
        self._init_futures.pop(rollout_id, None)

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task.

        Should be called during application startup.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("cleanup_task_started", interval_seconds=self._cleanup_interval)

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
            logger.info("cleanup_task_stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up completed rollout records."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self._prune_completed_records()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_loop_error", error=str(e))

    def _prune_completed_records(self) -> None:
        """Remove completed rollout records older than TTL."""
        now = time.monotonic()
        expired = [
            rid
            for rid, completed_at in self.completed_rollouts.items()
            if now - completed_at > self.record_ttl
        ]
        for rid in expired:
            self.completed_rollouts.pop(rid, None)
            self._init_futures.pop(rid, None)
        if expired:
            logger.debug("pruned_completed_records", count=len(expired))

    def is_duplicate(self, rollout_id: str) -> bool:
        """Check if this rollout is already running or recently completed.

        Used for idempotency - duplicate requests get the same response
        without starting a new rollout.

        Args:
            rollout_id: The rollout ID to check.

        Returns:
            True if the rollout is running or recently completed.
        """
        return (
            rollout_id in self.rollout_tasks
            or rollout_id in self.completed_rollouts
            or rollout_id in self._init_futures
        )

    def mark_started(self, rollout_id: str, task: asyncio.Task) -> None:
        """Mark a rollout as started.

        Args:
            rollout_id: The rollout ID.
            task: The asyncio task running the rollout.
        """
        self.rollout_tasks[rollout_id] = task

        # Update metrics
        metrics = get_metrics()
        metrics.rollouts_active.labels(agent_loop=self._agent_loop_name).inc()

    def mark_completed(self, rollout_id: str) -> None:
        """Mark a rollout as completed.

        Removes from active tasks and adds to completed records
        for idempotency checking.

        Args:
            rollout_id: The rollout ID.
        """
        self.rollout_tasks.pop(rollout_id, None)
        self.completed_rollouts[rollout_id] = time.monotonic()

        # Update metrics
        metrics = get_metrics()
        metrics.rollouts_active.labels(agent_loop=self._agent_loop_name).dec()

    async def cancel_all(self) -> None:
        """Cancel all running rollout tasks.

        Should be called during application shutdown to gracefully
        stop all in-progress rollouts.
        """
        if not self.rollout_tasks:
            return

        logger.info("cancelling_all_rollouts", count=len(self.rollout_tasks))
        for task in self.rollout_tasks.values():
            task.cancel()

        results = await asyncio.gather(
            *self.rollout_tasks.values(),
            return_exceptions=True,
        )

        cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
        errors = sum(1 for r in results if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError))
        logger.info("all_rollouts_cancelled", cancelled=cancelled, errors=errors)

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

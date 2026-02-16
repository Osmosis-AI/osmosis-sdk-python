# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.server.state (AppState)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

from osmosis_ai.rollout.config.settings import RolloutServerSettings
from osmosis_ai.rollout.server.state import AppState

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    max_concurrent: int = 10,
    record_ttl_seconds: float = 60.0,
    cleanup_interval_seconds: float = 300.0,
) -> AppState:
    """Create an AppState with explicit settings to avoid loading env vars."""
    settings = RolloutServerSettings(
        max_concurrent_rollouts=max_concurrent,
        record_ttl_seconds=record_ttl_seconds,
        cleanup_interval_seconds=cleanup_interval_seconds,
    )
    return AppState(
        max_concurrent=max_concurrent,
        record_ttl_seconds=record_ttl_seconds,
        cleanup_interval_seconds=cleanup_interval_seconds,
        settings=settings,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAppStateInit:
    """Tests for AppState initialization."""

    def test_initial_state_empty(self) -> None:
        """AppState starts with empty dicts and no cleanup task."""
        state = _make_state()
        assert state.rollout_tasks == {}
        assert state.completed_rollouts == {}
        assert state._init_futures == {}
        assert state._cleanup_task is None

    def test_active_count_starts_at_zero(self) -> None:
        """active_count property returns 0 when nothing is tracked."""
        state = _make_state()
        assert state.active_count == 0

    def test_completed_count_starts_at_zero(self) -> None:
        """completed_count property returns 0 when nothing is tracked."""
        state = _make_state()
        assert state.completed_count == 0

    def test_custom_max_concurrent(self) -> None:
        """max_concurrent is stored from constructor arg."""
        state = _make_state(max_concurrent=42)
        assert state._max_concurrent == 42

    def test_custom_record_ttl(self) -> None:
        """record_ttl is stored from constructor arg."""
        state = _make_state(record_ttl_seconds=120.0)
        assert state.record_ttl == 120.0

    def test_custom_cleanup_interval(self) -> None:
        """cleanup_interval is stored from constructor arg."""
        state = _make_state(cleanup_interval_seconds=30.0)
        assert state._cleanup_interval == 30.0

    def test_semaphore_limits_concurrency(self) -> None:
        """Semaphore is created with the max_concurrent value."""
        state = _make_state(max_concurrent=5)
        # asyncio.Semaphore stores the initial value as _value
        assert state.semaphore._value == 5


# =============================================================================
# get_or_create_init_future Tests
# =============================================================================


class TestGetOrCreateInitFuture:
    """Tests for AppState.get_or_create_init_future (deduplication)."""

    async def test_first_call_creates_future(self) -> None:
        """First call for a key creates a new future and returns created=True."""
        state = _make_state()
        future, created = state.get_or_create_init_future("rollout-1")
        assert created is True
        assert isinstance(future, asyncio.Future)

    async def test_second_call_returns_same_future(self) -> None:
        """Second call with same key returns the same future with created=False."""
        state = _make_state()
        future1, created1 = state.get_or_create_init_future("rollout-1")
        future2, created2 = state.get_or_create_init_future("rollout-1")

        assert created1 is True
        assert created2 is False
        assert future1 is future2

    async def test_different_keys_get_different_futures(self) -> None:
        """Different rollout_ids get independent futures."""
        state = _make_state()
        future_a, created_a = state.get_or_create_init_future("rollout-a")
        future_b, created_b = state.get_or_create_init_future("rollout-b")

        assert created_a is True
        assert created_b is True
        assert future_a is not future_b

    async def test_future_is_stored_internally(self) -> None:
        """The created future is stored in _init_futures dict."""
        state = _make_state()
        future, _ = state.get_or_create_init_future("rollout-x")
        assert "rollout-x" in state._init_futures
        assert state._init_futures["rollout-x"] is future


# =============================================================================
# clear_init_record Tests
# =============================================================================


class TestClearInitRecord:
    """Tests for AppState.clear_init_record."""

    async def test_clear_existing_record(self) -> None:
        """Clearing an existing init future removes it."""
        state = _make_state()
        state.get_or_create_init_future("rollout-1")
        assert "rollout-1" in state._init_futures

        state.clear_init_record("rollout-1")
        assert "rollout-1" not in state._init_futures

    def test_clear_nonexistent_record_no_error(self) -> None:
        """Clearing a non-existent key does not raise."""
        state = _make_state()
        # Should not raise
        state.clear_init_record("nonexistent")

    async def test_clear_only_affects_target_key(self) -> None:
        """Clearing one key leaves other keys intact."""
        state = _make_state()
        state.get_or_create_init_future("rollout-a")
        state.get_or_create_init_future("rollout-b")

        state.clear_init_record("rollout-a")
        assert "rollout-a" not in state._init_futures
        assert "rollout-b" in state._init_futures


# =============================================================================
# is_duplicate Tests
# =============================================================================


class TestIsDuplicate:
    """Tests for AppState.is_duplicate (three conditions)."""

    async def test_duplicate_when_active_future_exists(self) -> None:
        """Returns True when rollout_id has an active init future."""
        state = _make_state()
        state.get_or_create_init_future("rollout-1")
        assert state.is_duplicate("rollout-1") is True

    def test_duplicate_when_in_completed_history(self) -> None:
        """Returns True when rollout_id is in completed_rollouts."""
        state = _make_state()
        state.completed_rollouts["rollout-1"] = time.monotonic()
        assert state.is_duplicate("rollout-1") is True

    def test_duplicate_when_in_rollout_tasks(self) -> None:
        """Returns True when rollout_id has an active task."""
        state = _make_state()
        mock_task = MagicMock()
        state.rollout_tasks["rollout-1"] = mock_task  # type: ignore[assignment]
        assert state.is_duplicate("rollout-1") is True

    def test_not_duplicate_for_unknown_key(self) -> None:
        """Returns False for a completely unknown rollout_id."""
        state = _make_state()
        assert state.is_duplicate("unknown") is False

    async def test_not_duplicate_after_clearing_all_records(self) -> None:
        """After clearing init record and no task/completed, not duplicate."""
        state = _make_state()
        state.get_or_create_init_future("rollout-1")
        state.clear_init_record("rollout-1")
        assert state.is_duplicate("rollout-1") is False


# =============================================================================
# mark_started / mark_completed Tests
# =============================================================================


class TestMarkStartedAndCompleted:
    """Tests for AppState.mark_started and mark_completed."""

    def test_mark_started_records_task(self) -> None:
        """mark_started stores the task in rollout_tasks."""
        state = _make_state()
        mock_task = MagicMock()
        state.mark_started("rollout-1", mock_task)  # type: ignore[arg-type]
        assert "rollout-1" in state.rollout_tasks
        assert state.rollout_tasks["rollout-1"] is mock_task

    def test_mark_started_increments_active_count(self) -> None:
        """active_count increases after mark_started."""
        state = _make_state()
        assert state.active_count == 0
        state.mark_started("r1", MagicMock())  # type: ignore[arg-type]
        assert state.active_count == 1
        state.mark_started("r2", MagicMock())  # type: ignore[arg-type]
        assert state.active_count == 2

    def test_mark_completed_removes_from_active(self) -> None:
        """mark_completed removes the task from rollout_tasks."""
        state = _make_state()
        state.mark_started("rollout-1", MagicMock())  # type: ignore[arg-type]
        state.mark_completed("rollout-1")
        assert "rollout-1" not in state.rollout_tasks

    def test_mark_completed_adds_to_completed_history(self) -> None:
        """mark_completed adds entry to completed_rollouts."""
        state = _make_state()
        state.mark_started("rollout-1", MagicMock())  # type: ignore[arg-type]
        state.mark_completed("rollout-1")
        assert "rollout-1" in state.completed_rollouts
        assert state.completed_count == 1

    def test_mark_completed_records_monotonic_time(self) -> None:
        """mark_completed records a monotonic timestamp."""
        state = _make_state()
        before = time.monotonic()
        state.mark_completed("rollout-1")
        after = time.monotonic()
        ts = state.completed_rollouts["rollout-1"]
        assert before <= ts <= after

    def test_mark_completed_nonexistent_key_safe(self) -> None:
        """mark_completed for a key not in rollout_tasks doesn't raise."""
        state = _make_state()
        # Should not raise
        state.mark_completed("nonexistent")
        assert "nonexistent" in state.completed_rollouts

    def test_full_lifecycle(self) -> None:
        """Full lifecycle: start -> complete -> is_duplicate checks."""
        state = _make_state()
        mock_task = MagicMock()

        # Before start: not a duplicate
        assert state.is_duplicate("rollout-1") is False

        # After start: is duplicate (in rollout_tasks)
        state.mark_started("rollout-1", mock_task)  # type: ignore[arg-type]
        assert state.is_duplicate("rollout-1") is True
        assert state.active_count == 1

        # After complete: still duplicate (in completed_rollouts)
        state.mark_completed("rollout-1")
        assert state.is_duplicate("rollout-1") is True
        assert state.active_count == 0
        assert state.completed_count == 1


# =============================================================================
# _prune_completed_records Tests (TTL expiry)
# =============================================================================


class TestPruneCompletedRecords:
    """Tests for AppState._prune_completed_records."""

    def test_expired_records_are_pruned(self) -> None:
        """Records older than TTL are removed by pruning."""
        state = _make_state(record_ttl_seconds=60.0)

        # Insert a record with a timestamp well in the past
        state.completed_rollouts["old"] = time.monotonic() - 120.0
        assert state.completed_count == 1

        state._prune_completed_records()
        assert "old" not in state.completed_rollouts
        assert state.completed_count == 0

    def test_recent_records_are_kept(self) -> None:
        """Records within TTL are not pruned."""
        state = _make_state(record_ttl_seconds=60.0)

        state.completed_rollouts["recent"] = time.monotonic()
        state._prune_completed_records()
        assert "recent" in state.completed_rollouts

    def test_mixed_expired_and_recent(self) -> None:
        """Only expired records are pruned; recent ones remain."""
        state = _make_state(record_ttl_seconds=60.0)

        now = time.monotonic()
        state.completed_rollouts["old1"] = now - 120.0
        state.completed_rollouts["old2"] = now - 90.0
        state.completed_rollouts["recent1"] = now - 10.0
        state.completed_rollouts["recent2"] = now

        state._prune_completed_records()

        assert "old1" not in state.completed_rollouts
        assert "old2" not in state.completed_rollouts
        assert "recent1" in state.completed_rollouts
        assert "recent2" in state.completed_rollouts

    async def test_prune_also_removes_init_futures(self) -> None:
        """Pruning expired completed records also removes their init futures."""
        state = _make_state(record_ttl_seconds=60.0)

        state.get_or_create_init_future("old-rollout")
        state.completed_rollouts["old-rollout"] = time.monotonic() - 120.0

        state._prune_completed_records()

        assert "old-rollout" not in state.completed_rollouts
        assert "old-rollout" not in state._init_futures

    def test_prune_empty_state_no_error(self) -> None:
        """Pruning with no completed records does not raise."""
        state = _make_state()
        state._prune_completed_records()
        assert state.completed_count == 0

    def test_prune_with_time_mock(self) -> None:
        """Test pruning with mocked monotonic time for precision."""
        state = _make_state(record_ttl_seconds=100.0)

        # Insert a record at "time 1000"
        state.completed_rollouts["r1"] = 1000.0
        state.completed_rollouts["r2"] = 1050.0

        # Mock time to be 1110 (r1 expired at TTL=100, r2 still valid)
        with patch(
            "osmosis_ai.rollout.server.state.time.monotonic", return_value=1110.0
        ):
            state._prune_completed_records()

        assert "r1" not in state.completed_rollouts
        assert "r2" in state.completed_rollouts


# =============================================================================
# Cleanup Task Lifecycle Tests
# =============================================================================


class TestCleanupTaskLifecycle:
    """Tests for start_cleanup_task and stop_cleanup_task."""

    async def test_start_cleanup_creates_task(self) -> None:
        """start_cleanup_task creates a background asyncio.Task."""
        state = _make_state(cleanup_interval_seconds=100.0)
        try:
            state.start_cleanup_task()
            assert state._cleanup_task is not None
            assert isinstance(state._cleanup_task, asyncio.Task)
            assert not state._cleanup_task.done()
        finally:
            await state.stop_cleanup_task()

    async def test_stop_cleanup_cancels_task(self) -> None:
        """stop_cleanup_task cancels the background task and sets it to None."""
        state = _make_state(cleanup_interval_seconds=100.0)
        state.start_cleanup_task()
        assert state._cleanup_task is not None

        await state.stop_cleanup_task()
        assert state._cleanup_task is None

    async def test_stop_cleanup_when_not_started(self) -> None:
        """stop_cleanup_task is safe to call when no task is running."""
        state = _make_state()
        # Should not raise
        await state.stop_cleanup_task()
        assert state._cleanup_task is None

    async def test_start_cleanup_idempotent(self) -> None:
        """Calling start_cleanup_task twice doesn't create a second task."""
        state = _make_state(cleanup_interval_seconds=100.0)
        try:
            state.start_cleanup_task()
            first_task = state._cleanup_task

            state.start_cleanup_task()
            assert state._cleanup_task is first_task
        finally:
            await state.stop_cleanup_task()

    async def test_cleanup_loop_actually_prunes(self) -> None:
        """Verify the cleanup loop actually calls _prune_completed_records."""
        state = _make_state(record_ttl_seconds=60.0)

        # Insert an expired record
        state.completed_rollouts["expired"] = time.monotonic() - 120.0

        # Patch the cleanup interval to be very short so the loop fires fast,
        # and patch asyncio.sleep inside the module to return immediately.
        state._cleanup_interval = 0.01

        sleep_call_count = 0
        original_sleep = asyncio.sleep

        async def fast_sleep(delay):
            nonlocal sleep_call_count
            sleep_call_count += 1
            # Actually sleep a tiny bit to yield control
            await original_sleep(0.01)

        try:
            with patch("asyncio.sleep", side_effect=fast_sleep):
                state.start_cleanup_task()
                # Give the loop a chance to run at least once
                await original_sleep(0.05)

            assert "expired" not in state.completed_rollouts
        finally:
            await state.stop_cleanup_task()


# =============================================================================
# cancel_all Tests
# =============================================================================


class TestCancelAll:
    """Tests for AppState.cancel_all."""

    async def test_cancel_all_cancels_pending_tasks(self) -> None:
        """cancel_all cancels all pending rollout tasks."""
        state = _make_state()

        async def long_running():
            await asyncio.sleep(100)

        task1 = asyncio.create_task(long_running())
        task2 = asyncio.create_task(long_running())
        state.mark_started("r1", task1)
        state.mark_started("r2", task2)

        assert state.active_count == 2

        await state.cancel_all()

        assert state.active_count == 0
        assert state.rollout_tasks == {}
        assert task1.cancelled()
        assert task2.cancelled()

    async def test_cancel_all_empty_state(self) -> None:
        """cancel_all with no tasks does nothing and doesn't raise."""
        state = _make_state()
        await state.cancel_all()
        assert state.active_count == 0

    async def test_cancel_all_clears_rollout_tasks(self) -> None:
        """cancel_all clears the rollout_tasks dict."""
        state = _make_state()

        async def long_running():
            await asyncio.sleep(100)

        state.mark_started("r1", asyncio.create_task(long_running()))
        await state.cancel_all()
        assert state.rollout_tasks == {}

    async def test_cancel_all_handles_already_completed_tasks(self) -> None:
        """cancel_all handles tasks that have already completed."""
        state = _make_state()

        async def quick_task():
            return "done"

        task = asyncio.create_task(quick_task())
        # Let it complete
        await asyncio.sleep(0.01)
        assert task.done()

        state.mark_started("r1", task)
        # Should not raise even though the task is already done
        await state.cancel_all()
        assert state.active_count == 0


# =============================================================================
# active_count / completed_count Property Tests
# =============================================================================


class TestCountProperties:
    """Tests for active_count and completed_count properties."""

    def test_active_count_reflects_rollout_tasks(self) -> None:
        """active_count matches the number of entries in rollout_tasks."""
        state = _make_state()
        state.rollout_tasks["a"] = MagicMock()  # type: ignore[assignment]
        state.rollout_tasks["b"] = MagicMock()  # type: ignore[assignment]
        assert state.active_count == 2

    def test_completed_count_reflects_completed_rollouts(self) -> None:
        """completed_count matches the number of entries in completed_rollouts."""
        state = _make_state()
        state.completed_rollouts["x"] = time.monotonic()
        state.completed_rollouts["y"] = time.monotonic()
        state.completed_rollouts["z"] = time.monotonic()
        assert state.completed_count == 3

    def test_counts_are_independent(self) -> None:
        """active_count and completed_count track different things."""
        state = _make_state()
        state.rollout_tasks["active"] = MagicMock()  # type: ignore[assignment]
        state.completed_rollouts["done"] = time.monotonic()
        assert state.active_count == 1
        assert state.completed_count == 1


# =============================================================================
# Integration / Edge-case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    async def test_concurrent_get_or_create_same_key(self) -> None:
        """Multiple concurrent calls to get_or_create_init_future with same key."""
        state = _make_state()

        results = []
        for _ in range(10):
            future, created = state.get_or_create_init_future("same-key")
            results.append((future, created))

        # First should be created, rest should not
        assert results[0][1] is True
        for future, created in results[1:]:
            assert created is False
            assert future is results[0][0]

    def test_mark_completed_overwrites_previous_completion(self) -> None:
        """Completing the same key again updates the timestamp."""
        state = _make_state()
        state.mark_completed("rollout-1")
        first_time = state.completed_rollouts["rollout-1"]

        # Small sleep to ensure time difference
        import time as _time

        _time.sleep(0.001)

        state.mark_completed("rollout-1")
        second_time = state.completed_rollouts["rollout-1"]

        assert second_time >= first_time

    async def test_restart_cleanup_after_stop(self) -> None:
        """Can restart cleanup task after stopping it."""
        state = _make_state(cleanup_interval_seconds=100.0)
        try:
            state.start_cleanup_task()
            first_task = state._cleanup_task

            await state.stop_cleanup_task()
            assert state._cleanup_task is None

            state.start_cleanup_task()
            assert state._cleanup_task is not None
            assert state._cleanup_task is not first_task
        finally:
            await state.stop_cleanup_task()

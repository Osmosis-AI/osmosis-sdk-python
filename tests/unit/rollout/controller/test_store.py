"""Tests for the eval-agnostic in-memory callback store."""

from __future__ import annotations

import asyncio
import logging

import pytest

from osmosis_ai.rollout.controller.store import (
    CallbackStore,
    DuplicateRegistrationError,
    TerminalCallbackResult,
    UnknownRolloutIdError,
    duplicate_callback_payload,
)
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutSample,
    RolloutStatus,
)

ROLLOUT_ID = "a" * 32


def _completion(**overrides: object) -> RolloutCompleteRequest:
    payload: dict[str, object] = {
        "status": RolloutStatus.SUCCESS,
        "rollout_id": ROLLOUT_ID,
    }
    payload.update(overrides)
    return RolloutCompleteRequest.model_validate(payload)


def _grader(**overrides: object) -> GraderCompleteRequest:
    payload: dict[str, object] = {
        "status": GraderStatus.SUCCESS,
        "rollout_id": ROLLOUT_ID,
        "sample": RolloutSample(
            messages=[{"role": "assistant", "content": "ok"}],
            reward=1.0,
        ),
    }
    payload.update(overrides)
    return GraderCompleteRequest.model_validate(payload)


async def test_register_then_discard_forgets_live_rendezvous() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    ack = await store.handle_completion(ROLLOUT_ID, _completion())
    assert ack == {"ok": True}

    await store.discard(ROLLOUT_ID)

    with pytest.raises(UnknownRolloutIdError):
        await store.handle_completion(ROLLOUT_ID, _completion())


async def test_completion_is_not_terminal() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    await store.handle_completion(ROLLOUT_ID, _completion())

    assert store.completion_for(ROLLOUT_ID) is not None
    assert store.terminal_for(ROLLOUT_ID) is None


async def test_wait_completion_returns_accepted_payload() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    waiter = asyncio.create_task(store.wait_completion(ROLLOUT_ID))
    await asyncio.sleep(0)
    first = _completion()
    await store.handle_completion(ROLLOUT_ID, first)
    accepted = await waiter
    assert accepted == first
    assert await store.wait_completion(ROLLOUT_ID) == first
    await store.handle_grader(ROLLOUT_ID, _grader())
    await store.discard(ROLLOUT_ID)
    assert await store.wait_completion(ROLLOUT_ID) == first


async def test_identical_duplicate_completion_does_not_resolve_twice() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    first = _completion()
    first_ack = await store.handle_completion(ROLLOUT_ID, first)
    second_ack = await store.handle_completion(ROLLOUT_ID, _completion())
    assert second_ack == first_ack
    assert await store.wait_completion(ROLLOUT_ID) == first


async def test_conflicting_duplicate_completion_keeps_first_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    first = _completion()
    await store.handle_completion(ROLLOUT_ID, first)
    with caplog.at_level(logging.ERROR):
        ack = await store.handle_completion(
            ROLLOUT_ID,
            _completion(status=RolloutStatus.FAILURE, err_message="agent exploded"),
        )
    assert ack == {"ok": True}
    assert await store.wait_completion(ROLLOUT_ID) == first
    assert any(
        "conflicting duplicate" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    )


async def test_discard_cancels_unresolved_completion_and_terminal_futures() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    completion_waiter = asyncio.create_task(store.wait_completion(ROLLOUT_ID))
    terminal_waiter = asyncio.create_task(store.wait_terminal(ROLLOUT_ID))
    await asyncio.sleep(0)
    await store.discard(ROLLOUT_ID)
    with pytest.raises(asyncio.CancelledError):
        await completion_waiter
    with pytest.raises(asyncio.CancelledError):
        await terminal_waiter


async def test_seeded_terminal_retains_completion_for_wait_and_duplicates() -> None:
    store = CallbackStore()
    stored = _completion()
    store.seed_terminal(
        ROLLOUT_ID,
        acknowledgment={"ok": True, "seeded": True},
        grader=_grader(),
        completion=stored,
    )
    assert await store.wait_completion(ROLLOUT_ID) == stored
    ack = await store.handle_completion(ROLLOUT_ID, stored)
    assert ack == {"ok": True, "seeded": True}


async def test_grader_commit_hook_runs_before_acknowledgment() -> None:
    order: list[str] = []

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        order.append("commit-start")
        order.append("commit-end")
        return {"ok": True, "durable": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    ack = await store.handle_grader(ROLLOUT_ID, _grader())
    order.append("acked")

    assert ack == {"ok": True, "durable": True}
    assert order == ["commit-start", "commit-end", "acked"]
    assert store.terminal_for(ROLLOUT_ID) is not None


async def test_identical_duplicate_grader_returns_stored_ack_without_recommit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True, "n": commits}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    first = await store.handle_grader(ROLLOUT_ID, _grader())
    with caplog.at_level(logging.ERROR):
        second = await store.handle_grader(ROLLOUT_ID, _grader())

    assert first == {"ok": True, "n": 1}
    assert second == first
    assert commits == 1
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_conflicting_duplicate_grader_logs_error_and_returns_stored_ack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    first = await store.handle_grader(ROLLOUT_ID, _grader())
    conflict = _grader(
        status=GraderStatus.FAILURE,
        sample=None,
        err_message="grader exploded",
    )
    with caplog.at_level(logging.ERROR):
        second = await store.handle_grader(ROLLOUT_ID, conflict)

    assert second == first
    assert commits == 1
    assert any(
        "conflicting duplicate" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    )


async def test_first_terminal_result_wins_timeout_then_late_grader() -> None:
    commits: list[str] = []

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        source = result.source
        commits.append(source)
        return {"ok": True, "source": source}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    timeout_ack = await store.finalize_timeout(
        ROLLOUT_ID, acknowledgment={"ok": True, "source": "timeout"}
    )
    late = await store.handle_grader(ROLLOUT_ID, _grader())

    assert timeout_ack.acknowledgment["source"] == "timeout"
    assert late == timeout_ack.acknowledgment
    assert commits == ["timeout"]


async def test_wait_terminal_returns_grader_result() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    waiter = asyncio.create_task(store.wait_terminal(ROLLOUT_ID))
    await asyncio.sleep(0)
    await store.handle_grader(ROLLOUT_ID, _grader())
    result = await waiter
    assert result.source == "grader"
    assert result.grader is not None
    assert result.grader.status == GraderStatus.SUCCESS


async def test_seed_terminal_replays_without_commit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True}

    store = CallbackStore(on_terminal_commit=commit)
    store.seed_terminal(
        ROLLOUT_ID,
        acknowledgment={"ok": True, "seeded": True},
        grader=_grader(),
    )
    with caplog.at_level(logging.ERROR):
        ack = await store.handle_grader(ROLLOUT_ID, _grader())
    assert ack == {"ok": True, "seeded": True}
    assert commits == 0
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_register_rejects_live_and_finalized_ids() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    with pytest.raises(DuplicateRegistrationError):
        await store.register(ROLLOUT_ID)
    await store.handle_grader(ROLLOUT_ID, _grader())
    await store.discard(ROLLOUT_ID)
    with pytest.raises(DuplicateRegistrationError):
        await store.register(ROLLOUT_ID)
    ack = await store.handle_grader(ROLLOUT_ID, _grader())
    assert ack == {"ok": True}


async def test_finalized_id_cannot_be_registered_and_committed_again() -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True, "n": commits}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    first = await store.handle_grader(ROLLOUT_ID, _grader())
    await store.discard(ROLLOUT_ID)
    with pytest.raises(DuplicateRegistrationError):
        await store.register(ROLLOUT_ID)
    second = await store.handle_grader(ROLLOUT_ID, _grader())
    assert first == {"ok": True, "n": 1}
    assert second == first
    assert commits == 1


async def test_seed_terminal_does_not_overwrite_live_or_finalized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    with caplog.at_level(logging.ERROR):
        store.seed_terminal(
            ROLLOUT_ID,
            acknowledgment={"ok": True, "seeded": True},
            grader=_grader(),
        )
    assert store.terminal_for(ROLLOUT_ID) is None
    assert any("live session" in record.message.lower() for record in caplog.records)

    await store.handle_grader(ROLLOUT_ID, _grader())
    await store.discard(ROLLOUT_ID)
    stored = store.terminal_for(ROLLOUT_ID)
    assert stored is not None
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        store.seed_terminal(
            ROLLOUT_ID,
            acknowledgment={"ok": True, "seeded": True},
            grader=_grader(status=GraderStatus.FAILURE, sample=None),
        )
    kept = store.terminal_for(ROLLOUT_ID)
    assert kept is stored
    assert any(
        "conflicting seed_terminal" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    )


async def test_identical_repeated_seed_is_noop() -> None:
    store = CallbackStore()
    grader = _grader()
    store.seed_terminal(
        ROLLOUT_ID,
        acknowledgment={"ok": True, "seeded": True},
        grader=grader,
    )
    first = store.terminal_for(ROLLOUT_ID)
    store.seed_terminal(
        ROLLOUT_ID,
        acknowledgment={"ok": True, "seeded": True},
        grader=_grader(),
    )
    assert store.terminal_for(ROLLOUT_ID) is first


async def test_unknown_rollout_is_rejected() -> None:
    store = CallbackStore()
    with pytest.raises(UnknownRolloutIdError):
        await store.handle_grader(ROLLOUT_ID, _grader())


async def test_timeout_acknowledgment_survives_default_commit_hook() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    result = await store.finalize_timeout(
        ROLLOUT_ID, acknowledgment={"ok": True, "source": "timeout"}
    )
    assert result.acknowledgment == {"ok": True, "source": "timeout"}


async def test_late_completion_after_timeout_returns_stored_ack() -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    timeout = await store.finalize_timeout(
        ROLLOUT_ID, acknowledgment={"ok": True, "error_type": "callback_timeout"}
    )
    await store.discard(ROLLOUT_ID)

    ack = await store.handle_completion(ROLLOUT_ID, _completion())
    assert ack == timeout.acknowledgment


async def test_late_callbacks_after_timeout_do_not_log_conflict_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    timeout = await store.finalize_timeout(
        ROLLOUT_ID, acknowledgment={"ok": True, "error_type": "callback_timeout"}
    )
    await store.discard(ROLLOUT_ID)

    with caplog.at_level(logging.ERROR):
        grader_ack = await store.handle_grader(ROLLOUT_ID, _grader())
        completion_ack = await store.handle_completion(ROLLOUT_ID, _completion())

    assert grader_ack == timeout.acknowledgment
    assert completion_ack == timeout.acknowledgment
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_late_callbacks_after_cancel_do_not_log_conflict_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    tombstone = await store.acknowledge_without_commit(ROLLOUT_ID)
    await store.discard(ROLLOUT_ID)

    with caplog.at_level(logging.ERROR):
        grader_ack = await store.handle_grader(ROLLOUT_ID, _grader())
        completion_ack = await store.handle_completion(ROLLOUT_ID, _completion())

    assert grader_ack == tombstone.acknowledgment
    assert completion_ack == tombstone.acknowledgment
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_stored_real_payload_conflicts_still_log_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CallbackStore()
    await store.register(ROLLOUT_ID)
    await store.handle_completion(ROLLOUT_ID, _completion())
    await store.handle_grader(ROLLOUT_ID, _grader())
    await store.discard(ROLLOUT_ID)

    with caplog.at_level(logging.ERROR):
        await store.handle_grader(
            ROLLOUT_ID,
            _grader(status=GraderStatus.FAILURE, sample=None, err_message="boom"),
        )
        await store.handle_completion(
            ROLLOUT_ID,
            _completion(status=RolloutStatus.FAILURE, err_message="boom"),
        )

    conflict_errors = [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and "conflicting duplicate" in record.message.lower()
    ]
    assert len(conflict_errors) == 2


async def test_cancel_tombstone_acks_callbacks_without_commit() -> None:
    commits = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal commits
        commits += 1
        return {"ok": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    tombstone = await store.acknowledge_without_commit(ROLLOUT_ID)

    assert tombstone.source == "cancelled"
    assert commits == 0
    assert await store.handle_grader(ROLLOUT_ID, _grader()) == tombstone.acknowledgment
    assert await store.handle_completion(ROLLOUT_ID, _completion()) == (
        tombstone.acknowledgment
    )
    assert commits == 0


async def test_acknowledge_without_commit_rejects_unknown_rollout_id() -> None:
    store = CallbackStore()
    with pytest.raises(UnknownRolloutIdError):
        await store.acknowledge_without_commit(ROLLOUT_ID)

    await store.register(ROLLOUT_ID)
    ack = await store.handle_grader(ROLLOUT_ID, _grader())
    assert ack == {"ok": True}


class _BlockingCommit:
    """Commit hook that blocks so tests can race contenders against it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.starts = 0
        self.completions = 0

    async def __call__(self, result: TerminalCallbackResult) -> dict[str, object]:
        self.starts += 1
        self.started.set()
        await self.release.wait()
        self.completions += 1
        return {"ok": True, "durable": True}


async def test_cancelled_grader_handler_does_not_rerun_commit_hook() -> None:
    commit = _BlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    first = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    commit.release.set()
    retry_ack = await store.handle_grader(ROLLOUT_ID, _grader())

    assert (commit.starts, commit.completions) == (1, 1)
    assert retry_ack == {"ok": True, "durable": True}
    stored = store.terminal_for(ROLLOUT_ID)
    assert stored is not None
    assert stored.source == "grader"


async def test_timeout_observes_commit_surviving_cancelled_grader_handler() -> None:
    commit = _BlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    grader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await grader_task

    timeout_task = asyncio.create_task(
        store.finalize_timeout(
            ROLLOUT_ID, acknowledgment={"ok": True, "error_type": "callback_timeout"}
        )
    )
    await asyncio.sleep(0)
    commit.release.set()
    result = await timeout_task

    assert result.source == "grader"
    assert (commit.starts, commit.completions) == (1, 1)


async def test_cancel_observes_commit_surviving_cancelled_grader_handler() -> None:
    commit = _BlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    grader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await grader_task

    cancel_task = asyncio.create_task(store.acknowledge_without_commit(ROLLOUT_ID))
    await asyncio.sleep(0)
    commit.release.set()
    result = await cancel_task

    assert result.source == "grader"
    assert (commit.starts, commit.completions) == (1, 1)


async def test_failed_commit_hook_is_retryable() -> None:
    calls = 0

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("journal io error")
        return {"ok": True, "n": calls}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    with pytest.raises(RuntimeError, match="journal io error"):
        await store.handle_grader(ROLLOUT_ID, _grader())
    assert store.terminal_for(ROLLOUT_ID) is None

    ack = await store.handle_grader(ROLLOUT_ID, _grader())
    assert ack == {"ok": True, "n": 2}


async def test_discard_does_not_cancel_in_flight_commit() -> None:
    commit = _BlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    discard_task = asyncio.create_task(store.discard(ROLLOUT_ID))
    await asyncio.sleep(0)
    commit.release.set()

    ack = await grader_task
    await discard_task

    assert ack == {"ok": True, "durable": True}
    assert (commit.starts, commit.completions) == (1, 1)
    stored = store.terminal_for(ROLLOUT_ID)
    assert stored is not None
    assert stored.source == "grader"


class _FailFirstBlockingCommit:
    """Blocks the first commit, then fails it; later commits succeed."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = 0

    async def __call__(self, result: TerminalCallbackResult) -> dict[str, object]:
        self.attempts += 1
        if self.attempts == 1:
            self.started.set()
            await self.release.wait()
            raise RuntimeError("journal io error")
        return {"ok": True, "attempt": self.attempts, "source": result.source}


async def test_cancel_waiting_on_failed_commit_installs_tombstone() -> None:
    commit = _FailFirstBlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    cancel_task = asyncio.create_task(store.acknowledge_without_commit(ROLLOUT_ID))
    await asyncio.sleep(0)
    commit.release.set()

    with pytest.raises(RuntimeError, match="journal io error"):
        await grader_task
    tombstone = await cancel_task

    assert tombstone.source == "cancelled"
    assert commit.attempts == 1  # cancellation never runs the hook
    stored = store.terminal_for(ROLLOUT_ID)
    assert stored is not None
    assert stored.source == "cancelled"


async def test_timeout_waiting_on_failed_commit_claims_timeout_result() -> None:
    commit = _FailFirstBlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    timeout_task = asyncio.create_task(
        store.finalize_timeout(
            ROLLOUT_ID, acknowledgment={"ok": True, "error_type": "callback_timeout"}
        )
    )
    await asyncio.sleep(0)
    commit.release.set()

    with pytest.raises(RuntimeError, match="journal io error"):
        await grader_task
    result = await timeout_task

    assert result.source == "timeout"
    assert result.acknowledgment == {"ok": True, "attempt": 2, "source": "timeout"}
    assert commit.attempts == 2


async def test_waiter_cancellation_propagates_and_leaves_commit_running() -> None:
    commit = _BlockingCommit()
    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)

    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await commit.started.wait()
    waiter = asyncio.create_task(store.finalize_timeout(ROLLOUT_ID))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    commit.release.set()
    ack = await grader_task
    assert ack == {"ok": True, "durable": True}
    assert (commit.starts, commit.completions) == (1, 1)


async def test_in_flight_commit_wins_over_cancel_tombstone() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def commit(result: TerminalCallbackResult) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"ok": True, "durable": True}

    store = CallbackStore(on_terminal_commit=commit)
    await store.register(ROLLOUT_ID)
    grader_task = asyncio.create_task(store.handle_grader(ROLLOUT_ID, _grader()))
    await started.wait()
    cancel_task = asyncio.create_task(store.acknowledge_without_commit(ROLLOUT_ID))
    await asyncio.sleep(0)
    release.set()

    grader_ack = await grader_task
    cancelled = await cancel_task
    assert grader_ack == {"ok": True, "durable": True}
    assert cancelled.source == "grader"
    assert cancelled.acknowledgment == grader_ack


def test_duplicate_payload_helper_matches_identical_grader_bodies() -> None:
    assert duplicate_callback_payload(_grader(), _grader())
    assert not duplicate_callback_payload(
        _grader(),
        _grader(status=GraderStatus.FAILURE, sample=None),
    )

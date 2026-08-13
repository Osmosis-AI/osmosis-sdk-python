"""In-memory callback rendezvous for one process.

The store is eval-agnostic: callers supply an async terminal-commit hook
(local eval journals here; other tooling can no-op or persist elsewhere).
Each live session has separate completion and terminal futures. Callers
register before dispatch, may ``wait_completion`` then ``wait_terminal``,
and discard both in ``finally``. Accepted terminal acknowledgments stay
until the process exits so duplicate callbacks can be acknowledged after
cleanup. Journal replay is a small ``seed_terminal`` API; this module
does not read a journal.

The terminal commit is single-flight: the first grader/timeout contender
creates one commit task per session and every contender (including
cancellation) awaits that same task through ``asyncio.shield``, so
cancelling a callback handler or timeout waiter never cancels persistence
and never lets a second contender rerun the hook. A hook that raises is
treated as not-durable: the failed task is cleared so a retry can commit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from osmosis_ai.rollout.types import GraderCompleteRequest, RolloutCompleteRequest

logger: logging.Logger = logging.getLogger(__name__)

TerminalSource = Literal["grader", "timeout", "seeded", "cancelled"]


class UnknownRolloutIdError(KeyError):
    """No live session or seeded terminal exists for this rollout id."""


class DuplicateRegistrationError(ValueError):
    """``register`` was called twice for the same live rollout id."""


@dataclass(frozen=True)
class TerminalCallbackResult:
    """First-accepted terminal callback (or timeout) for one rollout."""

    rollout_id: str
    source: TerminalSource
    completion: RolloutCompleteRequest | None = None
    grader: GraderCompleteRequest | None = None
    acknowledgment: Mapping[str, Any] = field(default_factory=lambda: {"ok": True})


TerminalCommitHook = Callable[
    [TerminalCallbackResult], Awaitable[Mapping[str, Any] | None]
]


def duplicate_callback_payload(left: BaseModel | None, right: BaseModel | None) -> bool:
    """True when two protocol bodies serialize to the same JSON object."""
    if left is None or right is None:
        return left is right
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _terminal_payloads_match(
    left: TerminalCallbackResult, right: TerminalCallbackResult
) -> bool:
    return (
        duplicate_callback_payload(left.grader, right.grader)
        and duplicate_callback_payload(left.completion, right.completion)
        and dict(left.acknowledgment) == dict(right.acknowledgment)
    )


async def _default_commit(_result: TerminalCallbackResult) -> None:
    return None


@dataclass
class _LiveSession:
    completion: RolloutCompleteRequest | None = None
    completion_ack: dict[str, Any] = field(default_factory=lambda: {"ok": True})
    completion_future: asyncio.Future[RolloutCompleteRequest] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    terminal: TerminalCallbackResult | None = None
    terminal_future: asyncio.Future[TerminalCallbackResult] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Single-flight terminal commit; contenders await it via asyncio.shield.
    commit_task: asyncio.Task[TerminalCallbackResult] | None = None


class CallbackStore:
    """One in-memory rendezvous per rollout, plus a durable-ack finalized map."""

    def __init__(
        self,
        *,
        on_terminal_commit: TerminalCommitHook | None = None,
    ) -> None:
        self._on_terminal_commit = on_terminal_commit or _default_commit
        self._sessions: dict[str, _LiveSession] = {}
        self._finalized: dict[str, TerminalCallbackResult] = {}

    async def register(self, rollout_id: str) -> None:
        if rollout_id in self._sessions or rollout_id in self._finalized:
            raise DuplicateRegistrationError(
                f"callback session already registered for {rollout_id}"
            )
        self._sessions[rollout_id] = _LiveSession()

    def seed_terminal(
        self,
        rollout_id: str,
        *,
        acknowledgment: Mapping[str, Any],
        grader: GraderCompleteRequest | None = None,
        completion: RolloutCompleteRequest | None = None,
    ) -> None:
        """Record an already-durable terminal result for duplicate detection.

        First-wins: live sessions and existing finalized rows are not
        overwritten. An identical repeated seed is a no-op; a conflicting
        seed keeps the stored result and logs ERROR.
        """
        incoming = TerminalCallbackResult(
            rollout_id=rollout_id,
            source="seeded",
            completion=completion,
            grader=grader,
            acknowledgment=dict(acknowledgment),
        )
        if rollout_id in self._sessions:
            logger.error(
                "seed_terminal ignored; live session exists for rollout %s",
                rollout_id,
            )
            return
        existing = self._finalized.get(rollout_id)
        if existing is not None:
            if not _terminal_payloads_match(existing, incoming):
                logger.error(
                    "Conflicting seed_terminal for rollout %s; keeping first result",
                    rollout_id,
                )
            return
        self._finalized[rollout_id] = incoming

    async def discard(self, rollout_id: str) -> None:
        session = self._sessions.get(rollout_id)
        if session is None:
            return
        async with session.lock:
            self._sessions.pop(rollout_id, None)
            if not session.completion_future.done():
                session.completion_future.cancel()
            if not session.terminal_future.done():
                session.terminal_future.cancel()

    def completion_for(self, rollout_id: str) -> RolloutCompleteRequest | None:
        session = self._sessions.get(rollout_id)
        if session is not None:
            return session.completion
        finalized = self._finalized.get(rollout_id)
        return None if finalized is None else finalized.completion

    def terminal_for(self, rollout_id: str) -> TerminalCallbackResult | None:
        session = self._sessions.get(rollout_id)
        if session is not None and session.terminal is not None:
            return session.terminal
        return self._finalized.get(rollout_id)

    async def handle_completion(
        self, rollout_id: str, request: RolloutCompleteRequest
    ) -> dict[str, Any]:
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            return self._ack_stored_completion(rollout_id, finalized, request)

        assert session is not None
        async with session.lock:
            if session.completion is None:
                session.completion = request
                if not session.completion_future.done():
                    session.completion_future.set_result(request)
            elif not duplicate_callback_payload(session.completion, request):
                logger.error(
                    "Conflicting duplicate completion callback for rollout %s",
                    rollout_id,
                )
            if session.terminal is not None:
                return dict(session.terminal.acknowledgment)
            return dict(session.completion_ack)

    async def handle_grader(
        self, rollout_id: str, request: GraderCompleteRequest
    ) -> dict[str, Any]:
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            return self._ack_duplicate_terminal(rollout_id, finalized, request)

        assert session is not None
        async with session.lock:
            if session.terminal is not None:
                return self._ack_duplicate_terminal(
                    rollout_id, session.terminal, request
                )
            task = session.commit_task
            if task is None:
                result = TerminalCallbackResult(
                    rollout_id=rollout_id,
                    source="grader",
                    completion=session.completion,
                    grader=request,
                )
                task = self._start_commit(session, result)
        stored = await asyncio.shield(task)
        return self._ack_duplicate_terminal(rollout_id, stored, request)

    async def finalize_timeout(
        self,
        rollout_id: str,
        *,
        acknowledgment: Mapping[str, Any] | None = None,
    ) -> TerminalCallbackResult:
        """Win or observe the terminal race with a timeout result.

        When an observed in-flight commit fails without producing a durable
        result, the timeout re-arbitrates and may claim the terminal slot
        with its own commit. A failure of the timeout's own commit
        propagates. Task cancellation always propagates.
        """
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            return finalized

        assert session is not None
        while True:
            created = False
            async with session.lock:
                if session.terminal is not None:
                    return session.terminal
                task = session.commit_task
                if task is None:
                    result = TerminalCallbackResult(
                        rollout_id=rollout_id,
                        source="timeout",
                        completion=session.completion,
                        grader=None,
                        acknowledgment=dict(acknowledgment or {"ok": True}),
                    )
                    task = self._start_commit(session, result)
                    created = True
            try:
                return await asyncio.shield(task)
            except Exception:
                if created:
                    raise
                # An observed commit failed without a durable result;
                # re-arbitrate so the timeout can claim the terminal slot.
                continue

    async def acknowledge_without_commit(
        self,
        rollout_id: str,
        *,
        source: TerminalSource = "cancelled",
        acknowledgment: Mapping[str, Any] | None = None,
    ) -> TerminalCallbackResult:
        """Mark terminal without running the commit hook (cancellation).

        Subsequent completion/grader callbacks receive the stored
        acknowledgment. First accepted terminal still wins: an in-flight
        commit is observed, not replaced — but if that commit fails without
        a durable result, cancellation re-arbitrates and installs its
        non-durable tombstone. Task cancellation always propagates. An
        unknown rollout id raises ``UnknownRolloutIdError`` instead of
        creating a tombstone that would poison a future registration.
        """
        finalized = self._finalized.get(rollout_id)
        if finalized is not None:
            return finalized

        session = self._sessions.get(rollout_id)
        if session is None:
            raise UnknownRolloutIdError(rollout_id)

        while True:
            async with session.lock:
                if session.terminal is not None:
                    return session.terminal
                task = session.commit_task
                if task is None:
                    result = TerminalCallbackResult(
                        rollout_id=rollout_id,
                        source=source,
                        completion=session.completion,
                        acknowledgment=dict(acknowledgment or {"ok": True}),
                    )
                    session.terminal = result
                    self._finalized[rollout_id] = result
                    if not session.terminal_future.done():
                        session.terminal_future.set_result(result)
                    return result
            try:
                return await asyncio.shield(task)
            except Exception:
                # The in-flight commit failed without a durable result;
                # re-arbitrate so cancellation can install its tombstone.
                continue

    async def wait_completion(self, rollout_id: str) -> RolloutCompleteRequest:
        """Return the first accepted completion callback for this rollout.

        Training/dev consumers can wait here before waiting for the terminal
        grader result, preserving the two-stage remote-rollout lifecycle.
        """
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            if finalized.completion is None:
                raise UnknownRolloutIdError(rollout_id)
            return finalized.completion
        assert session is not None
        return await asyncio.shield(session.completion_future)

    async def wait_terminal(self, rollout_id: str) -> TerminalCallbackResult:
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            return finalized
        assert session is not None
        return await asyncio.shield(session.terminal_future)

    def _session_or_finalized(
        self, rollout_id: str
    ) -> tuple[_LiveSession | None, TerminalCallbackResult | None]:
        """Return the live session, else a finalized tombstone, else raise."""
        session = self._sessions.get(rollout_id)
        if session is not None:
            return session, None
        finalized = self._finalized.get(rollout_id)
        if finalized is not None:
            return None, finalized
        raise UnknownRolloutIdError(rollout_id)

    def _start_commit(
        self,
        session: _LiveSession,
        result: TerminalCallbackResult,
    ) -> asyncio.Task[TerminalCallbackResult]:
        """Create the session's single in-flight commit task (lock held)."""
        task = asyncio.create_task(self._run_commit(session, result))
        session.commit_task = task
        return task

    async def _run_commit(
        self,
        session: _LiveSession,
        result: TerminalCallbackResult,
    ) -> TerminalCallbackResult:
        try:
            ack = await self._on_terminal_commit(result)
        except BaseException:
            # A failed/cancelled hook produced no durable result; clear the
            # task so a retried callback can commit again.
            session.commit_task = None
            raise
        stored_ack = dict(ack) if ack is not None else dict(result.acknowledgment)
        stored = TerminalCallbackResult(
            rollout_id=result.rollout_id,
            source=result.source,
            completion=result.completion,
            grader=result.grader,
            acknowledgment=stored_ack,
        )
        session.terminal = stored
        self._finalized[result.rollout_id] = stored
        if not session.terminal_future.done():
            session.terminal_future.set_result(stored)
        return stored

    def _ack_stored_completion(
        self,
        rollout_id: str,
        stored: TerminalCallbackResult,
        request: RolloutCompleteRequest,
    ) -> dict[str, Any]:
        # A timeout/cancel result that never stored a completion payload may
        # legitimately receive a late completion; only a stored real payload
        # that differs is a conflict.
        if stored.completion is not None and not duplicate_callback_payload(
            stored.completion, request
        ):
            logger.error(
                "Conflicting duplicate completion callback for rollout %s "
                "(stored source=%s)",
                rollout_id,
                stored.source,
            )
        return dict(stored.acknowledgment)

    def _ack_duplicate_terminal(
        self,
        rollout_id: str,
        stored: TerminalCallbackResult,
        request: GraderCompleteRequest,
    ) -> dict[str, Any]:
        # A timeout/cancel result carries no grader payload; a late grader
        # callback after it is the expected protocol flow, not a conflict.
        if stored.grader is not None and not duplicate_callback_payload(
            stored.grader, request
        ):
            logger.error(
                "Conflicting duplicate terminal callback for rollout %s "
                "(stored source=%s)",
                rollout_id,
                stored.source,
            )
        return dict(stored.acknowledgment)

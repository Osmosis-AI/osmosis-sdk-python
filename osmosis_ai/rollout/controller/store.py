"""In-memory callback rendezvous for one process.

Callers supply the async terminal-commit hook (local eval journals through
it), register before dispatch, ``wait_terminal``, and discard in
``finally``. A completion callback is recorded on the live session for
duplicate detection but is not a rendezvous of its own. Accepted terminal
acknowledgments stay until the process exits so duplicate callbacks can be
acknowledged after cleanup.

The terminal commit is single-flight: the first grader/timeout contender
creates one commit task per session and every contender (including
cancellation) awaits that same task through ``asyncio.shield``, so
cancelling a callback handler or timeout waiter never cancels persistence
and never lets a second contender rerun the hook. A hook that raises is
treated as not-durable: the failed task is cleared so a RETRIED CALLBACK
can commit again; the contender that observed the failure propagates it
and its item stays pending for the next run.
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

TerminalSource = Literal["grader", "timeout", "cancelled"]


class UnknownRolloutIdError(KeyError):
    """No live session or finalized terminal exists for this rollout id."""


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


@dataclass
class _LiveSession:
    completion: RolloutCompleteRequest | None = None
    completion_ack: dict[str, Any] = field(default_factory=lambda: {"ok": True})
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
        on_terminal_commit: TerminalCommitHook,
    ) -> None:
        self._on_terminal_commit = on_terminal_commit
        self._sessions: dict[str, _LiveSession] = {}
        self._finalized: dict[str, TerminalCallbackResult] = {}

    async def register(self, rollout_id: str) -> None:
        if rollout_id in self._sessions or rollout_id in self._finalized:
            raise DuplicateRegistrationError(
                f"callback session already registered for {rollout_id}"
            )
        self._sessions[rollout_id] = _LiveSession()

    async def discard(self, rollout_id: str) -> None:
        session = self._sessions.get(rollout_id)
        if session is None:
            return
        async with session.lock:
            self._sessions.pop(rollout_id, None)
            if not session.terminal_future.done():
                session.terminal_future.cancel()

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

        A commit-hook failure — observed or the timeout's own — propagates;
        the item stays pending for the next run. Task cancellation always
        propagates.
        """
        session, finalized = self._session_or_finalized(rollout_id)
        if finalized is not None:
            return finalized

        assert session is not None
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
        return await asyncio.shield(task)

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
        commit is observed, not replaced, and its failure propagates to the
        caller. Task cancellation always propagates. An unknown rollout id
        raises ``UnknownRolloutIdError`` instead of creating a tombstone
        that would poison a future registration.
        """
        finalized = self._finalized.get(rollout_id)
        if finalized is not None:
            return finalized

        session = self._sessions.get(rollout_id)
        if session is None:
            raise UnknownRolloutIdError(rollout_id)

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
        return await asyncio.shield(task)

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

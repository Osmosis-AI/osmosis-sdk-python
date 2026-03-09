import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    ConcurrencyConfig,
    GraderConfig,
)


class ConcurrencyLimiter:
    def __init__(self, *, max_concurrent: int | None) -> None:
        self.max_concurrent = max_concurrent
        self.queued = 0
        self.running = 0
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        if self._semaphore is None:
            self.running += 1
            try:
                yield
            finally:
                self.running -= 1
            return

        self.queued += 1
        try:
            await self._semaphore.acquire()
        except BaseException:
            self.queued -= 1
            raise

        self.queued -= 1
        self.running += 1
        try:
            yield
        finally:
            self.running -= 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int | None]:
        return {
            "max_concurrent": self.max_concurrent,
            "queued": self.queued,
            "running": self.running,
        }


class RolloutServerState:
    def __init__(
        self,
        *,
        agent_workflow_config: AgentWorkflowConfig,
        grader_config: GraderConfig,
    ) -> None:
        self.agent_workflow_concurrency_limiter = self._build_concurrency_limiter(
            agent_workflow_config.concurrency
        )
        self.grader_concurrency_limiter = self._build_concurrency_limiter(
            grader_config.concurrency
        )

    @staticmethod
    def _build_concurrency_limiter(
        config: ConcurrencyConfig,
    ) -> ConcurrencyLimiter:
        return ConcurrencyLimiter(max_concurrent=config.max_concurrent)

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_workflow_concurrency": (
                self.agent_workflow_concurrency_limiter.snapshot()
            ),
            "grader_concurrency": self.grader_concurrency_limiter.snapshot(),
        }

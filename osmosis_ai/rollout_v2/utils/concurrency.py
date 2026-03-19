import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


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

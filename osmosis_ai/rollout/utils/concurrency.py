import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ConcurrencyLimiter:
    def __init__(self, *, max_concurrent: int | None) -> None:
        self.max_concurrent: int | None = max_concurrent
        self.queued: int = 0
        self.running: int = 0
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        semaphore = self._semaphore
        if semaphore is not None:
            self.queued += 1
            try:
                await semaphore.acquire()
            finally:
                self.queued -= 1

        self.running += 1
        try:
            yield
        finally:
            self.running -= 1
            if semaphore is not None:
                semaphore.release()

    def snapshot(self) -> dict[str, int | None]:
        return {
            "max_concurrent": self.max_concurrent,
            "queued": self.queued,
            "running": self.running,
        }

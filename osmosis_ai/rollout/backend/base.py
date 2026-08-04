from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from osmosis_ai.rollout.types import ExecutionRequest, ExecutionResult

ResultCallback = Callable[[ExecutionResult], Awaitable[None]]


class ExecutionBackend(ABC):
    @abstractmethod
    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def max_concurrency(self) -> int:
        """Max concurrent executions this backend supports. 0 = no limit."""
        return 0

    def has_capacity(self) -> bool:
        """Whether the backend can admit another rollout right now.

        The server rejects rollout requests with 429 when this is False, so
        controllers can retry instead of silently queueing without bound.
        """
        return True

    def cancel_rollouts(
        self,
        ids: Sequence[str] | None = None,
        prefix: str | None = None,
        all: bool = False,
    ) -> dict[str, str]:
        """Cancel matching in-flight rollouts; backends without cancellation
        support report nothing cancelled."""
        return {}

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

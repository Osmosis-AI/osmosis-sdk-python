from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
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

    @property
    def capture_final_result(self) -> bool:
        """Whether the server should capture a final result without a grader URL."""
        return False

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from osmosis_ai.rollout_v2.types import ExecutionRequest, ExecutionResult

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

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

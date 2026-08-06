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

    @property
    def capture_final_result(self) -> bool:
        """Whether the server should accept the grader result for archival
        even when the request carries no grader callback URL.

        True for backends that compute the final reward themselves; eval-only
        and local callers then still archive the reward and final status.
        """
        return False

    def has_capacity(self) -> bool:
        """Whether the backend can admit another rollout right now.

        The server rejects rollout requests with 429 when this is False, so
        controllers can retry instead of silently queueing without bound.
        """
        return True

    def rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """Live or retained state for one rollout; None when unknown."""
        return None

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

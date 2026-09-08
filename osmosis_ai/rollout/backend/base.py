from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from osmosis_ai.rollout.types import ExecutionOutcome, ExecutionRequest


class ExecutionBackend(ABC):
    @abstractmethod
    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionOutcome:
        raise NotImplementedError

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
        """Cancel matching in-flight rollouts; unsupported backends return {}.

        A backend returning ``not_found`` for finished work must retain its
        terminal ``rollout_status`` until execute() and result delivery finish.
        Without a terminal status, the server still cancels its own task.
        """
        return {}

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

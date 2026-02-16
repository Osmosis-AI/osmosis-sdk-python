"""Local test runner facade for test mode.

The shared execution logic now lives in `osmosis_ai.rollout.eval.common.runner`.
This module keeps test-mode naming conventions for readability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout.eval.common.runner import (
    LocalBatchResult as LocalTestBatchResult,
)
from osmosis_ai.rollout.eval.common.runner import (
    LocalRolloutRunner,
    validate_tools,
)
from osmosis_ai.rollout.eval.common.runner import (
    LocalRunResult as LocalTestRunResult,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop
    from osmosis_ai.rollout.eval.common.dataset import DatasetRow
    from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient


class LocalTestRunner(LocalRolloutRunner):
    """Test-mode wrapper over the shared LocalRolloutRunner."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: ExternalLLMClient,
        debug: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        super().__init__(
            agent_loop=agent_loop,
            llm_client=llm_client,
            debug=debug,
            debug_dir=debug_dir
            if debug_dir is not None
            else ("./test_debug" if debug else None),
            rollout_id_prefix="test",
            request_metadata={
                "execution_mode": "test",
                "test_mode": True,
            },
        )

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        max_turns: int = 10,
        completion_params: dict[str, Any] | None = None,
    ) -> LocalTestRunResult:
        return await super().run_single(
            row=row,
            row_index=row_index,
            max_turns=max_turns,
            completion_params=completion_params,
            rollout_id=f"test-{row_index}",
        )


__all__ = [
    "LocalTestBatchResult",
    "LocalTestRunResult",
    "LocalTestRunner",
    "validate_tools",
]

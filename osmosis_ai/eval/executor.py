"""WorkflowExecutor — runs an AgentWorkflow against a single prompt.

Thin executor that constructs the v2 ambient context, invokes the
workflow, and collects metrics + samples from the proxy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.eval.proxy import EvalProxy, RequestMetrics


@dataclass
class ExecutionResult:
    """Result from running a workflow against a single prompt."""

    success: bool
    duration_ms: float = 0.0
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    samples: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class WorkflowExecutor:
    """Runs an AgentWorkflow against a single dataset row."""

    def __init__(
        self,
        workflow_cls: type,
        workflow_config: Any | None,
        proxy: EvalProxy,
    ) -> None:
        self.workflow_cls = workflow_cls
        self.workflow_config = workflow_config
        self.proxy = proxy

    async def run_single(
        self,
        prompt: list[dict[str, str]],
        rollout_id: str,
    ) -> ExecutionResult:
        from osmosis_ai.rollout_v2.context import AgentWorkflowContext, RolloutContext

        rollout_ctx = RolloutContext(
            chat_completions_url=self.proxy.url,
            rollout_id=rollout_id,
        )
        from osmosis_ai.rollout_v2.utils.messages import (
            map_initial_messages_to_content_blocks,
        )

        ctx = AgentWorkflowContext(
            prompt=map_initial_messages_to_content_blocks(prompt),
            config=self.workflow_config,
        )
        workflow = self.workflow_cls(self.workflow_config)

        start = time.monotonic()
        try:
            with rollout_ctx:
                await workflow.run(ctx)
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
                metrics=self.proxy.collect_metrics(rollout_id),
            )

        return ExecutionResult(
            success=True,
            samples=rollout_ctx.get_samples(),
            duration_ms=(time.monotonic() - start) * 1000,
            metrics=self.proxy.collect_metrics(rollout_id),
        )


__all__ = ["ExecutionResult", "WorkflowExecutor"]

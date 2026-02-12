"""Shared local rollout execution runner.

Used by both:
- `osmosis test`
- `osmosis eval`
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext, RolloutResult
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema
from osmosis_ai.rollout.eval.common.dataset import DatasetRow, dataset_row_to_request
from osmosis_ai.rollout.eval.common.errors import ToolValidationError

logger = logging.getLogger(__name__)


class LocalLLMClientProtocol(Protocol):
    """LLM client contract required by LocalRolloutRunner."""

    def set_tools(self, tools: List[Any]) -> None: ...

    def clear_tools(self) -> None: ...

    def reset_metrics(self) -> None: ...

    def get_metrics(self) -> Any: ...


@dataclass
class LocalRunResult:
    """Result from running a single dataset row."""

    row_index: int
    success: bool
    result: Optional[RolloutResult] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    token_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalBatchResult:
    """Aggregated results from running a batch of rows."""

    results: List[LocalRunResult]
    total: int
    passed: int
    failed: int
    total_duration_ms: float
    total_tokens: int


TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_tools(tools: List[OpenAIFunctionToolSchema]) -> None:
    """Validate tool schemas before sending to provider APIs."""
    for i, tool in enumerate(tools):
        if not tool.function:
            raise ToolValidationError(f"Tool {i}: missing 'function' field")
        if not tool.function.name:
            raise ToolValidationError(f"Tool {i}: function must have a 'name'")
        if not tool.function.name.strip():
            raise ToolValidationError(f"Tool {i}: function name cannot be empty")

        if not TOOL_NAME_PATTERN.match(tool.function.name):
            raise ToolValidationError(
                f"Tool '{tool.function.name}': name must start with letter/underscore "
                f"and contain only alphanumeric characters and underscores"
            )

        if tool.function.parameters:
            params = tool.function.parameters
            if params.type != "object":
                raise ToolValidationError(
                    f"Tool '{tool.function.name}': parameters.type must be 'object'"
                )


class LocalRolloutRunner:
    """Executes agent loops locally against dataset rows."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: LocalLLMClientProtocol,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        rollout_id_prefix: str = "local",
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.llm_client = llm_client
        self.debug = debug
        self.rollout_id_prefix = rollout_id_prefix
        self.request_metadata = dict(request_metadata or {})

        if debug_dir is not None:
            self.debug_dir: Optional[str] = debug_dir
        elif debug:
            self.debug_dir = "./local_debug"
        else:
            self.debug_dir = None

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        rollout_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> LocalRunResult:
        """Run a single dataset row."""
        overall_start = time.monotonic()

        self.llm_client.reset_metrics()
        self.llm_client.clear_tools()

        merged_metadata = dict(self.request_metadata)
        if request_metadata:
            merged_metadata.update(request_metadata)
        metadata_overrides = merged_metadata or None

        try:
            request = dataset_row_to_request(
                row=row,
                row_index=row_index,
                max_turns=max_turns,
                completion_params=completion_params,
                rollout_id_prefix=self.rollout_id_prefix,
                rollout_id=rollout_id,
                metadata_overrides=metadata_overrides,
            )

            tools = self.agent_loop.get_tools(request)
            validate_tools(tools)
            self.llm_client.set_tools(tools)

            agent_start_time = time.monotonic()
            ctx = RolloutContext(
                request=request,
                tools=tools,
                llm=self.llm_client,  # type: ignore[arg-type]
                _start_time=agent_start_time,
                _debug_dir=self.debug_dir,
            )

            result = await self.agent_loop.run(ctx)
            metrics = self.llm_client.get_metrics()

            return LocalRunResult(
                row_index=row_index,
                success=(result.status == "COMPLETED"),
                result=result,
                duration_ms=(time.monotonic() - overall_start) * 1000,
                token_usage={
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.response_tokens,
                    "total_tokens": metrics.prompt_tokens + metrics.response_tokens,
                    "num_llm_calls": metrics.num_llm_calls,
                },
            )

        except ToolValidationError as e:
            logger.error("Tool validation error for row %d: %s", row_index, e)
            return LocalRunResult(
                row_index=row_index,
                success=False,
                error=f"Tool validation error: {e}",
                duration_ms=(time.monotonic() - overall_start) * 1000,
            )

        except Exception as e:
            logger.exception("Error running row %d", row_index)
            return LocalRunResult(
                row_index=row_index,
                success=False,
                error=str(e),
                duration_ms=(time.monotonic() - overall_start) * 1000,
            )

        finally:
            self.llm_client.clear_tools()

    async def run_batch(
        self,
        rows: List[DatasetRow],
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[int, int, LocalRunResult], None]] = None,
        start_index: int = 0,
    ) -> LocalBatchResult:
        """Run multiple rows sequentially."""
        results: List[LocalRunResult] = []
        total_start = time.monotonic()

        for i, row in enumerate(rows):
            row_index = start_index + i
            result = await self.run_single(
                row=row,
                row_index=row_index,
                max_turns=max_turns,
                completion_params=completion_params,
            )
            results.append(result)

            if on_progress:
                on_progress(i + 1, len(rows), result)

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in results)

        return LocalBatchResult(
            results=results,
            total=len(results),
            passed=passed,
            failed=failed,
            total_duration_ms=(time.monotonic() - total_start) * 1000,
            total_tokens=total_tokens,
        )

__all__ = [
    "LocalBatchResult",
    "LocalLLMClientProtocol",
    "LocalRolloutRunner",
    "LocalRunResult",
    "validate_tools",
]

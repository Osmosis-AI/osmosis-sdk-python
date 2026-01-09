"""Test runner for executing agent loops against datasets.

Converts DatasetRow -> RolloutRequest, validates tools, and runs agent_loop.run(ctx).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from osmosis_ai.rollout.core.base import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
)
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema
from osmosis_ai.rollout.test_mode.dataset import DatasetRow, dataset_row_to_request
from osmosis_ai.rollout.test_mode.exceptions import ToolValidationError
from osmosis_ai.rollout.test_mode.external_llm_client import ExternalLLMClient

logger = logging.getLogger(__name__)


@dataclass
class LocalTestRunResult:
    """Result from running a single test row.

    Attributes:
        row_index: Index of the row in the dataset.
        success: Whether the test passed (status == "COMPLETED").
        result: RolloutResult if execution completed.
        error: Error message if execution failed.
        duration_ms: Total execution time in milliseconds.
        token_usage: Token usage statistics from the LLM client.
    """

    row_index: int
    success: bool
    result: Optional[RolloutResult] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    token_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalTestBatchResult:
    """Aggregated results from running a batch of tests.

    Attributes:
        results: Individual test results.
        total: Total number of tests.
        passed: Number of passed tests.
        failed: Number of failed tests.
        total_duration_ms: Total execution time.
        total_tokens: Total tokens used.
    """

    results: List[LocalTestRunResult]
    total: int
    passed: int
    failed: int
    total_duration_ms: float
    total_tokens: int


# Valid tool name pattern (alphanumeric + underscore, starting with letter/underscore)
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_tools(tools: List[OpenAIFunctionToolSchema]) -> None:
    """Validate tool schemas before sending to LLM provider.

    Catches common errors early with clear messages, rather than
    letting them fail at the LLM API with cryptic errors.

    Args:
        tools: Tool schemas returned by agent_loop.get_tools()

    Raises:
        ToolValidationError: If tool schema is invalid.
    """
    for i, tool in enumerate(tools):
        # Check tool has required fields
        if not tool.function:
            raise ToolValidationError(f"Tool {i}: missing 'function' field")
        if not tool.function.name:
            raise ToolValidationError(f"Tool {i}: function must have a 'name'")
        if not tool.function.name.strip():
            raise ToolValidationError(f"Tool {i}: function name cannot be empty")

        # Check name format (alphanumeric + underscore, common LLM requirement)
        if not TOOL_NAME_PATTERN.match(tool.function.name):
            raise ToolValidationError(
                f"Tool '{tool.function.name}': name must start with letter/underscore "
                f"and contain only alphanumeric characters and underscores"
            )

        # Check parameters schema if present
        if tool.function.parameters:
            params = tool.function.parameters
            if params.type != "object":
                raise ToolValidationError(
                    f"Tool '{tool.function.name}': parameters.type must be 'object'"
                )


class LocalTestRunner:
    """Executes agent loop tests against dataset rows.

    Workflow: DatasetRow -> RolloutRequest -> get_tools() -> run() -> collect results.
    """

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: ExternalLLMClient,
        debug: bool = False,
        debug_dir: Optional[str] = None,
    ) -> None:
        """Initialize the test runner.

        Args:
            agent_loop: Agent loop instance to test.
            llm_client: Test LLM client instance.
            debug: Enable debug logging.
            debug_dir: Directory for debug output files. If not specified and
                       debug=True, defaults to "./test_debug".
        """
        self.agent_loop = agent_loop
        self.llm_client = llm_client
        self.debug = debug
        # Resolve debug_dir: use explicit value, or default when debug is enabled
        if debug_dir is not None:
            self.debug_dir: Optional[str] = debug_dir
        elif debug:
            self.debug_dir = "./test_debug"
        else:
            self.debug_dir = None

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
    ) -> LocalTestRunResult:
        """Run a single test row.

        Args:
            row: Dataset row to test.
            row_index: Index of the row (for logging and result).
            max_turns: Maximum agent turns.
            completion_params: LLM sampling parameters.

        Returns:
            LocalTestRunResult with execution status and metrics.
        """
        overall_start = time.monotonic()

        # Reset client state for this row
        self.llm_client.reset_metrics()
        self.llm_client.clear_tools()

        try:
            # 1. Convert dataset row to RolloutRequest
            request = dataset_row_to_request(
                row=row,
                row_index=row_index,
                max_turns=max_turns,
                completion_params=completion_params,
            )

            # 2. Get tools from agent
            tools = self.agent_loop.get_tools(request)

            # 3. Validate tool schemas
            validate_tools(tools)

            # 4. Inject tools into client
            self.llm_client.set_tools(tools)

            # 5. Start timing AFTER preparation (matches production mode)
            agent_start_time = time.monotonic()

            # 6. Create standard RolloutContext with ExternalLLMClient
            ctx = RolloutContext(
                request=request,
                tools=tools,
                llm=self.llm_client,
                _start_time=agent_start_time,
                _debug_dir=self.debug_dir,  # Already resolved in __init__
            )

            # 7. Run agent loop (uses same code path as production!)
            result = await self.agent_loop.run(ctx)

            # Get metrics
            metrics = self.llm_client.get_metrics()

            return LocalTestRunResult(
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
            return LocalTestRunResult(
                row_index=row_index,
                success=False,
                error=f"Tool validation error: {e}",
                duration_ms=(time.monotonic() - overall_start) * 1000,
            )

        except Exception as e:
            logger.exception("Error running row %d", row_index)
            return LocalTestRunResult(
                row_index=row_index,
                success=False,
                error=str(e),
                duration_ms=(time.monotonic() - overall_start) * 1000,
            )

        finally:
            # Always clear tools after row completion
            self.llm_client.clear_tools()

    async def run_batch(
        self,
        rows: List[DatasetRow],
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[int, int, LocalTestRunResult], None]] = None,
        start_index: int = 0,
    ) -> LocalTestBatchResult:
        """Run multiple test rows sequentially.

        Args:
            rows: List of dataset rows to test.
            max_turns: Maximum agent turns per row.
            completion_params: LLM sampling parameters.
            on_progress: Optional callback called after each row.
                         Arguments: (current_index, total_count, result)
            start_index: Starting row index for row numbering (e.g., from --offset).
                         Used for correct row_index in output, rollout IDs, and metadata.

        Returns:
            LocalTestBatchResult with aggregated statistics.
        """
        results: List[LocalTestRunResult] = []
        total_start = time.monotonic()

        for i, row in enumerate(rows):
            # Calculate absolute row index (for debugging, rollout IDs, metadata)
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

        # Aggregate results
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        total_tokens = sum(
            r.token_usage.get("total_tokens", 0) for r in results
        )

        return LocalTestBatchResult(
            results=results,
            total=len(results),
            passed=passed,
            failed=failed,
            total_duration_ms=(time.monotonic() - total_start) * 1000,
            total_tokens=total_tokens,
        )

__all__ = [
    "LocalTestBatchResult",
    "LocalTestRunResult",
    "LocalTestRunner",
    "validate_tools",
]

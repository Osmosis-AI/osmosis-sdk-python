"""CLI command for test mode.

Run agent loop tests against datasets using LLM providers via LiteLLM.

Examples:
    # Batch mode
    osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --model gpt-4o
    osmosis test ... --model anthropic/claude-sonnet-4-20250514

    # Interactive mode
    osmosis test ... --interactive
    osmosis test ... --interactive --row 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop
from osmosis_ai.rollout.console import Console

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop
    from osmosis_ai.rollout.test_mode.dataset import DatasetRow
    from osmosis_ai.rollout.test_mode.external_llm_client import ExternalLLMClient
    from osmosis_ai.rollout.test_mode.runner import BatchTestResult, LocalTestRunResult

logger = logging.getLogger(__name__)


@dataclass
class _SetupResult:
    """Result of setup phase containing initialized components."""

    agent_loop: "RolloutAgentLoop"
    llm_client: "ExternalLLMClient"
    rows: List["DatasetRow"]
    completion_params: Dict[str, Any]


# Alias for internal use
_load_agent_loop = load_agent_loop


def _format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m{seconds:.1f}s"


def _format_tokens(tokens: int) -> str:
    """Format token count with comma separators."""
    return f"{tokens:,}"


class TestCommand:
    """Handler for `osmosis test`."""

    def __init__(self) -> None:
        """Initialize the test command."""
        self.console = Console()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for test command."""
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
            "--agent",
            dest="module",
            required=True,
            help=(
                "Module path to the agent loop in format 'module:attribute'. "
                "Example: 'my_agent:MyAgentLoop'."
            ),
        )

        parser.add_argument(
            "-d",
            "--dataset",
            dest="dataset",
            required=True,
            help="Path to dataset file (.json, .jsonl, or .parquet).",
        )

        parser.add_argument(
            "--model",
            dest="model",
            default="gpt-4o",
            help=(
                "Model name to use. Can be:\n"
                "  - Simple name: 'gpt-4o' (auto-prefixed to 'openai/gpt-4o')\n"
                "  - LiteLLM format: 'provider/model' (e.g., 'anthropic/claude-sonnet-4-20250514')\n"
                "Default: gpt-4o"
            ),
        )

        parser.add_argument(
            "--limit",
            dest="limit",
            type=int,
            default=None,
            help="Maximum number of rows to test",
        )

        parser.add_argument(
            "--offset",
            dest="offset",
            type=int,
            default=0,
            help="Number of rows to skip",
        )

        parser.add_argument(
            "--api-key",
            dest="api_key",
            default=None,
            help="API key for the LLM provider (or use env var)",
        )

        parser.add_argument(
            "--base-url",
            dest="base_url",
            default=None,
            help="Base URL for OpenAI-compatible APIs (e.g., http://localhost:8000/v1)",
        )

        parser.add_argument(
            "--max-turns",
            dest="max_turns",
            type=int,
            default=10,
            help="Maximum agent turns per row (default: 10)",
        )

        parser.add_argument(
            "--max-tokens",
            dest="max_tokens",
            type=int,
            default=None,
            help="Maximum tokens per completion",
        )

        parser.add_argument(
            "--temperature",
            dest="temperature",
            type=float,
            default=None,
            help="LLM temperature",
        )

        parser.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            default=False,
            help="Enable debug output",
        )

        parser.add_argument(
            "--output",
            "-o",
            dest="output",
            default=None,
            help="Output results to JSON file",
        )

        parser.add_argument(
            "--quiet",
            "-q",
            dest="quiet",
            action="store_true",
            default=False,
            help="Suppress progress output",
        )

        parser.add_argument(
            "--interactive",
            "-i",
            dest="interactive",
            action="store_true",
            default=False,
            help="Enable interactive mode for step-by-step execution",
        )

        parser.add_argument(
            "--row",
            dest="row",
            type=int,
            default=None,
            help=(
                "Initial row to test in interactive mode (absolute index in dataset). "
                "With --offset 50 --limit 10, valid range is 50-59."
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        """Run the test command."""
        return asyncio.run(self._run_async(args))

    def _validate_args(self, args: argparse.Namespace) -> Optional[str]:
        """Validate argument combinations.

        Returns:
            Error message if validation fails, None otherwise.
        """
        if args.row is not None and not args.interactive:
            return "--row can only be used with --interactive mode"
        return None

    def _print_header(self, args: argparse.Namespace) -> None:
        """Print CLI header with version and mode info."""
        if args.quiet:
            return

        from osmosis_ai.consts import PACKAGE_VERSION

        mode_suffix = " (Interactive Mode)" if args.interactive else ""
        self.console.print(f"osmosis-rollout-test v{PACKAGE_VERSION}{mode_suffix}", style="bold")
        self.console.print()

    def _load_agent(
        self, args: argparse.Namespace
    ) -> Tuple[Optional["RolloutAgentLoop"], Optional[str]]:
        """Load the agent loop from module path.

        Returns:
            Tuple of (agent_loop, error_message). If successful, error is None.
        """
        if not args.quiet:
            self.console.print(f"Loading agent: {args.module}")

        try:
            agent_loop = _load_agent_loop(args.module)
        except CLIError as e:
            return None, str(e)

        if not args.quiet:
            self.console.print(f"  Agent name: {agent_loop.name}")

        return agent_loop, None

    def _load_dataset(
        self, args: argparse.Namespace
    ) -> Tuple[Optional[List["DatasetRow"]], Optional[str]]:
        """Load and validate the dataset.

        Returns:
            Tuple of (rows, error_message). If successful, error is None.
        """
        from osmosis_ai.rollout.test_mode.dataset import DatasetReader
        from osmosis_ai.rollout.test_mode.exceptions import (
            DatasetParseError,
            DatasetValidationError,
        )

        if not args.quiet:
            self.console.print(f"Loading dataset: {args.dataset}")

        try:
            reader = DatasetReader(args.dataset)
            total_rows = len(reader)
            rows = reader.read(limit=args.limit, offset=args.offset)
        except FileNotFoundError as e:
            return None, str(e)
        except (DatasetParseError, DatasetValidationError) as e:
            return None, str(e)

        if not rows:
            return None, "No rows to test"

        if not args.quiet:
            if args.limit:
                self.console.print(f"  Total rows: {total_rows} (testing {len(rows)})")
            else:
                self.console.print(f"  Total rows: {len(rows)}")

        return rows, None

    def _create_llm_client(
        self, args: argparse.Namespace
    ) -> Tuple[Optional["ExternalLLMClient"], Optional[str]]:
        """Initialize the LLM client.

        Returns:
            Tuple of (llm_client, error_message). If successful, error is None.
        """
        from osmosis_ai.rollout.test_mode.exceptions import ProviderError
        from osmosis_ai.rollout.test_mode.external_llm_client import ExternalLLMClient

        model = args.model
        if not args.quiet:
            provider_name = model.split("/")[0] if "/" in model else "openai"
            self.console.print(f"Initializing provider: {provider_name}")

        try:
            llm_client = ExternalLLMClient(
                model=model,
                api_key=args.api_key,
                api_base=args.base_url,
            )
        except ProviderError as e:
            return None, str(e)

        if not args.quiet:
            model_name = getattr(llm_client, "model", "default")
            self.console.print(f"  Model: {model_name}")

        return llm_client, None

    def _build_completion_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Build completion parameters from CLI arguments."""
        params: Dict[str, Any] = {}
        if args.temperature is not None:
            params["temperature"] = args.temperature
        if args.max_tokens is not None:
            params["max_tokens"] = args.max_tokens
        return params

    async def _run_interactive_mode(
        self,
        args: argparse.Namespace,
        setup: _SetupResult,
    ) -> int:
        """Run interactive mode session.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        from osmosis_ai.rollout.test_mode.interactive import InteractiveRunner

        interactive_runner = InteractiveRunner(
            agent_loop=setup.agent_loop,
            llm_client=setup.llm_client,
            debug=args.debug,
        )

        self.console.print()
        try:
            async with setup.llm_client:
                await interactive_runner.run_interactive_session(
                    rows=setup.rows,
                    max_turns=args.max_turns,
                    completion_params=setup.completion_params if setup.completion_params else None,
                    initial_row=args.row,
                    row_offset=args.offset,
                )
        except Exception as e:
            self.console.print_error(f"Error during interactive session: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        return 0

    async def _run_batch_mode(
        self,
        args: argparse.Namespace,
        setup: _SetupResult,
    ) -> "BatchTestResult":
        """Run batch mode tests.

        Returns:
            BatchTestResult with all test results.

        Raises:
            Exception: If test execution fails (propagates to caller for traceback).
        """
        from osmosis_ai.rollout.test_mode.runner import LocalTestRunner

        runner = LocalTestRunner(
            agent_loop=setup.agent_loop,
            llm_client=setup.llm_client,
            debug=args.debug,
        )

        def on_progress(
            current: int, total: int, result: "LocalTestRunResult"
        ) -> None:
            if args.quiet:
                return

            status_style = "green" if result.success else "red"
            status = "OK" if result.success else "FAILED"
            duration = _format_duration(result.duration_ms)
            tokens = result.token_usage.get("total_tokens", 0)

            error_suffix = ""
            if not result.success and result.error:
                error_text = result.error.replace("\n", " ")
                error_msg = error_text[:47] + "..." if len(error_text) > 50 else error_text
                error_suffix = f" - {error_msg}"

            status_styled = self.console.format_styled(status, status_style)
            self.console.print(
                f"[{current}/{total}] Row {result.row_index}: {status_styled} "
                f"({duration}, {_format_tokens(tokens)} tokens){error_suffix}"
            )

        if not args.quiet:
            self.console.print()
            self.console.print("Running tests...")

        async with setup.llm_client:
            batch_result = await runner.run_batch(
                rows=setup.rows,
                max_turns=args.max_turns,
                completion_params=setup.completion_params if setup.completion_params else None,
                on_progress=on_progress,
                start_index=args.offset,
            )

        return batch_result

    def _print_summary(self, batch_result: "BatchTestResult") -> None:
        """Print batch test summary."""
        self.console.print()
        self.console.print("Summary:", style="bold")
        self.console.print(f"  Total: {batch_result.total}")

        passed_style = "green" if batch_result.passed > 0 else None
        failed_style = "red" if batch_result.failed > 0 else None

        passed_text = (
            self.console.format_styled(str(batch_result.passed), passed_style)
            if passed_style
            else str(batch_result.passed)
        )
        failed_text = (
            self.console.format_styled(str(batch_result.failed), failed_style)
            if failed_style
            else str(batch_result.failed)
        )

        self.console.print(f"  Passed: {passed_text}")
        self.console.print(f"  Failed: {failed_text}")
        self.console.print(f"  Duration: {_format_duration(batch_result.total_duration_ms)}")
        self.console.print(f"  Total tokens: {_format_tokens(batch_result.total_tokens)}")

    def _write_output(
        self, args: argparse.Namespace, batch_result: "BatchTestResult"
    ) -> None:
        """Write test results to output file."""
        if not args.output:
            return

        output_data = {
            "summary": {
                "total": batch_result.total,
                "passed": batch_result.passed,
                "failed": batch_result.failed,
                "total_duration_ms": batch_result.total_duration_ms,
                "total_tokens": batch_result.total_tokens,
            },
            "results": [
                {
                    "row_index": r.row_index,
                    "success": r.success,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                    "token_usage": r.token_usage,
                    "reward": r.result.reward if r.result else None,
                    "finish_reason": r.result.finish_reason if r.result else None,
                }
                for r in batch_result.results
            ],
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        if not args.quiet:
            self.console.print(f"\nResults written to: {args.output}")

    async def _run_async(self, args: argparse.Namespace) -> int:
        """Async implementation of the test command."""
        # Validate arguments
        if error := self._validate_args(args):
            self.console.print_error(f"Error: {error}")
            return 1

        # Print header
        self._print_header(args)

        # Load agent loop
        agent_loop, error = self._load_agent(args)
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert agent_loop is not None

        # Load dataset
        rows, error = self._load_dataset(args)
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert rows is not None

        # Initialize LLM client
        llm_client, error = self._create_llm_client(args)
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert llm_client is not None

        # Build completion params
        completion_params = self._build_completion_params(args)

        # Create setup result
        setup = _SetupResult(
            agent_loop=agent_loop,
            llm_client=llm_client,
            rows=rows,
            completion_params=completion_params,
        )

        # Interactive mode
        if args.interactive:
            return await self._run_interactive_mode(args, setup)

        # Batch mode
        try:
            batch_result = await self._run_batch_mode(args, setup)
        except Exception as e:
            self.console.print_error(f"Error during test execution: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        # Print summary and write output
        if not args.quiet:
            self._print_summary(batch_result)

        self._write_output(args, batch_result)

        # Return exit code based on failures
        return 1 if batch_result.failed > 0 else 0


__all__ = ["TestCommand"]

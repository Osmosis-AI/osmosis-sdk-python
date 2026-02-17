"""CLI command for test mode.

Run agent loop tests against datasets using LLM providers via LiteLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.eval.common.cli import (
    build_completion_params,
    create_llm_client,
    format_duration,
    format_tokens,
    load_agent,
    load_dataset_rows,
    load_mcp_agent,
    verify_llm_client,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop
    from osmosis_ai.rollout.eval.common.dataset import DatasetRow
    from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
    from osmosis_ai.rollout.eval.test_mode.runner import (
        LocalTestBatchResult,
        LocalTestRunResult,
    )


logger = logging.getLogger(__name__)


@dataclass
class _SetupResult:
    """Result of setup phase containing initialized components."""

    agent_loop: RolloutAgentLoop
    llm_client: ExternalLLMClient
    rows: list[DatasetRow]
    completion_params: dict[str, Any]


class TestCommand:
    """Handler for `osmosis test`."""

    def __init__(self) -> None:
        self.console = Console()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for test command."""
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
            "--agent",
            dest="module",
            default=None,
            help=(
                "Module path to the agent loop in format 'module:attribute'. "
                "Example: 'my_agent:MyAgentLoop'."
            ),
        )

        parser.add_argument(
            "--mcp",
            dest="mcp",
            default=None,
            help=(
                "Path to MCP tools directory (must contain main.py with a FastMCP instance). "
                "Mutually exclusive with -m/--module."
            ),
        )

        parser.add_argument(
            "-d",
            "--dataset",
            dest="dataset",
            required=True,
            help="Path to dataset file (.parquet recommended, .jsonl, or .csv).",
        )

        parser.add_argument(
            "--model",
            dest="model",
            default="gpt-5-mini",
            help=(
                "Model name to use. Can be:\n"
                "  - Simple name: 'gpt-5-mini'\n"
                "  - LiteLLM format: 'provider/model' (e.g., 'anthropic/claude-sonnet-4-5')\n"
                "  - Any name with --base-url (e.g., 'Qwen/Qwen3-0.6B')\n"
                "Default: gpt-5-mini"
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
        return asyncio.run(self._run_async(args))

    def _validate_args(self, args: argparse.Namespace) -> str | None:
        if args.module and args.mcp:
            return "--module and --mcp are mutually exclusive."
        if not args.module and not args.mcp:
            return "Either --module (-m) or --mcp is required."
        if args.row is not None and not args.interactive:
            return "--row can only be used with --interactive mode"
        return None

    def _print_header(self, args: argparse.Namespace) -> None:
        if args.quiet:
            return

        from osmosis_ai.consts import PACKAGE_VERSION

        mode_suffix = " (Interactive Mode)" if args.interactive else ""
        self.console.print(
            f"osmosis-rollout-test v{PACKAGE_VERSION}{mode_suffix}", style="bold"
        )
        self.console.print()

    async def _run_interactive_mode(
        self,
        args: argparse.Namespace,
        setup: _SetupResult,
    ) -> int:
        from osmosis_ai.rollout.eval.test_mode.interactive import InteractiveRunner

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
                    completion_params=setup.completion_params
                    if setup.completion_params
                    else None,
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
    ) -> LocalTestBatchResult:
        from osmosis_ai.rollout.eval.test_mode.runner import LocalTestRunner

        runner = LocalTestRunner(
            agent_loop=setup.agent_loop,
            llm_client=setup.llm_client,
            debug=args.debug,
        )

        def on_progress(current: int, total: int, result: LocalTestRunResult) -> None:
            if args.quiet:
                return

            status_style = "green" if result.success else "red"
            status = "OK" if result.success else "FAILED"
            duration = format_duration(result.duration_ms)
            tokens = result.token_usage.get("total_tokens", 0)

            error_suffix = ""
            if not result.success and result.error:
                error_text = result.error.replace("\n", " ")
                error_msg = (
                    error_text[:47] + "..." if len(error_text) > 50 else error_text
                )
                error_suffix = f" - {error_msg}"

            status_styled = self.console.format_styled(status, status_style)
            self.console.print(
                f"[{current}/{total}] Row {result.row_index}: {status_styled} "
                f"({duration}, {format_tokens(tokens)} tokens){error_suffix}"
            )

        if not args.quiet:
            self.console.print()
            self.console.print("Running tests...")

        async with setup.llm_client:
            batch_result = await runner.run_batch(
                rows=setup.rows,
                max_turns=args.max_turns,
                completion_params=setup.completion_params
                if setup.completion_params
                else None,
                on_progress=on_progress,
                start_index=args.offset,
            )

        return batch_result

    def _print_summary(self, batch_result: LocalTestBatchResult) -> None:
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
        self.console.print(
            f"  Duration: {format_duration(batch_result.total_duration_ms)}"
        )
        self.console.print(
            f"  Total tokens: {format_tokens(batch_result.total_tokens)}"
        )

        if batch_result.stopped_early:
            reason = (
                f" Reason: {batch_result.stop_reason}"
                if batch_result.stop_reason
                else ""
            )
            self.console.print(
                f"  Stopped early due to systemic provider error.{reason}",
                style="red",
            )

    def _write_output(
        self, args: argparse.Namespace, batch_result: LocalTestBatchResult
    ) -> None:
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
        if args.debug:
            logging.basicConfig(level=logging.DEBUG)

        if error := self._validate_args(args):
            self.console.print_error(f"Error: {error}")
            return 1

        self._print_header(args)

        llm_client, error = create_llm_client(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            quiet=args.quiet,
            console=self.console,
        )
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert llm_client is not None

        error = await verify_llm_client(llm_client, args.quiet, self.console)
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1

        if args.mcp:
            agent_loop, error = load_mcp_agent(
                mcp_path=args.mcp,
                quiet=args.quiet,
                console=self.console,
            )
        else:
            agent_loop, error = load_agent(
                module=args.module,
                quiet=args.quiet,
                console=self.console,
            )
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1
        assert agent_loop is not None

        rows, error = load_dataset_rows(
            dataset_path=args.dataset,
            limit=args.limit,
            offset=args.offset,
            quiet=args.quiet,
            console=self.console,
            empty_error="No rows to test",
            action_label="testing",
        )
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1
        assert rows is not None

        completion_params = build_completion_params(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        setup = _SetupResult(
            agent_loop=agent_loop,
            llm_client=llm_client,
            rows=rows,
            completion_params=completion_params,
        )

        if args.interactive:
            return await self._run_interactive_mode(args, setup)

        try:
            batch_result = await self._run_batch_mode(args, setup)
        except Exception as e:
            self.console.print_error(f"Error during test execution: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        if not args.quiet:
            self._print_summary(batch_result)

        self._write_output(args, batch_result)
        return 1 if batch_result.failed > 0 else 0


__all__ = ["TestCommand"]

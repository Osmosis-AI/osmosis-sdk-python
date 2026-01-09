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
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop

if TYPE_CHECKING:
    from osmosis_ai.rollout.test_mode.runner import LocalTestRunResult

logger = logging.getLogger(__name__)


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

    async def _run_async(self, args: argparse.Namespace) -> int:
        """Async implementation of the test command."""
        # Import test mode modules
        from osmosis_ai.rollout.test_mode.dataset import DatasetReader
        from osmosis_ai.rollout.test_mode.exceptions import (
            DatasetParseError,
            DatasetValidationError,
            ProviderError,
        )
        from osmosis_ai.rollout.test_mode.external_llm_client import (
            ExternalLLMClient,
        )
        from osmosis_ai.rollout.test_mode.runner import LocalTestRunner

        # Validate --row requires --interactive
        if args.row is not None and not args.interactive:
            print(
                "Error: --row can only be used with --interactive mode",
                file=sys.stderr,
            )
            return 1

        # Print header
        is_interactive = args.interactive
        if not args.quiet:
            from osmosis_ai.consts import PACKAGE_VERSION

            mode_suffix = " (Interactive Mode)" if is_interactive else ""
            print(f"osmosis-rollout-test v{PACKAGE_VERSION}{mode_suffix}")
            print()

        # Load agent loop
        if not args.quiet:
            print(f"Loading agent: {args.module}")

        try:
            agent_loop = _load_agent_loop(args.module)
        except CLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if not args.quiet:
            print(f"  Agent name: {agent_loop.name}")

        # Load dataset
        if not args.quiet:
            print(f"Loading dataset: {args.dataset}")

        try:
            reader = DatasetReader(args.dataset)
            total_rows = len(reader)
            rows = reader.read(limit=args.limit, offset=args.offset)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except (DatasetParseError, DatasetValidationError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if not rows:
            print("Error: No rows to test", file=sys.stderr)
            return 1

        if not args.quiet:
            if args.limit:
                print(f"  Total rows: {total_rows} (testing {len(rows)})")
            else:
                print(f"  Total rows: {len(rows)}")

        # Initialize LLM client (via LiteLLM)
        model = args.model
        if not args.quiet:
            if "/" in model:
                provider_name = model.split("/")[0]
            else:
                provider_name = "openai"
            print(f"Initializing provider: {provider_name}")

        try:
            llm_client = ExternalLLMClient(
                model=model,
                api_key=args.api_key,
                api_base=args.base_url,
            )
        except ProviderError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if not args.quiet:
            model_name = getattr(llm_client, "model", "default")
            print(f"  Model: {model_name}")

        # Build completion params
        completion_params: Dict[str, Any] = {}
        if args.temperature is not None:
            completion_params["temperature"] = args.temperature
        if args.max_tokens is not None:
            completion_params["max_tokens"] = args.max_tokens

        # Interactive mode vs batch mode
        if is_interactive:
            # Interactive mode: step-by-step execution
            from osmosis_ai.rollout.test_mode.interactive import InteractiveRunner

            interactive_runner = InteractiveRunner(
                agent_loop=agent_loop,
                llm_client=llm_client,
                debug=args.debug,
            )

            print()
            try:
                async with llm_client:
                    await interactive_runner.run_interactive_session(
                        rows=rows,
                        max_turns=args.max_turns,
                        completion_params=completion_params if completion_params else None,
                        initial_row=args.row,
                        row_offset=args.offset,
                    )
            except Exception as e:
                print(f"Error during interactive session: {e}", file=sys.stderr)
                if args.debug:
                    import traceback

                    traceback.print_exc()
                return 1

            return 0

        # Batch mode: run all tests
        runner = LocalTestRunner(
            agent_loop=agent_loop,
            llm_client=llm_client,
            debug=args.debug,
        )

        # Progress callback
        def on_progress(
            current: int, total: int, result: "LocalTestRunResult"
        ) -> None:
            if args.quiet:
                return

            status = "OK" if result.success else "FAILED"
            duration = _format_duration(result.duration_ms)
            tokens = result.token_usage.get("total_tokens", 0)

            error_suffix = ""
            if not result.success and result.error:
                # Truncate error message
                error_msg = result.error[:50] + "..." if len(result.error) > 50 else result.error
                error_suffix = f" - {error_msg}"

            print(
                f"[{current}/{total}] Row {result.row_index}: {status} "
                f"({duration}, {_format_tokens(tokens)} tokens){error_suffix}"
            )

        # Run tests
        if not args.quiet:
            print()
            print("Running tests...")

        try:
            async with llm_client:
                batch_result = await runner.run_batch(
                    rows=rows,
                    max_turns=args.max_turns,
                    completion_params=completion_params if completion_params else None,
                    on_progress=on_progress,
                    start_index=args.offset,
                )
        except Exception as e:
            print(f"Error during test execution: {e}", file=sys.stderr)
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        # Print summary
        if not args.quiet:
            print()
            print("Summary:")
            print(f"  Total: {batch_result.total}")
            print(f"  Passed: {batch_result.passed}")
            print(f"  Failed: {batch_result.failed}")
            print(f"  Duration: {_format_duration(batch_result.total_duration_ms)}")
            print(f"  Total tokens: {_format_tokens(batch_result.total_tokens)}")

        # Write output file
        if args.output:
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
                print(f"\nResults written to: {args.output}")

        # Return exit code based on failures
        if batch_result.failed > 0:
            return 1
        return 0


__all__ = ["TestCommand"]

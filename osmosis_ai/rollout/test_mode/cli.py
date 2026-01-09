"""CLI command for test mode.

This module provides the TestCommand for running agent loop tests
against datasets using cloud LLM providers.

Example:
    osmosis test \\
        --agent my_agent:MyAgentLoop \\
        --dataset ./test_data.jsonl \\
        --provider openai \\
        --model gpt-4o-mini \\
        --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from osmosis_ai.rollout.core.base import RolloutAgentLoop

if TYPE_CHECKING:
    from osmosis_ai.rollout.test_mode.runner import TestRunResult

logger = logging.getLogger(__name__)


class CLIError(Exception):
    """CLI-specific error."""

    pass


def _load_agent_loop(module_path: str) -> RolloutAgentLoop:
    """Load an agent loop from a module path.

    Args:
        module_path: Path in format "module.path:attribute_name"
                     e.g., "my_agent:agent_loop" or "mypackage.agents:MyAgent"

    Returns:
        RolloutAgentLoop instance.

    Raises:
        CLIError: If the module or attribute cannot be loaded.
    """
    if ":" not in module_path:
        raise CLIError(
            f"Invalid module path '{module_path}'. "
            "Expected format: 'module.path:attribute_name' "
            "(e.g., 'my_agent:agent_loop' or 'mypackage.agents:MyAgent')"
        )

    module_name, attr_name = module_path.rsplit(":", 1)

    # Add current directory to sys.path if not already there
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise CLIError(f"Cannot import module '{module_name}': {e}")

    try:
        agent_loop = getattr(module, attr_name)
    except AttributeError:
        raise CLIError(
            f"Module '{module_name}' has no attribute '{attr_name}'. "
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    # If it's a class, instantiate it
    if isinstance(agent_loop, type):
        if not issubclass(agent_loop, RolloutAgentLoop):
            raise CLIError(
                f"'{attr_name}' is a class but not a RolloutAgentLoop subclass"
            )
        try:
            agent_loop = agent_loop()
        except Exception as e:
            raise CLIError(f"Cannot instantiate '{attr_name}': {e}")

    # Validate it's a RolloutAgentLoop instance
    if not isinstance(agent_loop, RolloutAgentLoop):
        raise CLIError(
            f"'{attr_name}' must be a RolloutAgentLoop instance or subclass, "
            f"got {type(agent_loop).__name__}"
        )

    return agent_loop


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
                "Example: 'my_agent:MyAgentLoop'"
            ),
        )

        parser.add_argument(
            "-d",
            "--dataset",
            dest="dataset",
            required=True,
            help="Path to dataset file (.json, .jsonl, or .parquet)",
        )

        parser.add_argument(
            "-p",
            "--provider",
            dest="provider",
            required=True,
            help="LLM provider name (e.g., 'openai')",
        )

        parser.add_argument(
            "--model",
            dest="model",
            default=None,
            help="Model name to use (default: provider-specific)",
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
        from osmosis_ai.rollout.test_mode.providers import get_provider, list_providers
        from osmosis_ai.rollout.test_mode.runner import TestRunner

        # Print header
        if not args.quiet:
            from osmosis_ai.consts import PACKAGE_VERSION

            print(f"osmosis-rollout-test v{PACKAGE_VERSION}")
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

        # Initialize provider
        if not args.quiet:
            print(f"Initializing provider: {args.provider}")

        try:
            provider_class = get_provider(args.provider)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            available = list_providers()
            if available:
                print(f"Available providers: {', '.join(available)}", file=sys.stderr)
            return 1

        # Build provider kwargs
        provider_kwargs: Dict[str, Any] = {}
        if args.api_key:
            provider_kwargs["api_key"] = args.api_key
        if args.model:
            provider_kwargs["model"] = args.model
        if args.base_url:
            provider_kwargs["base_url"] = args.base_url

        try:
            llm_client = provider_class(**provider_kwargs)
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

        # Create runner
        runner = TestRunner(
            agent_loop=agent_loop,
            llm_client=llm_client,
            debug=args.debug,
        )

        # Progress callback
        def on_progress(
            current: int, total: int, result: "TestRunResult"
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

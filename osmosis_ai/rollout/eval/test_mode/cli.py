"""CLI command for test mode.

Run agent loop tests against datasets using LLM providers via LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import typer

from osmosis_ai.cli.console import Console

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop
    from osmosis_ai.rollout.eval.common.dataset import DatasetRow
    from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
    from osmosis_ai.rollout.eval.test_mode.runner import (
        LocalTestBatchResult,
        LocalTestRunResult,
    )

app: typer.Typer = typer.Typer(help="Test a RolloutAgentLoop against a dataset.")


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
        self.console: Console = Console()

    def run(self, **kwargs: Any) -> int:
        args = SimpleNamespace(**kwargs)
        return asyncio.run(self._run_async(args))

    def _validate_args(self, args: Any) -> str | None:
        if args.module and args.mcp:
            return "--module and --mcp are mutually exclusive."
        if not args.module and not args.mcp:
            return "Either --module (-m) or --mcp is required."
        if args.row is not None and not args.interactive:
            return "--row can only be used with --interactive mode"
        return None

    def _print_header(self, args: Any) -> None:
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
        args: Any,
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
        args: Any,
        setup: _SetupResult,
    ) -> LocalTestBatchResult:
        from osmosis_ai.rollout.eval.test_mode.runner import LocalTestRunner

        runner = LocalTestRunner(
            agent_loop=setup.agent_loop,
            llm_client=setup.llm_client,
            debug=args.debug,
        )

        from osmosis_ai.rollout.eval.common.cli import format_duration, format_tokens

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
        from osmosis_ai.rollout.eval.common.cli import format_duration, format_tokens

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

    def _write_output(self, args: Any, batch_result: LocalTestBatchResult) -> None:
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

    async def _run_async(self, args: Any) -> int:
        from osmosis_ai.rollout.eval.common.cli import (
            build_completion_params,
            create_llm_client,
            load_agent,
            load_dataset_rows,
            load_mcp_agent,
            verify_llm_client,
        )

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


@app.callback(invoke_without_command=True)
def test(
    module: str | None = typer.Option(
        None, "-m", "--module", "--agent", help="Module path 'module:attribute'."
    ),
    mcp: str | None = typer.Option(None, "--mcp", help="Path to MCP tools directory."),
    dataset: str = typer.Option(..., "-d", "--dataset", help="Path to dataset file."),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name to use."),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of rows to test."
    ),
    offset: int = typer.Option(0, "--offset", help="Number of rows to skip."),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the LLM provider."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for OpenAI-compatible APIs."
    ),
    max_turns: int = typer.Option(
        10, "--max-turns", help="Maximum agent turns per row."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens per completion."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="LLM temperature."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output."),
    output: str | None = typer.Option(
        None, "-o", "--output", help="Output results to JSON file."
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Suppress progress output."
    ),
    interactive: bool = typer.Option(
        False, "-i", "--interactive", help="Interactive mode."
    ),
    row: int | None = typer.Option(
        None, "--row", help="Initial row in interactive mode."
    ),
) -> None:
    """Test a RolloutAgentLoop against a dataset."""
    rc = TestCommand().run(
        module=module,
        mcp=mcp,
        dataset=dataset,
        model=model,
        limit=limit,
        offset=offset,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        debug=debug,
        output=output,
        quiet=quiet,
        interactive=interactive,
        row=row,
    )
    if rc:
        raise typer.Exit(rc)


__all__ = ["TestCommand"]

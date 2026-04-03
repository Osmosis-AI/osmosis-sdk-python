"""CLI command for test mode.

Run agent workflow tests against datasets using LLM providers via LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console

if TYPE_CHECKING:
    from osmosis_ai.eval.common.dataset import DatasetRow
    from osmosis_ai.eval.common.llm_client import ExternalLLMClient
    from osmosis_ai.eval.test_mode.runner import (
        TestBatchResult,
        TestRunResult,
    )


@dataclass
class _SetupResult:
    """Result of setup phase containing initialized components."""

    workflow_cls: type
    workflow_config: Any
    llm_client: ExternalLLMClient
    rows: list[DatasetRow]


class TestCommand:
    """Handler for `osmosis rollout test`."""

    def __init__(self) -> None:
        self.console: Console = Console()

    def run(self, **kwargs: Any) -> int:
        args = SimpleNamespace(**kwargs)
        return asyncio.run(self._run_async(args))

    def _validate_args(self, args: Any) -> str | None:
        if not args.module:
            return "--module (-m) is required."
        return None

    def _print_header(self, args: Any) -> None:
        if args.quiet:
            return

        from osmosis_ai.consts import PACKAGE_VERSION

        self.console.print(f"osmosis-rollout-test v{PACKAGE_VERSION}", style="bold")
        self.console.print()

    async def _run_batch_mode(
        self,
        args: Any,
        setup: _SetupResult,
    ) -> TestBatchResult:
        from osmosis_ai.eval.executor import WorkflowExecutor
        from osmosis_ai.eval.proxy import EvalProxy
        from osmosis_ai.eval.test_mode.runner import TestRunner

        trace_dir = "./test_traces" if args.debug else None
        proxy = EvalProxy(client=setup.llm_client, trace_dir=trace_dir)
        await proxy.start()

        try:
            executor = WorkflowExecutor(
                workflow_cls=setup.workflow_cls,
                workflow_config=setup.workflow_config,
                proxy=proxy,
            )
            runner = TestRunner(executor=executor)

            from osmosis_ai.eval.common.cli import (
                format_duration,
                format_tokens,
                truncate_error,
            )

            def on_progress(current: int, total: int, result: TestRunResult) -> None:
                if args.quiet:
                    return

                status_style = "green" if result.success else "red"
                status = "OK" if result.success else "FAILED"
                duration = format_duration(result.duration_ms)
                tokens = result.token_usage.get("total_tokens", 0)

                error_suffix = ""
                if not result.success and result.error:
                    error_suffix = f" - {truncate_error(result.error)}"

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
                    on_progress=on_progress,
                    start_index=args.offset,
                )
        finally:
            await proxy.stop()

        return batch_result

    def _print_summary(self, batch_result: TestBatchResult) -> None:
        from osmosis_ai.eval.common.cli import format_duration, format_tokens

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
            self.console.print(
                "  Stopped early: repeated failures detected.",
                style="red",
            )

    def _write_output(self, args: Any, batch_result: TestBatchResult) -> None:
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
        from osmosis_ai.eval.common.cli import (
            create_llm_client,
            load_dataset_rows,
            load_workflow,
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

        workflow_cls, workflow_config, error = load_workflow(
            module=args.module,
            quiet=args.quiet,
            console=self.console,
        )
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1
        assert workflow_cls is not None

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

        setup = _SetupResult(
            workflow_cls=workflow_cls,
            workflow_config=workflow_config,
            llm_client=llm_client,
            rows=rows,
        )

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

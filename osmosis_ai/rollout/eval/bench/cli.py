"""CLI command for bench mode.

Run agent loop benchmarks against datasets with eval functions and pass@n support.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.eval.common.cli import (
    build_completion_params,
    create_llm_client,
    format_duration,
    load_agent,
    load_dataset_rows,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.eval.bench.eval_fn import EvalFnWrapper
    from osmosis_ai.rollout.eval.bench.runner import BenchResult, BenchRunResult


logger = logging.getLogger(__name__)


class BenchCommand:
    """Handler for `osmosis bench`."""

    def __init__(self) -> None:
        self.console = Console()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
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
            required=True,
            help=(
                "Model to benchmark. Use with --base-url for trained model "
                "endpoints (e.g., 'my-finetuned-model'), or LiteLLM provider "
                "format for comparison baselines (e.g., 'openai/gpt-4o')."
            ),
        )

        parser.add_argument(
            "--eval-fn",
            dest="eval_fns",
            action="append",
            required=True,
            metavar="MODULE:FN",
            help=(
                "Eval function in 'module:function' format. "
                "Can be specified multiple times."
            ),
        )

        parser.add_argument(
            "--n",
            dest="n_runs",
            type=int,
            default=1,
            help="Number of runs per row for pass@n (default: 1).",
        )

        parser.add_argument(
            "--pass-threshold",
            dest="pass_threshold",
            type=float,
            default=1.0,
            help="Score >= threshold counts as pass for pass@k (default: 1.0).",
        )

        parser.add_argument(
            "--max-turns",
            dest="max_turns",
            type=int,
            default=10,
            help="Maximum agent turns per run (default: 10).",
        )

        parser.add_argument(
            "--temperature",
            dest="temperature",
            type=float,
            default=None,
            help="LLM sampling temperature.",
        )

        parser.add_argument(
            "--max-tokens",
            dest="max_tokens",
            type=int,
            default=None,
            help="Maximum tokens per completion.",
        )

        parser.add_argument(
            "--api-key",
            dest="api_key",
            default=None,
            help="API key for the LLM provider (or use env var).",
        )

        parser.add_argument(
            "--base-url",
            dest="base_url",
            default=None,
            help="Base URL for OpenAI-compatible APIs.",
        )

        parser.add_argument(
            "--output",
            "-o",
            dest="output",
            default=None,
            help="Save results to JSON file.",
        )

        parser.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            default=False,
            help="Enable debug logging.",
        )

        parser.add_argument(
            "--quiet",
            "-q",
            dest="quiet",
            action="store_true",
            default=False,
            help="Suppress progress output.",
        )

        parser.add_argument(
            "--limit",
            dest="limit",
            type=int,
            default=None,
            help="Maximum number of rows to benchmark.",
        )

        parser.add_argument(
            "--offset",
            dest="offset",
            type=int,
            default=0,
            help="Number of rows to skip.",
        )

        parser.add_argument(
            "--batch-size",
            dest="batch_size",
            type=int,
            default=1,
            help="Number of concurrent runs (default: 1).",
        )

    def run(self, args: argparse.Namespace) -> int:
        return asyncio.run(self._run_async(args))

    def _print_header(self, args: argparse.Namespace) -> None:
        if args.quiet:
            return
        from osmosis_ai.consts import PACKAGE_VERSION

        self.console.print(f"osmosis-bench v{PACKAGE_VERSION}", style="bold")
        if args.base_url:
            self.console.print(f"Endpoint: {args.base_url}")
            self.console.print(f"Model: {args.model}")
        else:
            self.console.print(f"Model: {args.model}")
        self.console.print()

    def _load_eval_fns(
        self, args: argparse.Namespace
    ) -> Tuple[Optional[List["EvalFnWrapper"]], Optional[str]]:
        from osmosis_ai.rollout.eval.bench.eval_fn import EvalFnError, load_eval_fns

        if not args.quiet:
            self.console.print(f"Loading eval functions: {', '.join(args.eval_fns)}")

        try:
            eval_fns = load_eval_fns(args.eval_fns)
        except EvalFnError as e:
            return None, str(e)

        if not args.quiet:
            for fn in eval_fns:
                self.console.print(f"  {fn.name} ({fn._mode} mode)")

        return eval_fns, None

    def _write_output(self, args: argparse.Namespace, result: "BenchResult") -> None:
        if not args.output:
            return

        output_data = {
            "config": {
                "model": args.model,
                "n_runs": args.n_runs,
                "pass_threshold": args.pass_threshold,
                "eval_fns": args.eval_fns,
            },
            "summary": {
                "total_rows": result.total_rows,
                "total_runs": result.total_runs,
                "eval_fns": {
                    name: {
                        "mean": summary.mean,
                        "std": summary.std,
                        "min": summary.min,
                        "max": summary.max,
                        **{f"pass_at_{k}": v for k, v in summary.pass_at_k.items()},
                    }
                    for name, summary in result.eval_summaries.items()
                },
                "total_tokens": result.total_tokens,
                "total_duration_ms": result.total_duration_ms,
            },
            "rows": [
                {
                    "row_index": row.row_index,
                    "runs": [
                        {
                            "run_index": run.run_index,
                            "success": run.success,
                            "scores": run.scores,
                            "duration_ms": run.duration_ms,
                            "tokens": run.tokens,
                            **({"error": run.error} if run.error else {}),
                        }
                        for run in row.runs
                    ],
                }
                for row in result.rows
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

        self._print_header(args)

        agent_loop, error = load_agent(
            module=args.module,
            quiet=args.quiet,
            console=self.console,
        )
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert agent_loop is not None

        rows, error = load_dataset_rows(
            dataset_path=args.dataset,
            limit=args.limit,
            offset=args.offset,
            quiet=args.quiet,
            console=self.console,
            empty_error="No rows to benchmark",
            action_label="benchmarking",
        )
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert rows is not None

        eval_fns, error = self._load_eval_fns(args)
        if error:
            self.console.print_error(f"Error: {error}")
            return 1
        assert eval_fns is not None

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

        completion_params = build_completion_params(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        from osmosis_ai.rollout.eval.bench.runner import BenchRunner

        runner = BenchRunner(
            agent_loop=agent_loop,
            llm_client=llm_client,
            eval_fns=eval_fns,
            debug=args.debug,
        )

        def on_progress(current: int, total: int, result: "BenchRunResult") -> None:
            if args.quiet:
                return

            status_style = "green" if result.success else "red"
            status = "OK" if result.success else "FAILED"
            duration = format_duration(result.duration_ms)

            scores_str = ""
            if result.success and result.scores:
                score_parts = [f"{k}={v:.3f}" for k, v in result.scores.items()]
                scores_str = f" [{', '.join(score_parts)}]"

            error_suffix = ""
            if not result.success and result.error:
                error_text = result.error.replace("\n", " ")
                error_msg = error_text[:47] + "..." if len(error_text) > 50 else error_text
                error_suffix = f" - {error_msg}"

            status_styled = self.console.format_styled(status, status_style)
            self.console.print(
                f"[{current}/{total}] {status_styled} "
                f"({duration}, {result.tokens:,} tokens){scores_str}{error_suffix}"
            )

        if not args.quiet:
            self.console.print()
            n_info = f" x{args.n_runs} runs" if args.n_runs > 1 else ""
            batch_info = f", batch_size={args.batch_size}" if args.batch_size > 1 else ""
            self.console.print(f"Running benchmark ({len(rows)} rows{n_info}{batch_info})...")

        try:
            async with llm_client:
                bench_result = await runner.run_bench(
                    rows=rows,
                    n_runs=args.n_runs,
                    max_turns=args.max_turns,
                    completion_params=completion_params if completion_params else None,
                    pass_threshold=args.pass_threshold,
                    on_progress=on_progress,
                    start_index=args.offset,
                    batch_size=args.batch_size,
                )
        except Exception as e:
            self.console.print_error(f"Error during benchmark: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        if not args.quiet:
            from osmosis_ai.rollout.eval.bench.report import format_bench_report

            format_bench_report(bench_result, self.console)

        self._write_output(args, bench_result)
        return 0

__all__ = ["BenchCommand"]

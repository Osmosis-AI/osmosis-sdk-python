"""CLI command for eval mode.

Run agent loop evaluations against datasets with eval functions and pass@n support.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.eval.common.cli import (
    build_completion_params,
    create_llm_client,
    format_duration,
    load_agent,
    load_dataset_rows,
    load_mcp_agent,
    verify_llm_client,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
    from osmosis_ai.rollout.eval.evaluation.runner import EvalResult, EvalRunResult


logger = logging.getLogger(__name__)


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(self) -> None:
        self.console = Console()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
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
            help="Path to dataset file (.json, .jsonl, or .parquet).",
        )

        parser.add_argument(
            "--model",
            dest="model",
            required=True,
            help=(
                "Model to evaluate. Use with --base-url for trained model "
                "endpoints (e.g., 'my-finetuned-model'), or LiteLLM provider "
                "format for comparison baselines (e.g., 'openai/gpt-5-mini')."
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
            "--baseline-model",
            dest="baseline_model",
            default=None,
            help=(
                "Baseline model for comparison. Runs the same evaluation with "
                "a second model and reports per-model statistics."
            ),
        )

        parser.add_argument(
            "--baseline-base-url",
            dest="baseline_base_url",
            default=None,
            help="Base URL for the baseline model's API endpoint.",
        )

        parser.add_argument(
            "--baseline-api-key",
            dest="baseline_api_key",
            default=None,
            help="API key for the baseline model provider.",
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
            help="Maximum number of rows to evaluate.",
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

    def _validate_args(self, args: argparse.Namespace) -> str | None:
        if args.module and args.mcp:
            return "--module and --mcp are mutually exclusive."
        if not args.module and not args.mcp:
            return "Either --module (-m) or --mcp is required."
        if args.n_runs < 1:
            return "--n must be >= 1."
        if args.batch_size < 1:
            return "--batch-size must be >= 1."
        if args.max_turns < 1:
            return "--max-turns must be >= 1."
        if args.offset < 0:
            return "--offset must be >= 0."
        if args.limit is not None and args.limit < 1:
            return "--limit must be >= 1."
        if args.baseline_base_url and not args.baseline_model:
            return "--baseline-base-url requires --baseline-model."
        if args.baseline_api_key and not args.baseline_model:
            return "--baseline-api-key requires --baseline-model."
        return None

    def _print_header(self, args: argparse.Namespace) -> None:
        if args.quiet:
            return
        from osmosis_ai.consts import PACKAGE_VERSION

        self.console.print(f"osmosis-eval v{PACKAGE_VERSION}", style="bold")
        if args.base_url:
            self.console.print(f"Endpoint: {args.base_url}")
            self.console.print(f"Model: {args.model}")
        else:
            self.console.print(f"Model: {args.model}")
        if args.baseline_model:
            if args.baseline_base_url:
                self.console.print(f"Baseline endpoint: {args.baseline_base_url}")
            self.console.print(f"Baseline model: {args.baseline_model}")
        self.console.print()

    def _load_eval_fns(
        self, args: argparse.Namespace
    ) -> tuple[list[EvalFnWrapper] | None, str | None]:
        from osmosis_ai.rollout.eval.evaluation.eval_fn import (
            EvalFnError,
            load_eval_fns,
        )

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

    def _write_output(self, args: argparse.Namespace, result: EvalResult) -> None:
        if not args.output:
            return

        config: dict[str, Any] = {
            "model": args.model,
            "n_runs": args.n_runs,
            "pass_threshold": args.pass_threshold,
            "eval_fns": args.eval_fns,
        }
        if args.baseline_model:
            config["baseline_model"] = args.baseline_model

        summary: dict[str, Any] = {
            "total_rows": result.total_rows,
            "total_runs": result.total_runs,
            "stopped_early": result.stopped_early,
            **({"stop_reason": result.stop_reason} if result.stop_reason else {}),
            "eval_fns": {
                name: {
                    "mean": s.mean,
                    "std": s.std,
                    "min": s.min,
                    "max": s.max,
                    **{f"pass_at_{k}": v for k, v in s.pass_at_k.items()},
                }
                for name, s in result.eval_summaries.items()
            },
            "total_tokens": result.total_tokens,
            "total_duration_ms": result.total_duration_ms,
        }

        output_data: dict[str, Any] = {
            "config": config,
            "summary": summary,
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
                            **({"model_tag": run.model_tag} if run.model_tag else {}),
                            **({"error": run.error} if run.error else {}),
                        }
                        for run in row.runs
                    ],
                }
                for row in result.rows
            ],
        }

        if result.model_summaries:
            output_data["model_summaries"] = [
                {
                    "model": ms.model,
                    "model_tag": ms.model_tag,
                    "total_runs": ms.total_runs,
                    "total_tokens": ms.total_tokens,
                    "total_duration_ms": ms.total_duration_ms,
                    "eval_fns": {
                        name: {
                            "mean": s.mean,
                            "std": s.std,
                            "min": s.min,
                            "max": s.max,
                            **{f"pass_at_{k}": v for k, v in s.pass_at_k.items()},
                        }
                        for name, s in ms.eval_summaries.items()
                    },
                }
                for ms in result.model_summaries
            ]

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
            empty_error="No rows to evaluate",
            action_label="evaluating",
        )
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1
        assert rows is not None

        eval_fns, error = self._load_eval_fns(args)
        if error:
            self.console.print_error(f"Error: {error}")
            await llm_client.close()
            return 1
        assert eval_fns is not None

        completion_params = build_completion_params(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # Create optional baseline LLM client
        baseline_llm_client = None
        if args.baseline_model:
            baseline_llm_client, error = create_llm_client(
                model=args.baseline_model,
                api_key=args.baseline_api_key,
                base_url=args.baseline_base_url,
                quiet=args.quiet,
                console=self.console,
            )
            if error:
                self.console.print_error(f"Error (baseline): {error}")
                await llm_client.close()
                return 1
            assert baseline_llm_client is not None

            error = await verify_llm_client(
                baseline_llm_client, args.quiet, self.console
            )
            if error:
                self.console.print_error(f"Error (baseline): {error}")
                await baseline_llm_client.close()
                await llm_client.close()
                return 1

        from osmosis_ai.rollout.eval.evaluation.runner import EvalRunner

        runner = EvalRunner(
            agent_loop=agent_loop,
            llm_client=llm_client,
            eval_fns=eval_fns,
            debug=args.debug,
            baseline_llm_client=baseline_llm_client,
        )

        def on_progress(current: int, total: int, result: EvalRunResult) -> None:
            if args.quiet:
                return

            status_style = "green" if result.success else "red"
            status = "OK" if result.success else "FAILED"
            duration = format_duration(result.duration_ms)

            tag_prefix = ""
            if result.model_tag:
                tag_prefix = f"[{result.model_tag}] "

            scores_str = ""
            if result.success and result.scores:
                score_parts = [f"{k}={v:.3f}" for k, v in result.scores.items()]
                scores_str = f" [{', '.join(score_parts)}]"

            error_suffix = ""
            if not result.success and result.error:
                error_text = result.error.replace("\n", " ")
                error_msg = (
                    error_text[:47] + "..." if len(error_text) > 50 else error_text
                )
                error_suffix = f" - {error_msg}"

            status_styled = self.console.format_styled(status, status_style)
            self.console.print(
                f"[{current}/{total}] {tag_prefix}{status_styled} "
                f"({duration}, {result.tokens:,} tokens){scores_str}{error_suffix}"
            )

        if not args.quiet:
            self.console.print()
            n_info = f" x{args.n_runs} runs" if args.n_runs > 1 else ""
            batch_info = (
                f", batch_size={args.batch_size}" if args.batch_size > 1 else ""
            )
            model_info = " x2 models" if args.baseline_model else ""
            self.console.print(
                f"Running evaluation ({len(rows)} rows{n_info}{model_info}{batch_info})..."
            )

        try:
            async with llm_client:
                if baseline_llm_client is not None:
                    async with baseline_llm_client:
                        eval_result = await runner.run_eval(
                            rows=rows,
                            n_runs=args.n_runs,
                            max_turns=args.max_turns,
                            completion_params=completion_params
                            if completion_params
                            else None,
                            pass_threshold=args.pass_threshold,
                            on_progress=on_progress,
                            start_index=args.offset,
                            batch_size=args.batch_size,
                        )
                else:
                    eval_result = await runner.run_eval(
                        rows=rows,
                        n_runs=args.n_runs,
                        max_turns=args.max_turns,
                        completion_params=completion_params
                        if completion_params
                        else None,
                        pass_threshold=args.pass_threshold,
                        on_progress=on_progress,
                        start_index=args.offset,
                        batch_size=args.batch_size,
                    )
        except Exception as e:
            self.console.print_error(f"Error during evaluation: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        if not args.quiet:
            from osmosis_ai.rollout.eval.evaluation.report import format_eval_report

            format_eval_report(eval_result, self.console, model=args.model)

        self._write_output(args, eval_result)
        return 0


__all__ = ["EvalCommand"]

"""CLI command for eval mode.

Run agent loop evaluations against datasets with eval functions and pass@n support.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import time
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
    from osmosis_ai.rollout.eval.evaluation.runner import EvalRunResult


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(self) -> None:
        self.console: Console = Console()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        # Create subparsers for cache management commands
        subparsers = parser.add_subparsers(dest="eval_subcommand")

        cache_parser = subparsers.add_parser("cache", help="Cache management commands")
        cache_subparsers = cache_parser.add_subparsers(
            dest="cache_action", help="Cache actions"
        )

        dir_parser = cache_subparsers.add_parser(
            "dir", help="Print cache root directory path"
        )
        dir_parser.set_defaults(handler=self._run_cache_dir)

        cache_parser.set_defaults(handler=self._run_cache_default)

        # Default handler: run evaluation
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
            required=False,
            default=None,
            help="Path to dataset file (.parquet recommended, .jsonl, or .csv).",
        )

        parser.add_argument(
            "--model",
            dest="model",
            required=False,
            default=None,
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
            required=False,
            default=None,
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

        parser.add_argument(
            "--fresh",
            dest="fresh",
            action="store_true",
            default=False,
            help="Force restart evaluation from scratch, discarding any cached results.",
        )

        parser.add_argument(
            "--log-samples",
            dest="log_samples",
            action="store_true",
            default=False,
            help="Store conversation messages to a JSONL file alongside the cache.",
        )

        parser.add_argument(
            "--output-path",
            dest="output_path",
            default=None,
            help="Directory path for structured output (results JSON and optional samples JSONL).",
        )

        parser.add_argument(
            "--retry-failed",
            dest="retry_failed",
            action="store_true",
            default=False,
            help="Re-execute only failed runs from a previous evaluation. Mutually exclusive with --fresh.",
        )

    def _run_cache_dir(self, args: argparse.Namespace) -> int:
        from osmosis_ai.rollout.eval.evaluation.cache import _get_cache_root

        self.console.print(str(_get_cache_root()))
        return 0

    def _run_cache_default(self, args: argparse.Namespace) -> int:
        self.console.print("Usage: osmosis eval cache <command>")
        self.console.print("")
        self.console.print("Commands:")
        self.console.print("  dir     Print cache root directory path")
        return 0

    def run(self, args: argparse.Namespace) -> int:
        # If a subcommand like "cache" was used, the handler is already set
        # to the subcommand handler, so this method is only called for the
        # default eval run path.

        # For the default eval path, dataset/model/eval_fns are required
        if not getattr(args, "dataset", None):
            self.console.print_error("Error: --dataset (-d) is required.")
            return 1
        if not getattr(args, "model", None):
            self.console.print_error("Error: --model is required.")
            return 1
        if not getattr(args, "eval_fns", None):
            self.console.print_error("Error: --eval-fn is required.")
            return 1

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
        if getattr(args, "fresh", False) and getattr(args, "retry_failed", False):
            return "--fresh and --retry-failed are mutually exclusive."
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

    def _write_orchestrator_output(
        self,
        output_path: str,
        model: str,
        dataset_path: str,
        cache_data: dict,
        samples_path: Path | None,
        task_id: str,
    ) -> Path:
        """Write structured orchestrator output to the specified directory.

        Creates {output_path}/{model_sanitized}/{dataset_sanitized}/ directory
        structure and writes results JSON and optional samples JSONL.

        Returns the path to the written results JSON file.
        """
        from osmosis_ai.rollout.eval.evaluation.cache import _sanitize_path_part

        model_dir = _sanitize_path_part(model)
        dataset_dir = _sanitize_path_part(Path(dataset_path).stem)

        out_dir = Path(output_path) / model_dir / dataset_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        unix_ts = int(time.time())

        results_filename = f"results_{unix_ts}_{task_id}.json"
        results_path = out_dir / results_filename

        output_data: dict[str, Any] = {
            "task_id": cache_data.get("task_id"),
            "config_hash": cache_data.get("config_hash"),
            "config": cache_data.get("config"),
            "summary": cache_data.get("summary"),
            "runs": cache_data.get("runs", []),
            "created_at": cache_data.get("created_at"),
            "completed_at": cache_data.get("updated_at"),
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        if samples_path is not None and samples_path.exists():
            samples_filename = f"samples_{unix_ts}_{task_id}.jsonl"
            dest_samples = out_dir / samples_filename
            shutil.copy2(samples_path, dest_samples)

        return results_path

    def _print_orchestrator_summary(
        self,
        summary: dict | None,
        total_completed: int,
        total_expected: int,
    ) -> None:
        """Print a summary from an OrchestratorResult."""
        if summary is None:
            self.console.print(f"\nCompleted {total_completed}/{total_expected} runs.")
            return

        self.console.print()
        self.console.print("Evaluation Results:", style="bold")

        total_runs = summary.get("total_runs", total_completed)
        total_tokens = summary.get("total_tokens", 0)
        total_duration_ms = summary.get("total_duration_ms", 0.0)

        self.console.print(f"  Total runs: {total_runs}")
        self.console.print(f"  Duration: {format_duration(total_duration_ms)}")
        self.console.print(f"  Total tokens: {total_tokens:,}")

        eval_fns = summary.get("eval_fns", {})
        if eval_fns:
            self.console.print()
            for fn_name, stats in eval_fns.items():
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                s_min = stats.get("min", 0.0)
                s_max = stats.get("max", 0.0)
                self.console.print(
                    f"  {fn_name}: mean={mean:.3f} std={std:.3f} "
                    f"min={s_min:.3f} max={s_max:.3f}"
                )
                # Print pass@k if present
                for key, val in stats.items():
                    if key.startswith("pass_at_"):
                        k_val = key.replace("pass_at_", "")
                        self.console.print(f"    pass@{k_val}: {val * 100:.1f}%")

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

        # Compute fingerprints and use orchestrator
        from osmosis_ai.rollout.eval.evaluation.cache import (
            CacheConfig,
            JsonFileCacheBackend,
            _get_cache_root,
            compute_dataset_fingerprint,
            compute_eval_fns_fingerprint,
            compute_module_fingerprint,
            compute_task_id,
        )

        module_spec = args.module or args.mcp
        dataset_fingerprint = compute_dataset_fingerprint(args.dataset)
        module_fingerprint = compute_module_fingerprint(module_spec)
        eval_fns_fingerprint = compute_eval_fns_fingerprint(args.eval_fns)

        task_id, config_hash = compute_task_id(
            model=args.model,
            base_url=args.base_url,
            baseline_model=args.baseline_model,
            baseline_base_url=args.baseline_base_url,
            module=module_spec,
            dataset=args.dataset,
            eval_fns=args.eval_fns,
            n_runs=args.n_runs,
            max_turns=args.max_turns,
            pass_threshold=args.pass_threshold,
            offset=args.offset,
            limit=args.limit,
            completion_params=completion_params if completion_params else None,
            module_fingerprint=module_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            eval_fns_fingerprint=eval_fns_fingerprint,
        )

        config_dict: dict[str, object] = {
            "model": args.model,
            "base_url": args.base_url,
            "baseline_model": args.baseline_model,
            "baseline_base_url": args.baseline_base_url,
            "module": module_spec,
            "dataset": args.dataset,
            "eval_fns": sorted(args.eval_fns),
            "n_runs": args.n_runs,
            "max_turns": args.max_turns,
            "pass_threshold": args.pass_threshold,
            "offset": args.offset,
            "limit": args.limit,
            "completion_params": completion_params,
        }
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        cache_config = CacheConfig(
            task_id=task_id,
            config_hash=config_hash,
            model=args.model,
            dataset_path=args.dataset,
            config=config_dict,
            total_rows=len(rows),
        )

        cache_backend = JsonFileCacheBackend()

        # Validate --output-path is not the cache root
        output_path = getattr(args, "output_path", None)
        if output_path is not None:
            output_path_resolved = Path(output_path).resolve()
            cache_root_resolved = _get_cache_root().resolve()
            if output_path_resolved == cache_root_resolved:
                self.console.print_error(
                    "Error: --output-path cannot be the same as the cache root directory."
                )
                await llm_client.close()
                if baseline_llm_client is not None:
                    await baseline_llm_client.close()
                return 1

        model_tags: list[str | None] = (
            ["primary", "baseline"] if args.baseline_model else [None]
        )

        fresh = getattr(args, "fresh", False)
        log_samples = getattr(args, "log_samples", False)

        from osmosis_ai.rollout.eval.evaluation.orchestrator import EvalOrchestrator

        orchestrator = EvalOrchestrator(
            runner=runner,
            cache_backend=cache_backend,
            cache_config=cache_config,
            rows=rows,
            n_runs=args.n_runs,
            max_turns=args.max_turns,
            completion_params=completion_params if completion_params else None,
            pass_threshold=args.pass_threshold,
            batch_size=args.batch_size,
            log_samples=log_samples,
            fresh=fresh,
            dataset_path=Path(args.dataset),
            dataset_fingerprint=dataset_fingerprint,
            start_index=args.offset,
            model_tags=model_tags,
            on_progress=on_progress,
        )

        try:
            async with llm_client:
                if baseline_llm_client is not None:
                    async with baseline_llm_client:
                        orch_result = await orchestrator.run()
                else:
                    orch_result = await orchestrator.run()
        except Exception as e:
            self.console.print_error(f"Error during evaluation: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        # Handle OrchestratorResult status
        if orch_result.status == "already_completed":
            if not args.quiet:
                self.console.print(
                    "\nEvaluation already completed (cached).",
                    style="bold",
                )
                self._print_orchestrator_summary(
                    orch_result.summary,
                    orch_result.total_completed,
                    orch_result.total_expected,
                )
                self.console.print(f"\nCache: {orch_result.cache_path}")
            if output_path:
                results_path = self._write_orchestrator_output(
                    output_path=output_path,
                    model=args.model,
                    dataset_path=args.dataset,
                    cache_data=orch_result.cache_data,
                    samples_path=orch_result.samples_path,
                    task_id=task_id,
                )
                if not args.quiet:
                    self.console.print(f"Output written to: {results_path}")
            # Also write legacy output if --output/-o is set
            self._write_legacy_output_from_cache(args, orch_result.cache_data)
            return 0

        if orch_result.status == "interrupted":
            if not args.quiet:
                self.console.print(
                    f"\nEvaluation interrupted. "
                    f"Progress: {orch_result.total_completed}/{orch_result.total_expected} runs.",
                    style="yellow",
                )
                self.console.print(f"Cache: {orch_result.cache_path}")
                self.console.print(
                    "Re-run the same command to resume from where you left off."
                )
            return 130

        if orch_result.status == "dataset_modified":
            self.console.print_error(
                f"Error: Dataset was modified during evaluation"
                f"{(' (' + orch_result.stop_reason + ')') if orch_result.stop_reason else ''}. "
                f"Results may be inconsistent. Use --fresh to restart."
            )
            return 1

        if orch_result.status == "systemic_error":
            if not args.quiet:
                self.console.print(
                    f"\nEvaluation stopped due to systemic error: "
                    f"{orch_result.stop_reason}",
                    style="red",
                )
                self._print_orchestrator_summary(
                    orch_result.summary,
                    orch_result.total_completed,
                    orch_result.total_expected,
                )
                self.console.print(
                    f"\nPartial results cached at: {orch_result.cache_path}"
                )
            return 1

        # status == "completed"
        if not args.quiet:
            self._print_orchestrator_summary(
                orch_result.summary,
                orch_result.total_completed,
                orch_result.total_expected,
            )
            self.console.print(f"\nCache: {orch_result.cache_path}")

        if output_path:
            results_path = self._write_orchestrator_output(
                output_path=output_path,
                model=args.model,
                dataset_path=args.dataset,
                cache_data=orch_result.cache_data,
                samples_path=orch_result.samples_path,
                task_id=task_id,
            )
            if not args.quiet:
                self.console.print(f"Output written to: {results_path}")

        # Also write legacy output if --output/-o is set
        self._write_legacy_output_from_cache(args, orch_result.cache_data)

        return 0

    def _write_legacy_output_from_cache(
        self, args: argparse.Namespace, cache_data: dict
    ) -> None:
        """Write legacy --output/-o JSON from cache data for backward compatibility.

        Reconstructs the nested ``rows`` structure expected by the original format:
        ``{"config": ..., "summary": ..., "rows": [{"row_index": N, "runs": [...]}]}``
        """
        if not getattr(args, "output", None):
            return

        config: dict[str, Any] = {
            "model": args.model,
            "n_runs": args.n_runs,
            "pass_threshold": args.pass_threshold,
            "eval_fns": args.eval_fns,
        }
        if args.baseline_model:
            config["baseline_model"] = args.baseline_model

        summary_data = cache_data.get("summary", {}) or {}
        flat_runs = cache_data.get("runs", [])

        # Reconstruct nested rows structure from flat runs list
        rows_map: dict[int, list[dict[str, Any]]] = {}
        for run in flat_runs:
            row_idx = run.get("row_index", 0)
            if row_idx not in rows_map:
                rows_map[row_idx] = []
            rows_map[row_idx].append(
                {
                    "run_index": run.get("run_index", 0),
                    "success": run.get("success", False),
                    "scores": run.get("scores", {}),
                    "duration_ms": run.get("duration_ms", 0.0),
                    "tokens": run.get("tokens", 0),
                    **({"model_tag": run["model_tag"]} if run.get("model_tag") else {}),
                    **({"error": run["error"]} if run.get("error") else {}),
                }
            )

        rows_list = [
            {"row_index": row_idx, "runs": runs}
            for row_idx, runs in sorted(rows_map.items())
        ]

        output_data: dict[str, Any] = {
            "config": config,
            "summary": summary_data,
            "rows": rows_list,
        }

        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        if not args.quiet:
            self.console.print(f"\nResults written to: {args.output}")


__all__ = ["EvalCommand"]

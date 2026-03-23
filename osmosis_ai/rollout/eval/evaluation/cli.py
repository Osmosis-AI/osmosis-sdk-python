"""CLI command for eval mode.

Run agent loop evaluations against datasets with eval functions and pass@n support.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console

if TYPE_CHECKING:
    from osmosis_ai.rollout.eval.evaluation.cache import BuildSummaryResult
    from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
    from osmosis_ai.rollout.eval.evaluation.runner import EvalRunResult


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(self) -> None:
        self.console: Console = Console()

    def _run_cache_dir(self) -> int:
        from osmosis_ai.rollout.eval.evaluation.cache import JsonFileCacheBackend

        backend = JsonFileCacheBackend()
        self.console.print(str(backend.cache_root))
        return 0

    @staticmethod
    def _filter_caches(
        entries: list[dict],
        model: str | None,
        dataset: str | None,
        status: str | None,
    ) -> list[dict]:
        """Filter cache entries by model/dataset/status."""
        result = entries
        if model:
            model_lower = model.lower()
            result = [
                e
                for e in result
                if model_lower in e.get("config", {}).get("model", "").lower()
            ]
        if dataset:
            dataset_lower = dataset.lower()
            result = [
                e
                for e in result
                if dataset_lower in e.get("config", {}).get("dataset", "").lower()
            ]
        if status:
            result = [e for e in result if e.get("status") == status]
        return result

    def _run_cache_ls(
        self,
        *,
        cache_model: str | None = None,
        cache_dataset: str | None = None,
        cache_status: str | None = None,
    ) -> int:
        from osmosis_ai.rollout.eval.evaluation.cache import JsonFileCacheBackend

        backend = JsonFileCacheBackend()
        entries = backend.list_caches()
        entries = self._filter_caches(
            entries,
            model=cache_model,
            dataset=cache_dataset,
            status=cache_status,
        )

        # Sort by created_at descending (newest first)
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)

        if not entries:
            self.console.print("No cached evaluations found.")
            return 0

        if self.console.is_tty:
            # Rich table for interactive terminals
            from rich.table import Table as RichTable

            table = RichTable(show_header=True, header_style="bold")
            table.add_column("TASK ID")
            table.add_column("MODEL")
            table.add_column("DATASET")
            table.add_column("STATUS")
            table.add_column("RUNS", justify="right")
            table.add_column("CREATED")
            for e in entries:
                config = e.get("config", {})
                created = e.get("created_at", "")
                if created and len(created) >= 16:
                    created = created[:16].replace("T", " ")
                table.add_row(
                    e.get("task_id", ""),
                    config.get("model", ""),
                    config.get("dataset", ""),
                    e.get("status", ""),
                    str(e.get("runs_count", 0)),
                    created,
                )
            self.console.rich.print(table)
        else:
            # Tab-separated output for piping
            for e in entries:
                config = e.get("config", {})
                created = e.get("created_at", "")
                if created and len(created) >= 16:
                    created = created[:16].replace("T", " ")
                line = "\t".join(
                    [
                        e.get("task_id", ""),
                        config.get("model", ""),
                        config.get("dataset", ""),
                        e.get("status", ""),
                        str(e.get("runs_count", 0)),
                        created,
                    ]
                )
                self.console.print(line, markup=False, highlight=False)

        return 0

    def _run_cache_rm(
        self,
        *,
        task_id: str | None = None,
        rm_all: bool = False,
        cache_model: str | None = None,
        cache_dataset: str | None = None,
        cache_status: str | None = None,
        yes: bool = False,
    ) -> int:
        from osmosis_ai.rollout.eval.evaluation.cache import JsonFileCacheBackend

        has_filter = any([cache_model, cache_dataset, cache_status])

        if not task_id and not rm_all and not has_filter:
            self.console.print_error(
                "Error: Provide a task_id, --all, or at least one filter "
                "(--model, --dataset, --status)."
            )
            return 1

        backend = JsonFileCacheBackend()
        all_entries = backend.list_caches()

        # Select targets
        if task_id:
            targets = [e for e in all_entries if e.get("task_id") == task_id]
        elif rm_all:
            targets = all_entries
        else:
            targets = self._filter_caches(
                all_entries, cache_model, cache_dataset, cache_status
            )

        if not targets:
            self.console.print("No matching cached evaluations found.")
            return 1

        # Single task_id deletion: no confirmation needed
        is_batch = not task_id
        if is_batch and not yes:
            self.console.print(f"Will delete {len(targets)} cached evaluation(s):")
            for e in targets:
                config = e.get("config", {})
                self.console.print(
                    f"  {e.get('task_id', '')}"
                    f"  {config.get('model', '')}"
                    f"  {config.get('dataset', '')}"
                    f"  ({e.get('status', '')})"
                )
            try:
                answer = self.console.input(
                    f"Delete {len(targets)} cached evaluation(s)? [y/N] "
                )
            except (EOFError, KeyboardInterrupt):
                self.console.print("\nAborted.")
                return 130
            if answer.strip().lower() not in ("y", "yes"):
                self.console.print("Aborted.")
                return 0

        deleted = 0
        for entry in targets:
            cache_path = Path(entry["path"])
            backend.delete_cache(cache_path)
            # Clean up empty parent directories
            parent = cache_path.parent
            try:
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    grandparent = parent.parent
                    if (
                        grandparent.is_dir()
                        and grandparent != backend.cache_root
                        and not any(grandparent.iterdir())
                    ):
                        grandparent.rmdir()
            except OSError:
                pass  # Best-effort cleanup of empty directories
            deleted += 1

        self.console.print(f"Deleted {deleted} cached evaluation(s).")
        return 0

    def run(self, **kwargs: Any) -> int:
        args = SimpleNamespace(**kwargs)
        return asyncio.run(self._run_async(args))

    def _validate_args(self, args: Any) -> str | None:
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

    def _print_header(self, args: Any) -> None:
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
        self, args: Any
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
        """Write structured orchestrator output to the specified directory."""
        from osmosis_ai.rollout.eval.evaluation.cache import (
            atomic_write_json,
            sanitize_path_part,
        )

        model_dir = sanitize_path_part(model)
        dataset_dir = sanitize_path_part(Path(dataset_path).stem)

        out_dir = Path(output_path) / model_dir / dataset_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        unix_ts = int(time.time())

        results_filename = f"results_{unix_ts}_{task_id}.json"
        results_path = out_dir / results_filename

        output_data = dict(cache_data)  # Copy all fields from cache
        output_data["status"] = "completed"  # Always completed for output

        atomic_write_json(results_path, output_data)

        if samples_path is not None and samples_path.exists():
            samples_filename = f"samples_{unix_ts}_{task_id}.jsonl"
            dest_samples = out_dir / samples_filename
            shutil.copy2(samples_path, dest_samples)

        return results_path

    def _print_orchestrator_summary(
        self,
        summary: BuildSummaryResult | None,
        total_completed: int,
        total_expected: int,
    ) -> None:
        """Print a summary from an OrchestratorResult."""
        from osmosis_ai.rollout.eval.common.cli import format_duration

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
                median = stats.get("median", 0.0)
                std = stats.get("std", 0.0)
                self.console.print(
                    f"  {fn_name}: mean={mean:.3f} median={median:.3f} std={std:.3f}"
                )
                # Print pass@k if present
                for k_val, val in sorted(stats.get("pass_at_k", {}).items()):
                    self.console.print(f"    pass@{k_val}: {float(val) * 100:.1f}%")

    def _print_resume_hints(
        self,
        cache_backend: Any,
        task_id: str,
        model: str,
        dataset_path: str,
        total_expected: int,
        log_samples: bool,
        module_spec: str,
        is_mcp: bool,
    ) -> None:
        """Print informational messages when resuming from cached progress."""
        from osmosis_ai.rollout.eval.evaluation.cache import sanitize_path_part

        # Peek at existing cache file to check for prior completed runs
        model_dir = sanitize_path_part(model)
        dataset_dir = sanitize_path_part(Path(dataset_path).stem)
        cache_dir = cache_backend.cache_root / model_dir / dataset_dir

        if not cache_dir.exists():
            return

        pattern = f"*_{task_id}.json"
        matches = sorted(cache_dir.glob(pattern), reverse=True)
        if not matches:
            return

        try:
            cache_data = json.loads(matches[0].read_text())
        except (json.JSONDecodeError, OSError):
            return

        completed_runs = cache_data.get("runs", [])
        if not completed_runs:
            return

        status = cache_data.get("status", "in_progress")
        if status == "completed":
            return

        completed = len(completed_runs)

        # Determine fingerprint scope description
        if is_mcp:
            scope = f"MCP directory {module_spec}"
        else:
            module_name = module_spec.partition(":")[0]
            try:
                mod = importlib.import_module(module_name)
                source_file = getattr(mod, "__file__", None)
                if source_file and Path(source_file).name == "__init__.py":
                    scope = f"package {module_name}/"
                else:
                    scope = f"file {module_name}.py"
            except Exception:
                scope = f"file {module_name}.py"

        self.console.print(
            f"Resuming eval ({completed}/{total_expected} runs completed)"
        )
        self.console.print(
            f"Note: Module fingerprint covers {scope}. External dependency changes\n"
            f"      are not detected. Use --fresh if you changed external imports."
        )

        # Check --log-samples with missing prior samples
        if log_samples:
            samples_path = matches[0].with_suffix(".jsonl")
            if not samples_path.exists() or samples_path.stat().st_size == 0:
                self.console.print(
                    "Note: Only new runs will be logged. "
                    "Use --fresh to re-run all with full logging."
                )

    async def _run_async(self, args: Any) -> int:
        from osmosis_ai.rollout.eval.common.cli import (
            build_completion_params,
            create_llm_client,
            format_duration,
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

        module_spec = f"mcp:{Path(args.mcp).resolve()!s}" if args.mcp else args.module
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
            "dataset_fingerprint": dataset_fingerprint,
            "module_fingerprint": module_fingerprint,
            "eval_fns_fingerprint": eval_fns_fingerprint,
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

        # Print resume hints if resuming from cached progress
        if not args.quiet and not fresh:
            total_expected = len(rows) * args.n_runs * len(model_tags)
            self._print_resume_hints(
                cache_backend=cache_backend,
                task_id=task_id,
                model=args.model,
                dataset_path=args.dataset,
                total_expected=total_expected,
                log_samples=log_samples,
                module_spec=module_spec,
                is_mcp=bool(args.mcp),
            )

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
            retry_failed=getattr(args, "retry_failed", False),
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
                if orch_result.dataset_fingerprint_warning:
                    self.console.print(orch_result.dataset_fingerprint_warning)
                self._print_orchestrator_summary(
                    orch_result.summary,
                    orch_result.total_completed,
                    orch_result.total_expected,
                )
                self.console.print(f"\nCache: {orch_result.cache_path}")
                self.console.print("Use --fresh to re-run from scratch.")
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
            if orch_result.samples_path:
                self.console.print(f"Samples: {orch_result.samples_path}")
            if not args.log_samples:
                self.console.print(
                    "Tip: Use --log-samples to save full conversation logs for debugging."
                )

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

        return 0


__all__ = ["EvalCommand"]

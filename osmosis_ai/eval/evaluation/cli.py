"""CLI command for eval mode.

Run agent loop evaluations against datasets using TOML config.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    MessageResult,
    OperationResult,
    OutputFormat,
    get_output_context,
    serialize_eval_cache_entry,
)

if TYPE_CHECKING:
    from osmosis_ai.eval.evaluation.cache import BuildSummaryResult


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(self) -> None:
        self.console: Console = Console()

    @staticmethod
    def _structured_output() -> bool:
        return get_output_context().format is not OutputFormat.rich

    def _fail(self, message: str, *, code: str = "VALIDATION") -> int:
        if self._structured_output():
            raise CLIError(message.removeprefix("Error: "), code=code)
        self.console.print_error(message)
        return 1

    @staticmethod
    def _stderr_line(message: str) -> None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    def _run_cache_dir(self) -> int | DetailResult:
        from osmosis_ai.eval.evaluation.cache import JsonFileCacheBackend

        backend = JsonFileCacheBackend()
        if self._structured_output():
            path = str(backend.cache_root)
            return DetailResult(
                title="Eval Cache",
                data={"cache_root": path},
                fields=[DetailField(label="Cache root", value=path)],
            )
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
                if model_lower
                in (
                    e.get("config", {}).get("llm_model", "")
                    or e.get("config", {}).get("model", "")
                ).lower()
            ]
        if dataset:
            dataset_lower = dataset.lower()
            result = [
                e
                for e in result
                if dataset_lower
                in (
                    e.get("config", {}).get("eval_dataset", "")
                    or e.get("config", {}).get("dataset", "")
                ).lower()
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
    ) -> int | ListResult:
        from osmosis_ai.eval.evaluation.cache import JsonFileCacheBackend

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

        if self._structured_output():
            return ListResult(
                title="Eval Caches",
                items=[serialize_eval_cache_entry(entry) for entry in entries],
                total_count=len(entries),
                has_more=False,
                next_offset=None,
                columns=[
                    ListColumn(key="task_id", label="Task ID", no_wrap=True),
                    ListColumn(key="model", label="Model"),
                    ListColumn(key="dataset", label="Dataset"),
                    ListColumn(key="status", label="Status"),
                    ListColumn(key="runs_count", label="Runs", align="right"),
                    ListColumn(key="created_at", label="Created"),
                ],
            )

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
    ) -> int | OperationResult | MessageResult:
        from osmosis_ai.eval.evaluation.cache import JsonFileCacheBackend

        has_filter = any([cache_model, cache_dataset, cache_status])

        if not task_id and not rm_all and not has_filter:
            return self._fail(
                "Error: Provide a task_id, --all, or at least one filter "
                "(--model, --dataset, --status).",
            )

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
            if self._structured_output():
                raise CLIError(
                    "No matching cached evaluations found.",
                    code="NOT_FOUND",
                )
            self.console.print("No matching cached evaluations found.")
            return 1

        # Single task_id deletion: no confirmation needed
        is_batch = not task_id
        if is_batch and not yes:
            if self._structured_output():
                raise CLIError(
                    "Use --yes to confirm cache deletion in non-interactive mode.",
                    code="INTERACTIVE_REQUIRED",
                )
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
        if self._structured_output():
            return OperationResult(
                operation="eval.cache.rm",
                status="success",
                resource={"deleted_count": deleted},
                message=f"Deleted {deleted} cached evaluation(s).",
            )
        return 0

    def run(self, **kwargs: Any) -> int | DetailResult:
        args = SimpleNamespace(**kwargs)
        return asyncio.run(self._run_async(args))

    @staticmethod
    def _resolve_api_key(config: Any) -> str | None:
        import os

        if not config.llm_api_key_env:
            return None
        value = os.environ.get(config.llm_api_key_env)
        if not value:
            from osmosis_ai.cli.errors import CLIError

            raise CLIError(
                f"Environment variable '{config.llm_api_key_env}' "
                f"(from [llm].api_key_env) is not set or empty."
            )
        return value

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
        from osmosis_ai.eval.evaluation.cache import (
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
        from osmosis_ai.eval.common.cli import format_duration

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

        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)

        reward_stats = summary.get("reward_stats")
        if reward_stats:
            self.console.print()
            mean = reward_stats.get("mean", 0.0)
            median = reward_stats.get("median", 0.0)
            std = reward_stats.get("std", 0.0)
            self.console.print(
                f"  reward: mean={mean:.3f} median={median:.3f} std={std:.3f}"
            )
            self.console.print(
                f"  range: [{reward_stats.get('min', 0.0):.3f}, {reward_stats.get('max', 0.0):.3f}]"
            )
            self.console.print(f"  Passed: {passed}/{total_runs}")
            for k_val, val in sorted(reward_stats.get("pass_at_k", {}).items()):
                self.console.print(f"    pass@{k_val}: {float(val) * 100:.1f}%")
        else:
            self.console.print(f"  Passed: {passed}/{total_runs}")
            if failed:
                self.console.print(f"  Failed: {failed}")

    @staticmethod
    def _partial_failures(cache_data: dict[str, Any]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for run in cache_data.get("runs", []):
            if run.get("success", True):
                continue
            failures.append(
                {
                    "row_index": run.get("row_index"),
                    "run_index": run.get("run_index"),
                    "model_tag": run.get("model_tag"),
                    "error": run.get("error"),
                }
            )
        return failures

    def _build_eval_run_result(
        self,
        *,
        status: str,
        summary: BuildSummaryResult | None,
        total_completed: int,
        total_expected: int,
        cache_path: Path,
        samples_path: Path | None,
        output_path: Path | None,
        cache_data: dict[str, Any],
        dataset_fingerprint_warning: str | None = None,
        exit_code: int = 0,
    ) -> DetailResult:
        summary_data: dict[str, Any] = dict(summary or {})
        partial_failures = self._partial_failures(cache_data)
        failed_runs = int(summary_data.get("failed", len(partial_failures)))
        total_runs = int(summary_data.get("total_runs", total_expected))
        data = {
            "status": status,
            "total_runs": total_runs,
            "completed_runs": total_completed,
            "expected_runs": total_expected,
            "failed_runs": failed_runs,
            "cache_path": str(cache_path),
            "samples_path": str(samples_path) if samples_path else None,
            "output_path": str(output_path) if output_path else None,
            "partial_failures": partial_failures,
            "summary": summary_data,
        }
        if dataset_fingerprint_warning:
            data["dataset_fingerprint_warning"] = dataset_fingerprint_warning

        return DetailResult(
            title="Eval Run",
            data=data,
            fields=[
                DetailField(label="Status", value=status),
                DetailField(label="Total runs", value=str(total_runs)),
                DetailField(label="Completed runs", value=str(total_completed)),
                DetailField(label="Failed runs", value=str(failed_runs)),
                DetailField(label="Cache", value=str(cache_path)),
                DetailField(
                    label="Output",
                    value=str(output_path) if output_path else "",
                ),
            ],
            exit_code=exit_code,
        )

    async def _run_async(self, args: Any) -> int | DetailResult:
        output = get_output_context()
        structured_output = output.format is not OutputFormat.rich

        # Reject conflicting CLI flags before config load so invalid fixtures cannot mask this error.
        if args.fresh and args.retry_failed:
            return self._fail(
                "Error: --fresh and --retry-failed are mutually exclusive."
            )

        from osmosis_ai.eval.common.cli import (
            _resolve_grader,
            format_duration,
            load_dataset_rows,
            load_workflow,
            truncate_error,
        )
        from osmosis_ai.eval.config import load_eval_config
        from osmosis_ai.platform.cli.workspace_contract import (
            ensure_workspace_config_path,
            resolve_workspace_root,
            validate_workspace_contract,
        )

        # 1. Load TOML config
        config_path = Path(args.config_path)
        try:
            config = load_eval_config(config_path)
        except CLIError as e:
            return self._fail(f"Error: {e}")

        try:
            workspace_root = resolve_workspace_root(config_path)
            validate_workspace_contract(workspace_root)
            ensure_workspace_config_path(
                config_path,
                workspace_root,
                config_dir="configs/eval",
                command_label="`osmosis eval run`",
            )
        except CLIError as e:
            return self._fail(f"Error: {e}")

        # Apply CLI overrides (CLI flags take precedence over TOML).
        # Optional values: CLI wins when not None.
        # Booleans: CLI can enable (or), TOML sets default.
        limit = args.limit if args.limit is not None else config.eval_limit
        offset = args.offset if args.offset is not None else config.eval_offset
        if offset is not None and offset < 0:
            return self._fail("Error: --offset must be >= 0")
        fresh = args.fresh or config.eval_fresh
        retry_failed = args.retry_failed or config.eval_retry_failed
        batch_size = (
            args.batch_size_override
            if args.batch_size_override is not None
            else config.runs_batch_size
        )
        log_samples = args.log_samples or config.output_log_samples
        output_path = (
            args.output_path if args.output_path is not None else config.output_path
        )
        quiet = args.quiet or config.output_quiet
        debug = args.debug or config.output_debug

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            # Suppress verbose tracebacks from backend loggers;
            # errors are already surfaced via the progress callback.
            logging.getLogger("osmosis_ai.rollout.backend").setLevel(logging.CRITICAL)

        if fresh and retry_failed:
            return self._fail(
                "Error: --fresh and --retry-failed are mutually exclusive."
            )

        if not quiet:
            from osmosis_ai.consts import PACKAGE_VERSION

            self.console.print(f"osmosis-eval v{PACKAGE_VERSION}", style="bold")
            self.console.print(f"Config: {config_path}")
            self.console.print(f"Model: {config.llm_model}")
            if config.baseline_model:
                self.console.print(f"Baseline: {config.baseline_model}")
            self.console.print()

        # 2. Resolve API key
        try:
            api_key = self._resolve_api_key(config)
        except CLIError as e:
            return self._fail(f"Error: {e}")

        # 3. Load dataset
        rows, error = load_dataset_rows(
            dataset_path=config.eval_dataset,
            limit=limit,
            offset=offset,
            quiet=quiet,
            console=self.console,
            empty_error="No rows to evaluate",
            action_label="evaluating",
        )
        if error:
            return self._fail(f"Error: {error}")
        assert rows is not None

        # 4. Load workflow
        workflow_cls, workflow_config, entrypoint_module, error = load_workflow(
            rollout=config.eval_rollout,
            entrypoint=config.eval_entrypoint,
            quiet=quiet,
            console=self.console,
            workspace_root=workspace_root,
        )
        if error:
            return self._fail(f"Error: {error}")
        assert workflow_cls is not None
        assert entrypoint_module is not None

        # 5. Resolve grader from [grader] override or auto-discover from entrypoint
        try:
            grader_cls, grader_config = _resolve_grader(
                entrypoint_module,
                explicit_grader=config.grader_module,
                explicit_config=config.grader_config,
            )
        except (CLIError, ImportError, TypeError, ValueError) as e:
            return self._fail(f"Error: {e}")

        if grader_cls is None:
            return self._fail(
                "No Grader was found in the entrypoint module. "
                "`osmosis eval run` requires a concrete Grader (and typically a "
                "GraderConfig) alongside the workflow. Configure `[grader].module` "
                "if the grader lives outside the entrypoint."
            )

        if not quiet:
            self.console.print(f"  Grader: {grader_cls.__name__}")

        # 6. Create proxy and start
        from osmosis_ai.eval.llm_proxy import LiteLLMProxy

        trace_dir = None
        if debug:
            trace_dir = str(Path("./eval_traces") / "debug")

        proxy = LiteLLMProxy(
            model=config.llm_model,
            api_key=api_key,
            base_url=config.llm_base_url,
            trace_dir=trace_dir,
        )

        # 7. Preflight check
        try:
            await proxy.preflight_check()
        except CLIError as e:
            return self._fail(f"Error: {e}")

        # 8. Start proxy
        await proxy.start()

        # 9. Construct backend + driver
        from osmosis_ai.rollout.backend.local.backend import LocalBackend
        from osmosis_ai.rollout.driver import InProcessDriver, RolloutDriver

        backend = LocalBackend(
            workflow=workflow_cls,
            workflow_config=workflow_config,
            grader=grader_cls,
            grader_config=grader_config,
        )
        driver = InProcessDriver(backend=backend, proxy=proxy)

        # 10. Build drivers list
        drivers: list[tuple[str | None, RolloutDriver]] = []
        baseline_proxy = None
        if config.baseline_model:
            drivers.append(("primary", driver))

            # Resolve baseline API key
            baseline_api_key = None
            if config.baseline_api_key_env:
                import os

                baseline_api_key = os.environ.get(config.baseline_api_key_env)
                if not baseline_api_key:
                    await proxy.stop()
                    return self._fail(
                        f"Error: Environment variable '{config.baseline_api_key_env}' is not set."
                    )

            baseline_proxy = LiteLLMProxy(
                model=config.baseline_model,
                api_key=baseline_api_key,
                base_url=config.baseline_base_url,
                trace_dir=trace_dir,
            )
            try:
                await baseline_proxy.preflight_check()
            except CLIError as e:
                await proxy.stop()
                return self._fail(f"Error (baseline): {e}")

            await baseline_proxy.start()
            baseline_driver = InProcessDriver(backend=backend, proxy=baseline_proxy)
            drivers.append(("baseline", baseline_driver))
        else:
            drivers.append((None, driver))

        # 11. Compute cache config
        from osmosis_ai.eval.evaluation.cache import (
            CacheConfig,
            JsonFileCacheBackend,
            _get_cache_root,
            compute_dataset_fingerprint,
            compute_module_fingerprint,
            compute_task_id,
        )

        dataset_fingerprint = compute_dataset_fingerprint(config.eval_dataset)

        module_fingerprint = compute_module_fingerprint(entrypoint_module) or ""

        grader_fingerprint = compute_module_fingerprint(grader_cls.__module__)

        # Merge CLI overrides into config for cache identity.
        # Exclude non-semantic fields (presentation/runtime flags) so that
        # changing --quiet, --debug, --batch-size, etc. doesn't invalidate
        # the cache and break resume.
        _non_semantic = {
            "eval_fresh",
            "eval_retry_failed",
            "llm_api_key_env",
            "runs_batch_size",
            "output_log_samples",
            "output_path",
            "output_quiet",
            "output_debug",
            "baseline_api_key_env",
        }
        config_for_hash = {
            k: v for k, v in config.model_dump().items() if k not in _non_semantic
        }
        config_for_hash["offset"] = offset
        config_for_hash["limit"] = limit

        task_id, config_hash = compute_task_id(
            config=config_for_hash,
            workflow_fingerprint=module_fingerprint,
            grader_fingerprint=grader_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
        )

        config_dict = {
            **config_for_hash,
            "dataset_fingerprint": dataset_fingerprint,
            "module_fingerprint": module_fingerprint,
            "grader_fingerprint": grader_fingerprint,
        }

        cache_config = CacheConfig(
            task_id=task_id,
            config_hash=config_hash,
            model=config.llm_model,
            dataset_path=config.eval_dataset,
            config=config_dict,
            total_rows=len(rows),
        )

        cache_backend = JsonFileCacheBackend()

        # Validate --output-path
        if output_path is not None:
            output_path_resolved = Path(output_path).resolve()
            cache_root_resolved = _get_cache_root().resolve()
            if output_path_resolved == cache_root_resolved:
                await proxy.stop()
                if baseline_proxy:
                    await baseline_proxy.stop()
                return self._fail(
                    "Error: --output-path cannot be the same as the cache root directory."
                )

        def on_progress(current: int, total: int, result: dict) -> None:
            if quiet:
                return
            if output.format is OutputFormat.json:
                return

            status_style = "green" if result["success"] else "red"
            status = "OK" if result["success"] else "FAILED"
            duration = format_duration(result.get("duration_ms", 0))

            tag_prefix = ""
            if result.get("model_tag"):
                tag_prefix = f"[{result['model_tag']}] "

            reward_str = ""
            if result["success"] and result.get("reward") is not None:
                reward_str = f" [reward={result['reward']:.3f}]"

            error_suffix = ""
            if not result["success"] and result.get("error"):
                error_suffix = f" - {truncate_error(result['error'])}"

            tokens = result.get("tokens", 0)
            line = (
                f"[{current}/{total}] {tag_prefix}{status} "
                f"({duration}, {tokens:,} tokens){reward_str}{error_suffix}"
            )
            if output.format is OutputFormat.plain:
                self._stderr_line(line)
                return
            status_styled = self.console.format_styled(status, status_style)
            self.console.print(
                f"[{current}/{total}] {tag_prefix}{status_styled} "
                f"({duration}, {tokens:,} tokens){reward_str}{error_suffix}"
            )

        if not quiet and output.format is not OutputFormat.json:
            self.console.print()
            n_info = f" x{config.runs_n} runs" if config.runs_n > 1 else ""
            batch_info = f", batch_size={batch_size}" if batch_size > 1 else ""
            model_info = " x2 models" if config.baseline_model else ""
            message = f"Running evaluation ({len(rows)} rows{n_info}{model_info}{batch_info})..."
            if output.format is OutputFormat.plain:
                self._stderr_line(message)
            else:
                self.console.print(message)

        # 12. Run orchestrator
        from osmosis_ai.eval.evaluation.orchestrator import EvalOrchestrator

        orchestrator = EvalOrchestrator(
            drivers=drivers,
            cache_backend=cache_backend,
            cache_config=cache_config,
            rows=rows,
            n_runs=config.runs_n,
            pass_threshold=config.runs_pass_threshold,
            batch_size=batch_size,
            log_samples=log_samples,
            fresh=fresh,
            retry_failed=retry_failed,
            dataset_path=Path(config.eval_dataset),
            dataset_fingerprint=dataset_fingerprint,
            start_index=offset,
            on_progress=on_progress,
        )

        try:
            orch_result = await orchestrator.run()
        except Exception as e:
            if debug:
                import traceback

                traceback.print_exc()
            return self._fail(f"Error during evaluation: {e}", code="INTERNAL")
        finally:
            await proxy.stop()
            if baseline_proxy:
                await baseline_proxy.stop()

        # 13. Handle result status
        if orch_result.status == "already_completed":
            results_path: Path | None = None
            if not quiet:
                self.console.print(
                    "\nEvaluation already completed (cached).", style="bold"
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
                    model=config.llm_model,
                    dataset_path=config.eval_dataset,
                    cache_data=orch_result.cache_data,
                    samples_path=orch_result.samples_path,
                    task_id=task_id,
                )
                results_path = results_path
                if not quiet:
                    self.console.print(f"Output written to: {results_path}")
            if structured_output:
                return self._build_eval_run_result(
                    status=orch_result.status,
                    summary=orch_result.summary,
                    total_completed=orch_result.total_completed,
                    total_expected=orch_result.total_expected,
                    cache_path=orch_result.cache_path,
                    samples_path=orch_result.samples_path,
                    output_path=results_path,
                    cache_data=orch_result.cache_data,
                    dataset_fingerprint_warning=orch_result.dataset_fingerprint_warning,
                )
            return 0

        if orch_result.status == "interrupted":
            if not quiet:
                self.console.print(
                    f"\nEvaluation interrupted. "
                    f"Progress: {orch_result.total_completed}/{orch_result.total_expected} runs.",
                    style="yellow",
                )
                self.console.print(f"Cache: {orch_result.cache_path}")
                self.console.print("Re-run the same command to resume.")
            if structured_output:
                return self._build_eval_run_result(
                    status=orch_result.status,
                    summary=orch_result.summary,
                    total_completed=orch_result.total_completed,
                    total_expected=orch_result.total_expected,
                    cache_path=orch_result.cache_path,
                    samples_path=orch_result.samples_path,
                    output_path=None,
                    cache_data=orch_result.cache_data,
                    exit_code=130,
                )
            return 130

        if orch_result.status == "dataset_modified":
            return self._fail(
                f"Error: Dataset was modified during evaluation"
                f"{(' (' + orch_result.stop_reason + ')') if orch_result.stop_reason else ''}. "
                f"Results may be inconsistent. Use --fresh to restart."
            )

        if orch_result.status == "systemic_error":
            if not quiet:
                self.console.print(
                    f"\nEvaluation stopped due to systemic error: {orch_result.stop_reason}",
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
            if structured_output:
                return self._build_eval_run_result(
                    status=orch_result.status,
                    summary=orch_result.summary,
                    total_completed=orch_result.total_completed,
                    total_expected=orch_result.total_expected,
                    cache_path=orch_result.cache_path,
                    samples_path=orch_result.samples_path,
                    output_path=None,
                    cache_data=orch_result.cache_data,
                    exit_code=1,
                )
            return 1

        # completed
        if not quiet:
            self._print_orchestrator_summary(
                orch_result.summary,
                orch_result.total_completed,
                orch_result.total_expected,
            )
            self.console.print(f"\nCache: {orch_result.cache_path}")
            if orch_result.samples_path:
                self.console.print(f"Samples: {orch_result.samples_path}")

        results_path = None
        if output_path:
            results_path = self._write_orchestrator_output(
                output_path=output_path,
                model=config.llm_model,
                dataset_path=config.eval_dataset,
                cache_data=orch_result.cache_data,
                samples_path=orch_result.samples_path,
                task_id=task_id,
            )
            if not quiet:
                self.console.print(f"Output written to: {results_path}")

        if structured_output:
            return self._build_eval_run_result(
                status=orch_result.status,
                summary=orch_result.summary,
                total_completed=orch_result.total_completed,
                total_expected=orch_result.total_expected,
                cache_path=orch_result.cache_path,
                samples_path=orch_result.samples_path,
                output_path=results_path,
                cache_data=orch_result.cache_data,
            )

        return 0


__all__ = ["EvalCommand"]

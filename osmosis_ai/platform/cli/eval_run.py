"""`osmosis eval run` -- local evaluation, driven by the shared eval TOML.

This layer owns everything CLI-shaped: workspace and login context, config
parsing, dataset resolution, confirmation, secret resolution, the progress
display, and the result envelope. Scheduling and durable state live in
``osmosis_ai.eval.local``, which holds no CLI dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError, CLIErrorCode
from osmosis_ai.cli.output import CommandResult, OperationResult
from osmosis_ai.cli.output.context import get_output_context
from osmosis_ai.cli.output.display import format_duration_ms
from osmosis_ai.cli.output.error import classify_error
from osmosis_ai.cli.paths import parse_cli_path
from osmosis_ai.cli.prompts import confirm_async, require_confirmation_async
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli.eval_config import (
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)
from osmosis_ai.platform.cli.secret_resolution import resolve_run_secrets
from osmosis_ai.platform.cli.shared_config import build_submit_summary_rows
from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_workspace_directory_contract,
)

COMMAND_LABEL = "`osmosis eval run`"
EVAL_CONFIG_DIR = "configs/eval"
DEFAULT_OUTPUT_SUBPATH = (".osmosis", "evals")


class _ProgressDisplay:
    """Live one-line progress on a terminal; a printed line per item elsewhere.

    The supervisor reports once per completed work item, so on anything larger
    than a smoke test the line-per-item form scrolls the stage lines -- and the
    plan table -- out of the scrollback. The bar keeps the run on one line and
    leaves the final counts on screen. Redirected rich output falls back to the
    printed form, which is what a log file wants; ``--plain`` and ``--json``
    print no progress at all, because ``console.print`` is a no-op there and
    their stdout belongs to the result alone.
    """

    def __init__(self) -> None:
        self._progress: Any | None = None
        self._task_id: Any = None
        self._resolved = False

    def update(self, snapshot: Any) -> None:
        detail = f"pass {snapshot.pass_rate:.0%} · failed {snapshot.failed}"
        progress = self._start(snapshot)
        if progress is None:
            console.print(
                f"{snapshot.completed}/{snapshot.total} {detail}", style="dim"
            )
            return
        progress.update(self._task_id, completed=snapshot.completed, detail=detail)

    def close(self) -> None:
        progress, self._progress = self._progress, None
        if progress is None:
            return
        # Stopping renders one final frame. An unfinished task -- an interrupted
        # run, a halted dispatch, any run that did not reach its total -- would
        # freeze its spinner mid-animation and leave the glyph on screen for
        # good. Marking the task finished swaps the spinner for its blank
        # finished text, so the line the user keeps carries only the counts.
        for task in progress.tasks:
            if task.finished_time is None:
                task.finished_time = task.elapsed or 0.0
        progress.stop()

    def _start(self, snapshot: Any) -> Any | None:
        if self._resolved:
            return self._progress
        self._resolved = True
        from osmosis_ai.cli.output.context import OutputFormat

        if not console.is_tty or get_output_context().format is not OutputFormat.rich:
            return None
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[detail]}"),
            TimeElapsedColumn(),
            console=console.rich,
        )
        progress.start()
        self._task_id = progress.add_task("rollouts", total=snapshot.total, detail="")
        self._progress = progress
        return progress


@dataclass(frozen=True)
class _Hooks:
    """Bridges the supervisor's callbacks onto CLI output primitives."""

    yes: bool
    secrets_file: str | None
    verbose: bool = False
    display: _ProgressDisplay = field(default_factory=_ProgressDisplay)

    def note(self, message: str) -> None:
        console.print(message, style="dim")

    def stage(self, message: str) -> None:
        # --verbose echoes the whole log stream, and every stage is a line in
        # it; printing here too would double each milestone.
        if self.verbose:
            return
        console.print(f"→ {console.escape(message)}", style="cyan")

    async def confirm_dispatch(self, *, pending: int, model_path: str) -> None:
        await require_confirmation_async(
            f"{pending} rollouts x model {model_path} — continue?",
            yes=self.yes,
            default=False,
        )

    async def confirm_new_run(self, *, run_name: str, total: int) -> bool:
        # Declining (and every non-interactive path, --yes included) keeps the
        # idempotent no-op, so a scripted rerun never silently starts new work.
        # Default yes: a new run touches nothing, and the cost confirmation
        # still stands between the answer and any dispatch.
        from osmosis_ai.cli.output.context import OutputFormat

        output = get_output_context()
        if self.yes or output.format is not OutputFormat.rich or not output.interactive:
            return False
        return bool(
            await confirm_async(
                f"run {run_name!r} already has all {total} results — start a new run?",
                default=True,
            )
        )

    def resolve_secrets(self, names: Sequence[str]) -> dict[str, str]:
        if not names:
            return {}
        # Every secret must resolve locally: a name existing in the platform
        # store does not satisfy it here (§7.3). Provider keys included — the
        # in-process LiteLLM bridge reads them from the environment.
        return resolve_run_secrets(
            names=list(names),
            secrets_file=self.secrets_file,
            stored_names=set(),
        )

    def progress(self, snapshot: Any) -> None:
        self.display.update(snapshot)


def _missing_extra_error(exc: ModuleNotFoundError) -> CLIError:
    # No harbor variant: the rollout server runs in the rollout's own uv
    # environment, so harbor never has to be importable in this process.
    return CLIError(
        "Local evaluation requires optional dependencies. Install them with "
        '`pip install "osmosis-ai[eval]"`.',
        code=CLIErrorCode.VALIDATION,
        details={"missing_module": exc.name, "extra": "eval"},
    )


def _resolve_output_root(output: str | None, workspace_directory: Path) -> Path:
    """Absolute output root.

    Resolved, not merely expanded: the runner decides whether to exclude the
    output tree from the rollout source digest by comparing the two paths, and a
    relative path would silently fail that check and make the run change its own
    digest on every invocation (§5).
    """
    if output is None:
        return workspace_directory.joinpath(*DEFAULT_OUTPUT_SUBPATH).resolve()
    return parse_cli_path(output, expand_user=True).path.resolve()


def _numeric(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CLIError(f"evaluation.{field} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise CLIError(f"evaluation.{field} must be finite, got {value!r}")
    return float(value)


def _integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CLIError(f"evaluation.{field} must be an integer, got {value!r}")
    return value


def run(
    config_path: Path,
    *,
    name: str | None = None,
    output: str | None = None,
    dataset_file: str | None = None,
    secrets_file: str | None = None,
    rows: str | None = None,
    fresh: bool = False,
    retry_failed: bool = False,
    max_in_flight: int | None = None,
    yes: bool = False,
    tunnel: str | None = None,
    rollout_port: int | None = None,
    listener_port: int | None = None,
    advertise_url: str | None = None,
    verbose: bool = False,
    upload: bool = False,
) -> CommandResult:
    """Run an evaluation locally against the workspace's rollout server."""
    import asyncio

    # Fail flag mistakes before the plan table and the cost confirmation.
    if tunnel is not None and tunnel != "cloudflared":
        raise CLIError(f"--tunnel supports only 'cloudflared', got {tunnel!r}")
    if tunnel is not None and advertise_url is not None:
        raise CLIError(
            "--tunnel and --advertise-url are mutually exclusive: --tunnel "
            "manages a tunnel for you, --advertise-url points at one you run."
        )
    if listener_port is not None and not 1 <= listener_port <= 65535:
        raise CLIError(
            f"--listener-port must be between 1 and 65535, got {listener_port}"
        )
    if advertise_url is not None:
        from urllib.parse import urlparse

        parsed = urlparse(advertise_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise CLIError(
                "--advertise-url must be an http(s) base URL with a host, "
                f"got {advertise_url!r}"
            )
        if listener_port is None:
            # An external tunnel forwards to a fixed local port; an ephemeral
            # listener bind would leave it nothing stable to target.
            raise CLIError(
                "--advertise-url requires --listener-port: your tunnel needs "
                "a fixed local port to forward to."
            )
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            console.print_warning(
                "--advertise-url uses plain http: the per-run bridge bearer "
                "and all model traffic cross the network unencrypted."
            )
    # Tunnel flags are runtime-only — deliberately never part of the manifest
    # — so every follow-up command that re-runs rollouts must carry them
    # explicitly: resuming a cloud-sandbox run without them would fail each
    # pending item against the loopback guard.
    rerun_flags = ""
    if tunnel is not None:
        rerun_flags += f" --tunnel {tunnel}"
    if advertise_url is not None:
        rerun_flags += f" --advertise-url {quote(advertise_url)}"
    if listener_port is not None:
        rerun_flags += f" --listener-port {listener_port}"

    try:
        from osmosis_ai.eval.local.dataset import (
            DatasetCache,
            DatasetResolutionError,
            PlatformDatasetFetcher,
            default_dataset_cache_root,
            parse_row_selector,
            resolve_explicit_dataset_file,
            resolve_platform_dataset,
            select_rows,
        )
        from osmosis_ai.eval.local.runner import (
            EvalRunSpec,
            LocalEvalError,
            LocalEvalOptions,
            LocalEvalRunner,
        )
        from osmosis_ai.eval.local.state import LocalEvalStateError
    except ModuleNotFoundError as exc:
        raise _missing_extra_error(exc) from exc

    context = require_git_workspace_directory_context()
    workspace_directory = context.workspace_directory
    validate_workspace_directory_contract(workspace_directory)

    resolved_config_path = config_path.expanduser().resolve()
    ensure_workspace_directory_config_path(
        resolved_config_path,
        workspace_directory,
        config_dir=EVAL_CONFIG_DIR,
        command_label=COMMAND_LABEL,
    )
    config = load_eval_submit_config(resolved_config_path)
    # Path contract only: the rollout runs in its own uv-resolved environment, so
    # importing it here would validate the wrong dependency set.
    validate_eval_submit_context_paths(config, workspace_directory)

    evaluation = config.evaluation_config
    # ``or 1.0`` would turn a configured 0.0 -- "every graded row passes" -- into
    # the default, so the fallback has to test for absence, not falsiness.
    pass_threshold = _numeric(evaluation.get("pass_threshold"), field="pass_threshold")
    n = _integer(evaluation.get("n"), field="n")
    if n is not None and n <= 0:
        raise CLIError(f"evaluation.n must be a positive integer, got {n!r}")
    spec = EvalRunSpec(
        rollout_name=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        model_path=config.experiment_model_path,
        dataset_name=config.experiment_dataset,
        n=1 if n is None else n,
        batch_size=_integer(evaluation.get("batch_size"), field="batch_size"),
        pass_threshold=1.0 if pass_threshold is None else pass_threshold,
        agent_timeout_sec=_numeric(
            evaluation.get("agent_workflow_timeout_s"), field="agent_workflow_timeout_s"
        ),
        grader_timeout_sec=_numeric(
            evaluation.get("grader_timeout_s"), field="grader_timeout_s"
        ),
        env=dict(config.env),
        secret_names=tuple(config.secrets),
        branch=config.experiment_branch,
        commit_sha=config.experiment_commit_sha,
    )

    advanced = config.advanced_config
    if advanced:
        # §5: recorded as provenance, one warning for the rest. Local execution
        # consumes no [advanced] keys today.
        console.print_warning(
            "\\[advanced] keys are recorded but not consumed by `osmosis eval run`: "
            + ", ".join(sorted(advanced))
        )

    output_root = _resolve_output_root(output, workspace_directory)
    rollout_dir = (workspace_directory / "rollouts" / spec.rollout_name).resolve()

    try:
        if dataset_file is not None:
            dataset = resolve_explicit_dataset_file(
                parse_cli_path(dataset_file, expand_user=True).path
            )
        else:
            cache = DatasetCache(default_dataset_cache_root())
            fetcher = PlatformDatasetFetcher(
                credentials=context.credentials, git_identity=context.git_identity
            )
            with get_output_context().status("Resolving dataset..."):
                dataset = resolve_platform_dataset(
                    spec.dataset_name,
                    cache=cache,
                    fetcher=fetcher,
                    on_event=lambda message: console.print(message, style="dim"),
                )
        row_selector = parse_row_selector(rows) if rows else None
        selection = select_rows(
            dataset.path,
            extension=dataset.extension,
            limit=_integer(evaluation.get("limit"), field="limit"),
            row_selector=row_selector,
        )
    except DatasetResolutionError as exc:
        raise CLIError(str(exc)) from exc

    hooks = _Hooks(yes=yes, secrets_file=secrets_file, verbose=verbose)
    runner = LocalEvalRunner(
        spec=spec,
        options=LocalEvalOptions(
            name=name,
            fresh=fresh,
            retry_failed=retry_failed,
            max_in_flight=max_in_flight,
            rollout_port=rollout_port,
            verbose=verbose,
            listener_port=listener_port,
            advertise_url=advertise_url,
            tunnel=tunnel,
        ),
        dataset=dataset,
        selection=selection,
        rollout_dir=rollout_dir,
        output_root=output_root,
        hooks=hooks,
        provenance=_provenance(workspace_directory, spec, advanced=advanced),
    )

    _print_plan(spec, dataset=dataset, selection=selection, output_root=output_root)
    imported: Any | None = None

    def _upload_completed(summary: Any) -> None:
        nonlocal imported

        from osmosis_ai.eval.local.upload import (
            LocalEvalUploadError,
            build_eval_upload_plan,
        )
        from osmosis_ai.platform.cli.eval_upload import upload_plan

        hooks.display.close()
        retry = _upload_command(summary.run_dir)
        try:
            imported = upload_plan(
                build_eval_upload_plan(summary.run_dir), context=context
            )
        except KeyboardInterrupt as exc:
            raise CLIError(
                f"Local evaluation results are complete at {summary.run_dir}, but "
                f"the platform upload was interrupted.\nRetry with: {retry}"
            ) from exc
        except (AuthenticationExpiredError, PlatformAPIError) as exc:
            message = (
                f"Local evaluation results are complete at {summary.run_dir}, but "
                f"the platform upload failed: {exc}\nRetry with: {retry}"
            )
            classified = classify_error(exc)
            raise CLIError(
                message, code=classified.code, details=classified.details
            ) from exc
        except LocalEvalUploadError as exc:
            raise CLIError(
                f"Local evaluation results are complete at {summary.run_dir}, but "
                f"cannot be uploaded: {exc}"
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise CLIError(
                f"Local evaluation results are complete at {summary.run_dir}, but "
                f"the local evaluation upload failed: {exc}\nRetry with: {retry}"
            ) from exc
        except Exception as exc:
            classified = classify_error(exc)
            raise CLIError(
                f"Local evaluation results are complete at {summary.run_dir}, but "
                f"the upload failed: {exc}\nRetry with: {retry}",
                code=classified.code,
                details=classified.details,
            ) from exc

    try:
        summary = asyncio.run(
            runner.run(after_finalize=_upload_completed) if upload else runner.run()
        )
    except (LocalEvalError, LocalEvalStateError) as exc:
        raise CLIError(str(exc)) from exc
    except KeyboardInterrupt:
        started = runner.active_run_name
        hint = (
            "Resume the pending work items: "
            f"osmosis eval run {quote(str(config_path))} --name {started}"
            f"{rerun_flags}"
            if started
            else "Nothing was started."
        )
        raise CLIError(
            f"Interrupted. {hint}",
            code=CLIErrorCode.VALIDATION,
        ) from None
    finally:
        # The bar owns a live terminal region; leaving it running would print
        # the summary, an error, or a traceback into it.
        hooks.display.close()

    _print_failures(summary)
    _print_summary(summary)
    return _result(
        summary,
        context=context,
        config_path=config_path,
        dataset_source=dataset.source,
        imported=imported,
        upload_requested=upload,
        rerun_flags=rerun_flags,
    )


def _provenance(
    workspace_directory: Path, spec: Any, *, advanced: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Record what actually executed. Never mutates the workspace (§5)."""
    from osmosis_ai.platform.cli.workspace_repo import summarize_local_git_state

    state = summarize_local_git_state(workspace_directory)
    provenance: dict[str, Any] = {
        "config_branch": spec.branch,
        "config_commit_sha": spec.commit_sha,
        "advanced": dict(advanced) if advanced else None,
    }
    if state is not None:
        provenance.update(
            {
                "git_head": state.head_sha,
                "git_branch": state.branch,
                "git_dirty": state.is_dirty,
            }
        )
        if spec.commit_sha and state.head_sha != spec.commit_sha:
            console.print_warning(
                f"config pins commit {spec.commit_sha} but the workspace is at "
                f"{state.head_sha}; the local run executes the workspace as it is."
            )
        elif spec.branch and state.branch and state.branch != spec.branch:
            console.print_warning(
                f"config names branch {spec.branch!r} but the workspace is on "
                f"{state.branch!r}; the local run executes the workspace as it is."
            )
    return {key: value for key, value in provenance.items() if value is not None}


def _print_plan(spec: Any, *, dataset: Any, selection: Any, output_root: Path) -> None:
    """What is about to run, before the cost confirmation.

    Deliberately the same table ``osmosis eval submit`` prints: the two commands
    read one TOML, so they owe the user one reading of it.
    """
    rows = build_submit_summary_rows(
        rollout=spec.rollout_name,
        entrypoint=spec.entrypoint,
        model=spec.model_path,
        dataset=f"{dataset.dataset_name} ({dataset.source})",
        branch=spec.branch,
        commit_sha=spec.commit_sha,
    )
    sampled = len(selection.rows)
    runs = max(1, spec.n)
    rows.append(("Rows", f"{sampled} of {selection.total_dataset_rows}"))
    if runs > 1:
        rows.append(("Runs Per Row", str(runs)))
    rows.append(("Work Items", str(sampled * runs)))
    rows.append(("Output", str(output_root)))
    console.table(
        [(label, console.escape(value)) for label, value in rows],
        title="Local Evaluation",
    )


def _print_summary(summary: Any) -> None:
    """The numbers the run produced (§4.3).

    ``metrics`` is what ``metrics.json`` holds, so the terminal and the file
    can never disagree -- nothing here is recomputed.
    """
    metrics: dict[str, Any] = summary.metrics or {}
    completed = summary.succeeded + summary.failed + summary.skipped
    counts = f"{summary.succeeded} ok · {summary.failed} failed"
    if summary.skipped:
        counts += f" · {summary.skipped} skipped"
    rows: list[tuple[str, str]] = [
        ("Run", summary.run_name),
        (
            "Work Items",
            f"{completed}/{summary.total_work_items} complete · {counts}",
        ),
        (
            "Pass Rate",
            f"{metrics.get('pass_rate', 0.0):.1%} "
            f"({metrics.get('passed', 0)}/{metrics.get('completed_samples', 0)} "
            f"scored, threshold {metrics.get('pass_threshold', 1.0):g})",
        ),
    ]
    # Only these two are conditional in ``aggregate_metrics``: rewards need a
    # graded sample, and pass@k needs at least two attempts per row.
    stats = metrics.get("reward_stats")
    if stats:
        rows.append(
            (
                "Reward",
                f"mean {stats['mean']:.3f} · median {stats['median']:.3f} · "
                f"min {stats['min']:.3f} · max {stats['max']:.3f}",
            )
        )
    points = metrics.get("pass_at_k")
    if points:
        rows.append(
            (
                "pass@k",
                " · ".join(f"k={point['k']} {point['value']:.2f}" for point in points),
            )
        )
    rows.append(("Tokens Used", f"{metrics.get('tokens_used', 0):,}"))
    rows.append(("Duration", format_duration_ms(summary.duration_ms)))
    rows.append(("Output", str(summary.run_dir)))
    console.table(
        [(label, console.escape(value)) for label, value in rows],
        title="Results" + (" (incomplete)" if summary.cancelled else ""),
    )


def _print_failures(summary: Any) -> None:
    """Print failed rows with zero-friction paths (§4.3)."""
    if not summary.failures:
        return
    console.separator("Failed rows")
    for failure in summary.failures:
        source = (
            ""
            if failure.source_row_index == failure.row_index
            else f" (source {failure.source_row_index})"
        )
        error = failure.error_type or "unknown"
        console.print(
            f"row {failure.row_index}{source} run {failure.run_index}: "
            f"error_type={error} -> {failure.rollout_dir}"
        )


def _upload_command(run_dir: Path) -> str:
    return f"osmosis eval upload {quote(str(run_dir))}"


def _result(
    summary: Any,
    *,
    context: Any,
    config_path: Path,
    dataset_source: str,
    imported: Any | None = None,
    upload_requested: bool = False,
    rerun_flags: str = "",
) -> OperationResult:
    resource: dict[str, Any] = {
        "run_name": summary.run_name,
        "local_run_id": summary.local_run_id,
        "output_path": str(summary.run_dir),
        "dataset_source": dataset_source,
        "total_work_items": summary.total_work_items,
        "dispatched": summary.dispatched,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "resumed": summary.resumed,
        "cancelled": summary.cancelled,
        "duration_ms": summary.duration_ms,
        "metrics": summary.metrics,
        "failed_rows": [
            {
                "row_index": failure.row_index,
                "source_row_index": failure.source_row_index,
                "run_index": failure.run_index,
                "rollout_id": failure.rollout_id,
                "error_type": failure.error_type,
                "rollout_dir": str(failure.rollout_dir),
            }
            for failure in summary.failures
        ],
    }
    resource.update(git_result_context(context))
    if imported is not None:
        resource["upload"] = {
            "session_id": imported.session_id,
            "eval_run_id": imported.eval_run_id,
            "eval_run_name": imported.eval_run_name,
            "status": imported.status,
            "expected_files": imported.expected_files,
            "uploaded_files": imported.uploaded_files,
        }
        resource["platform_url"] = imported.platform_url
    incomplete = summary.cancelled or (
        summary.succeeded + summary.failed + summary.skipped < summary.total_work_items
    )
    noop = not incomplete and summary.dispatched == 0 and summary.total_work_items > 0
    config_arg = quote(str(config_path))
    next_steps = []
    if noop:
        next_steps.append(
            f"Start a new run alongside it: osmosis eval run {config_arg}{rerun_flags}"
        )
        next_steps.append(
            "Re-run this name from scratch (archives its results): "
            f"osmosis eval run {config_arg} --name {summary.run_name} --fresh"
            f"{rerun_flags}"
        )
    if incomplete:
        upload_flag = " --upload" if upload_requested else ""
        next_steps.append(
            f"Resume: osmosis eval run {config_arg} "
            f"--name {summary.run_name}{upload_flag}{rerun_flags}"
        )
    if summary.failed:
        upload_flag = " --upload" if upload_requested else ""
        next_steps.append(
            "Retry failures under the same name: "
            f"osmosis eval run {config_arg} --name {summary.run_name} "
            f"--retry-failed{upload_flag}{rerun_flags}"
        )
    if imported is not None:
        imported_name = imported.eval_run_name or summary.run_name
        next_steps.append(
            f"View: {imported.platform_url}"
            if imported.platform_url
            else f"Inspect the imported run: osmosis eval info {imported_name}"
        )
    elif not incomplete:
        next_steps.append(f"Upload: {_upload_command(summary.run_dir)}")
    if noop:
        message = (
            f"Run {summary.run_name!r} was already complete; nothing ran: "
            f"{summary.run_dir}"
        )
    else:
        message = (
            f"Evaluation run {'interrupted' if incomplete else 'finished'}: "
            f"{summary.run_dir}"
        )
    if incomplete and upload_requested:
        message += " Upload skipped because the local evaluation is incomplete."
    return OperationResult(
        operation="eval.run",
        status="partial" if incomplete else "success",
        resource=resource,
        message=message,
        display_next_steps=next_steps,
        exit_code=1 if incomplete else 0,
    )

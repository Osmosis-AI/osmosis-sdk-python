"""Training run management commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


def _detail_fields(rows: list[tuple[str, str]]) -> list[Any]:
    from osmosis_ai.cli.output import DetailField

    return [DetailField(label=label, value=value) for label, value in rows]


@app.command("list")
def list_runs(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of runs to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all training runs."),
) -> Any:
    """List training runs for the current workspace directory."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_training_run,
    )
    from osmosis_ai.cli.output.display import (
        format_local_date,
    )
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        format_run_status,
        require_git_workspace_directory_context,
        validate_list_options,
    )
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching training runs..."):
        if fetch_all:
            training_runs, total_count = fetch_all_pages(
                lambda lim, off: client.list_training_runs(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=context.git_identity,
                ),
                items_attr="training_runs",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_training_runs(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                git_identity=context.git_identity,
            )
            training_runs = page.training_runs
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Training Runs",
        items=[serialize_training_run(r) for r in training_runs],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="dataset_name", label="Dataset", ratio=2, overflow="fold"),
            ListColumn(key="model_name", label="Base Model", ratio=2, overflow="fold"),
            ListColumn(key="rollout_name", label="Rollout", ratio=2, overflow="fold"),
            ListColumn(key="created_at", label="Submitted", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Submitted By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_training_run(run),
                "name": run.name or "(unnamed)",
                "status": format_run_status(run),
                "dataset_name": run.dataset_name or "—",
                "model_name": run.model_name or "—",
                "rollout_name": run.rollout_name or "—",
                "created_at": format_local_date(run.created_at),
                "creator_name": run.creator_name or "—",
            }
            for run in training_runs
        ],
        display_hints=["Use osmosis train info <name> for details."],
    )


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Training run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output path for metrics JSON. Non-.json extensions are replaced with"
            " .json; a trailing '/' or existing directory generates a default"
            " filename inside it. (default in rich mode: .osmosis/metrics/)"
        ),
    ),
) -> Any:
    """Show training run details, checkpoints, and metrics."""
    import json

    from osmosis_ai.cli.metrics_export import build_export_dict
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        DetailSection,
        OutputFormat,
        get_output_context,
        serialize_checkpoint,
        serialize_training_run,
    )
    from osmosis_ai.cli.output.display import format_local_datetime
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.models import (
        RUN_STATUSES_IN_PROGRESS,
        RUN_STATUSES_TERMINAL,
    )
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.utils import (
        build_run_detail_rows,
        require_git_workspace_directory_context,
    )
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output_ctx = get_output_context()
    with output_ctx.status("Fetching training run..."):
        run = client.get_training_run(
            name,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    rows = build_run_detail_rows(run)
    if run.status == "pending":
        rows.insert(3, ("Progress", "Waiting to start..."))
    if run.started_at:
        rows.append(("Started", format_local_datetime(run.started_at)))
    if run.completed_at:
        rows.append(("Completed", format_local_datetime(run.completed_at)))
    if run.examples_processed_count is not None:
        rows.append(("Rows Processed", f"{run.examples_processed_count:,}"))
    if run.notes:
        rows.append(("Notes", console.escape(run.notes)))

    checkpoints = []
    sections: list[DetailSection] = []
    display_hints: list[str] = []

    if run.status in RUN_STATUSES_TERMINAL:
        try:
            with output_ctx.status("Fetching checkpoints..."):
                ckpts = client.list_training_run_checkpoints(
                    name,
                    credentials=credentials,
                    git_identity=context.git_identity,
                )
        except PlatformAPIError:
            ckpts = None

        if ckpts is not None:
            fetched_checkpoints = getattr(ckpts, "checkpoints", None)
            if isinstance(fetched_checkpoints, list) and fetched_checkpoints:
                checkpoints = fetched_checkpoints

    metrics_data = None
    metrics_error: str | None = None
    is_in_progress = run.status in RUN_STATUSES_IN_PROGRESS
    if run.status == "pending":
        metrics_error = "Metrics are not yet available for pending training runs."
    else:
        try:
            with output_ctx.status("Fetching metrics..."):
                metrics_data = client.get_training_run_metrics(
                    run.id,
                    credentials=credentials,
                    git_identity=context.git_identity,
                )
        except (PlatformAPIError, KeyError) as exc:
            metrics_error = str(exc) or "Could not fetch metrics data."

    export: dict[str, Any] | None = None
    output_path: str | None = None
    save_warning: str | None = None
    if metrics_data is not None:
        export = build_export_dict(run, metrics_data)
        if metrics_data.overview.total_steps is not None:
            latest = metrics_data.overview.latest_step or 0
            total = metrics_data.overview.total_steps
            rows.insert(3, ("Progress", f"{latest} / {total} rollout steps"))
        if metrics_data.overview.duration_formatted:
            rows.append(("Duration", metrics_data.overview.duration_formatted))

    fields = _detail_fields(rows)
    if run.platform_url:
        display_hints.append(f"View: {run.platform_url}")
    if checkpoints:
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Checkpoint", ratio=4, overflow="fold")
        table.add_column("Step", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("ID", no_wrap=True)
        plain_lines = []
        for cp in checkpoints:
            cp_name = cp.checkpoint_name or "(unnamed)"
            table.add_row(
                Text(cp_name),
                str(cp.checkpoint_step),
                cp.status,
                cp.id[:8],
            )
            plain_lines.append(
                f"Checkpoint: {cp_name} step {cp.checkpoint_step} "
                f"[{cp.status}] {cp.id[:8]}"
            )
        sections.append(DetailSection(rich=table, plain_lines=plain_lines))
        display_hints.append("Deploy with: osmosis deploy <checkpoint-name>")

    if metrics_data is not None and metrics_data.metrics:
        from osmosis_ai.cli.metrics_graph import (
            render_metric_trends,
            should_render_metric_trends,
        )

        if output_ctx.format is OutputFormat.rich and should_render_metric_trends(
            is_tty=console.is_tty,
            terminal_width=console.width,
            metrics=metrics_data.metrics,
        ):
            trends = render_metric_trends(
                metrics_data.metrics, terminal_width=console.width
            )
            if trends:
                sections.append(DetailSection(rich=trends))
    elif metrics_data is not None:
        fields.append(DetailField(label="Metrics", value="No metric data found."))

    if metrics_error is not None:
        fields.append(DetailField(label="Metrics", value=metrics_error))

    if is_in_progress and metrics_data is not None:
        fields.append(
            DetailField(
                label="Note",
                value="Training is in progress. Metrics shown are a snapshot.",
            )
        )

    if metrics_data is not None:
        should_write_file = output is not None or output_ctx.format is OutputFormat.rich
        if should_write_file:
            try:
                out_path = (
                    _resolve_output_path(output, run.name, run.id)
                    if output
                    else _resolve_default_output(
                        run.name,
                        run.id,
                        workspace_directory=context.workspace_directory,
                    )
                )
                out_path.write_text(
                    json.dumps(export, indent=2, ensure_ascii=False) + "\n"
                )
                output_path = str(out_path)
                display_hints.append(f"Saved metrics to {output_path}")
            except (CLIError, OSError) as exc:
                save_warning = f"Could not save metrics: {exc}"
                display_hints.append(save_warning)

    return DetailResult(
        title="Training Run Info",
        data={
            "training_run": {
                **serialize_training_run(run),
                "examples_processed_count": run.examples_processed_count,
                "notes": run.notes,
            },
            **({"platform_url": run.platform_url} if run.platform_url else {}),
            "checkpoints": [serialize_checkpoint(cp) for cp in checkpoints],
            "in_progress": is_in_progress,
            "metrics_available": metrics_data is not None,
            "metrics_error": metrics_error,
            "metrics": export,
            "output_path": output_path,
            "save_warning": save_warning,
            **git_result_context(context),
        },
        fields=fields,
        sections=sections,
        display_hints=display_hints,
    )


@app.command("submit")
def submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to training config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a new training run."""
    from osmosis_ai.platform.cli.train import submit as _submit

    return _submit(config_path, yes=yes)


def _safe_name(name: str) -> str:
    """Sanitise a run name for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _default_filename(run_name: str | None, run_id: str) -> str:
    """Build the default JSON filename from run metadata."""
    short_id = run_id[:8]
    safe = _safe_name(run_name) if run_name else None
    return f"{safe}_{short_id}.json" if safe else f"{short_id}.json"


def _resolve_output_path(output: str, run_name: str | None, run_id: str) -> Path:
    """Resolve a user-supplied ``-o`` value into a concrete file path.

    Rules:
    * Trailing ``/`` or existing directory → directory mode (generate default
      filename inside the directory).
    * Has a file extension → use as-is.
    * No extension → auto-append ``.json``.

    Parent directories are created automatically.
    """
    from osmosis_ai.cli.paths import parse_cli_path

    parsed_output = parse_cli_path(output)
    path = parsed_output.path

    try:
        # Directory mode: trailing separator or existing directory
        if parsed_output.has_trailing_separator or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            return path / _default_filename(run_name, run_id)

        # Ensure .json extension (output is always JSON)
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(f"Cannot create output path: {exc}") from exc

    return path


def _resolve_default_output(
    run_name: str | None, run_id: str, *, workspace_directory: Path
) -> Path:
    """Resolve the default output path under .osmosis/metrics/."""
    metrics_dir = workspace_directory / ".osmosis" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / _default_filename(run_name, run_id)


@app.command("stop")
def stop(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    require_confirmation(
        f'Stop training run "{name}"?',
        yes=yes,
        summary=[("Name", name)],
    )

    output = get_output_context()
    with output.status("Stopping training run..."):
        client.stop_training_run(
            name,
            credentials=credentials,
            git_identity=context.git_identity,
        )
    return OperationResult(
        operation="train.stop",
        status="success",
        resource={"name": name, **git_result_context(context)},
        message=f'Training run "{name}" stopped.',
    )

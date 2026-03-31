"""Training run management commands."""

from __future__ import annotations

import os
import re
from pathlib import Path

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError, not_implemented

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


@app.command("list")
def list_runs(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum number of runs to show."),
) -> None:
    """List training runs for a project."""
    from osmosis_ai.platform.cli.project import _require_auth, _resolve_project_id
    from osmosis_ai.platform.cli.utils import format_date, format_run_status

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    with console.spinner("Fetching training runs..."):
        project_id = _resolve_project_id(project, workspace_name=ws_name)
        client = OsmosisClient()
        result = client.list_training_runs(
            project_id, limit=limit, credentials=credentials
        )

    if not result.training_runs:
        console.print("No training runs found.")
        return

    console.print(f"Training Runs ({result.total_count}):", style="bold")
    for r in result.training_runs:
        status_str = format_run_status(r)
        name = (
            console.escape(r.name)
            if r.name
            else console.format_styled("(unnamed)", "dim")
        )
        model = console.escape(r.model_name) if r.model_name else "—"
        acc = f"acc:{r.eval_accuracy:.2f}" if r.eval_accuracy is not None else ""
        date = format_date(r.created_at)

        console.print(f"  {r.id}  {name}  {status_str}  {model}  {acc}  {date}")

    if result.has_more:
        remaining = result.total_count - len(result.training_runs)
        console.print(f"  ... and {remaining} more")


@app.command("status")
def status(
    id: str = typer.Argument(
        ..., help="Training run ID (or short prefix from 'train list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Show training run details."""
    from osmosis_ai.platform.cli.project import _require_auth
    from osmosis_ai.platform.cli.utils import (
        build_run_detail_rows,
        format_date,
        resolve_run_id,
    )

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    run_id = resolve_run_id(id, project, ws_name, credentials, client=client)
    run = client.get_training_run(run_id, credentials=credentials)

    rows = build_run_detail_rows(run)
    # Additional fields only shown in the detailed status view
    if run.output_model_id:
        rows.append(("Output Model", run.output_model_id))
    if run.examples_processed_count is not None:
        rows.append(("Examples", str(run.examples_processed_count)))
    if run.notes:
        rows.append(("Notes", console.escape(run.notes)))
    if run.hf_status:
        rows.append(("HF Status", run.hf_status))
    if run.started_at:
        rows.append(("Started", format_date(run.started_at)))
    if run.completed_at:
        rows.append(("Completed", format_date(run.completed_at)))

    console.table(rows, title="Training Run")


@app.command("submit")
def submit() -> None:
    """Submit a new training run."""
    not_implemented("train", "submit")


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
    path = Path(output)

    try:
        # Directory mode: trailing separator or existing directory
        if output.endswith(("/", os.sep)) or path.is_dir():
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
    run_name: str | None, run_id: str, *, cwd: Path | None = None
) -> Path:
    """Resolve the default output path under .osmosis/metrics/."""
    if cwd is None:
        cwd = Path.cwd()
    workspace_toml = cwd / ".osmosis" / "workspace.toml"
    if not workspace_toml.is_file():
        raise CLIError(
            "Not in an Osmosis workspace directory.\n"
            "  Run from a directory created by 'osmosis init',"
            " or use -o to specify an output path."
        )
    metrics_dir = cwd / ".osmosis" / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    return metrics_dir / _default_filename(run_name, run_id)


@app.command("metrics")
def metrics(
    id: str = typer.Argument(
        ..., help="Training run ID (or short prefix from 'train list')."
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output path. Non-.json extensions are replaced with .json;"
            " a trailing '/' or existing directory generates a default"
            " filename inside it. (default: .osmosis/metrics/)"
        ),
    ),
) -> None:
    """Export training run metrics to a JSON file."""
    import json

    from osmosis_ai.cli.metrics_export import build_export_dict
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.models import RUN_STATUSES_TERMINAL
    from osmosis_ai.platform.cli.project import _require_auth
    from osmosis_ai.platform.cli.utils import platform_entity_url

    ws_name, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner("Fetching training run..."):
        run = client.get_training_run(id, credentials=credentials)
        # Best-effort: resolve project name for the platform URL.
        project_name: str | None = None
        if run.project_id:
            try:
                project_detail = client.get_project(
                    run.project_id, credentials=credentials
                )
                project_name = project_detail.project_name
            except Exception:
                pass

    if run.status not in RUN_STATUSES_TERMINAL:
        raise CLIError(
            f"Metrics are only available for terminal training runs "
            f"(current status: {run.status})."
        )

    with console.spinner("Fetching metrics..."):
        metrics_data = client.get_training_run_metrics(run.id, credentials=credentials)
    export = build_export_dict(run, metrics_data)

    # ── Platform URL ──────────────────────────────────────────────
    if project_name:
        url = platform_entity_url(ws_name, project_name, "training", run.id)
        console.print()
        console.print(f"View full details: {url}", style="cyan")
        console.print()

    # ── Summary table ─────────────────────────────────────────────
    rows: list[tuple[str, str]] = []
    if run.name:
        rows.append(("Name", console.escape(run.name)))
    rows.append(("Status", run.status))
    if run.model_name:
        rows.append(("Model", console.escape(run.model_name)))
    if metrics_data.overview.duration_formatted:
        rows.append(("Duration", metrics_data.overview.duration_formatted))
    if metrics_data.overview.reward is not None:
        rows.append(("Final Reward", f"{metrics_data.overview.reward:.4f}"))
    if metrics_data.overview.reward_delta is not None:
        rows.append(("Reward Delta", f"{metrics_data.overview.reward_delta:+.4f}"))
    if metrics_data.overview.examples_processed_count is not None:
        rows.append(("Examples", f"{metrics_data.overview.examples_processed_count:,}"))
    total_steps = export["summary"]["total_steps"]
    if total_steps:
        rows.append(("Steps", f"{total_steps:,}"))

    if not metrics_data.metrics:
        console.print("No metric data found.", style="dim")
    else:
        console.table(rows, title="Training Run Metrics")
        from osmosis_ai.cli.metrics_graph import (
            render_metric_trends,
            should_render_metric_trends,
        )

        if should_render_metric_trends(
            is_tty=console.is_tty,
            terminal_width=console.rich.width,
            metrics=metrics_data.metrics,
        ):
            trends = render_metric_trends(
                metrics_data.metrics, terminal_width=console.rich.width
            )
            if trends:
                console.print()
                console.separator("Metric Trends")
                console.print(trends)

    # ── Save to file (best-effort) ────────────────────────────────
    console.print()
    try:
        out_path = (
            _resolve_output_path(output, run.name, run.id)
            if output
            else _resolve_default_output(run.name, run.id)
        )
        out_path.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n")
        console.print(f"Saved to {out_path}", style="green")
    except (CLIError, OSError) as exc:
        console.print(f"Could not save metrics: {exc}", style="yellow")


@app.command("traces")
def traces() -> None:
    """Show training run traces."""
    not_implemented("train", "traces")


@app.command("stop")
def stop(
    id: str = typer.Argument(
        ..., help="Training run ID (or short prefix from 'train list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Stop a training run."""
    from osmosis_ai.platform.cli.project import _require_auth
    from osmosis_ai.platform.cli.utils import resolve_run_id

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    run_id = resolve_run_id(id, project, ws_name, credentials, client=client)

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(f"Stop training run {run_id[:8]}...?", yes=yes)

    client.stop_training_run(run_id, credentials=credentials)
    console.print(f"Training run {run_id[:8]} stopped.", style="green")


@app.command("delete")
def delete(
    id: str = typer.Argument(
        ..., help="Training run ID (or short prefix from 'train list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a training run."""
    from osmosis_ai.platform.cli.project import _require_auth
    from osmosis_ai.platform.cli.utils import resolve_run_id

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    run_id = resolve_run_id(id, project, ws_name, credentials, client=client)

    # Non-blocking preflight: inform user about preserved output model
    msg = f"Delete training run {run_id[:8]}...? This cannot be undone."
    try:
        affected = client.get_training_run_affected_resources(
            run_id, credentials=credentials
        )
        if affected.output_model:
            m = affected.output_model
            msg += (
                f"\n  Note: output model '{console.escape(m.name)}'"
                f" ({m.id[:8]}) will be preserved."
            )
    except Exception:
        console.print("  Warning: unable to check affected resources.", style="dim")

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(msg, yes=yes)

    result = client.delete_training_run(run_id, credentials=credentials)
    console.print(f"Training run {run_id[:8]} deleted.", style="green")

    if result.preserved_output_model:
        m = result.preserved_output_model
        console.print(
            f"  Output model preserved: {console.escape(m.name)} ({m.id[:8]})",
            style="dim",
        )

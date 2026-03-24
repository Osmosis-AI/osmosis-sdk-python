"""Training run management commands."""

from __future__ import annotations

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import not_implemented

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

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()
    result = client.list_training_runs(project_id, limit=limit, credentials=credentials)

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


@app.command("metrics")
def metrics() -> None:
    """Show training run metrics."""
    not_implemented("train", "metrics")


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

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(
        f"Delete training run {run_id[:8]}...? This cannot be undone.", yes=yes
    )

    result = client.delete_training_run(run_id, credentials=credentials)
    console.print(f"Training run {run_id[:8]} deleted.", style="green")

    if result.preserved_output_model:
        m = result.preserved_output_model
        console.print(
            f"  Output model preserved: {console.escape(m.name)} ({m.id[:8]})",
            style="dim",
        )

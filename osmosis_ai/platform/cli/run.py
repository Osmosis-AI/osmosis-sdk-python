"""Handler for `osmosis run` commands."""

from __future__ import annotations

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.models import (
    RUN_STATUSES_ERROR,
    RUN_STATUSES_IN_PROGRESS,
    RUN_STATUSES_STOPPED,
    RUN_STATUSES_SUCCESS,
)
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)

from .constants import MSG_SESSION_EXPIRED
from .project import _require_auth, _resolve_project_id

app = typer.Typer(help="Manage training runs.")


def _format_run_status(status: str, *, for_prompt: bool = False) -> str:
    """Format a training run status string with color styling."""
    if for_prompt:
        return f"[{status}]"

    if status in RUN_STATUSES_SUCCESS:
        return console.format_styled(f"[{status}]", "green")
    elif status in RUN_STATUSES_IN_PROGRESS:
        return console.format_styled(f"[{status}]", "yellow")
    elif status in RUN_STATUSES_ERROR:
        return console.format_styled(f"[{status}]", "red")
    elif status in RUN_STATUSES_STOPPED:
        return console.format_styled(f"[{status}]", "dim")
    return console.escape(f"[{status}]")


@app.command("list")
def list_runs(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum number of runs to show."),
) -> None:
    """List training runs for a project."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()
    try:
        result = client.list_training_runs(
            project_id, limit=limit, credentials=credentials
        )
    except AuthenticationExpiredError:
        raise CLIError(MSG_SESSION_EXPIRED) from None
    except PlatformAPIError as e:
        raise CLIError(str(e)) from e

    if not result.training_runs:
        console.print("No training runs found.")
        return

    console.print(f"Training Runs ({result.total_count}):", style="bold")
    for r in result.training_runs:
        # Status with processing step
        status_info = f"[{r.status}]"
        if r.processing_step:
            pct = f" {r.processing_percent:.0f}%" if r.processing_percent else ""
            status_info = f"[{r.status}: {r.processing_step}{pct}]"

        # Colorize
        if r.status in RUN_STATUSES_SUCCESS:
            status_str = console.format_styled(status_info, "green")
        elif r.status in RUN_STATUSES_IN_PROGRESS:
            status_str = console.format_styled(status_info, "yellow")
        elif r.status in RUN_STATUSES_ERROR:
            status_str = console.format_styled(status_info, "red")
        elif r.status in RUN_STATUSES_STOPPED:
            status_str = console.format_styled(status_info, "dim")
        else:
            status_str = console.escape(status_info)

        name = r.name or console.format_styled("(unnamed)", "dim")
        model = r.model_name or "—"
        acc = f"acc:{r.eval_accuracy:.2f}" if r.eval_accuracy is not None else ""
        date = r.created_at[:10] if r.created_at else ""

        console.print(f"  {r.id[:8]}  {name}  {status_str}  {model}  {acc}  {date}")

    if result.has_more:
        remaining = result.total_count - len(result.training_runs)
        console.print(f"  ... and {remaining} more")


@app.command("status")
def status(
    id: str = typer.Argument(
        ..., help="Training run ID (or short prefix from 'run list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Show training run details."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    from .utils import resolve_id_prefix

    client = OsmosisClient()

    run_id = id
    if len(id) < 32:
        project_id = _resolve_project_id(project, workspace_name=ws_name)
        try:
            result = client.list_training_runs(
                project_id, limit=50, credentials=credentials
            )
        except AuthenticationExpiredError:
            raise CLIError(MSG_SESSION_EXPIRED) from None
        except PlatformAPIError as e:
            raise CLIError(str(e)) from e
        run_id = resolve_id_prefix(
            id,
            result.training_runs,
            entity_name="training run",
            has_more=result.has_more,
        )

    try:
        run = client.get_training_run(run_id, credentials=credentials)
    except AuthenticationExpiredError:
        raise CLIError(MSG_SESSION_EXPIRED) from None
    except PlatformAPIError as e:
        raise CLIError(str(e)) from e

    rows: list[tuple[str, str]] = [
        ("Name", run.name or "(unnamed)"),
        ("ID", run.id),
        ("Status", run.status),
    ]
    if run.processing_step:
        pct = (
            f" ({run.processing_percent:.0f}%)"
            if run.processing_percent is not None
            else ""
        )
        rows.append(("Step", f"{run.processing_step}{pct}"))
    rows.append(("Model", run.model_name or "—"))
    if run.output_model_id:
        rows.append(("Output Model", run.output_model_id))
    if run.eval_accuracy is not None:
        rows.append(("Accuracy", f"{run.eval_accuracy:.4f}"))
    if run.reward_increase_delta is not None:
        rows.append(("Reward Delta", f"{run.reward_increase_delta:+.4f}"))
    if run.examples_processed_count is not None:
        rows.append(("Examples", str(run.examples_processed_count)))
    if run.error_message:
        rows.append(("Error", run.error_message))
    if run.notes:
        rows.append(("Notes", run.notes))
    if run.hf_status:
        rows.append(("HF Status", run.hf_status))
    if run.creator_name:
        rows.append(("Creator", run.creator_name))
    if run.created_at:
        rows.append(("Created", run.created_at[:10]))
    if run.started_at:
        rows.append(("Started", run.started_at[:10]))
    if run.completed_at:
        rows.append(("Completed", run.completed_at[:10]))

    console.table(rows, title="Training Run")

"""LoRA deployment management commands.

Deployments track the lifecycle of LoRA checkpoints registered with
inference. The ``osmosis deployment`` group covers noun-based CRUD:

    osmosis deployment list                    -> GET    /api/cli/deployments
    osmosis deployment info   <checkpoint>     -> GET    /api/cli/deployments/[checkpointId]
    osmosis deployment rename <old> <new>      -> PATCH  /api/cli/deployments/[checkpointId]
    osmosis deployment delete <checkpoint>     -> DELETE /api/cli/deployments/[checkpointId]

The ``deploy`` / ``undeploy`` verbs live as **top-level** commands
(registered in ``cli/main.py``) to avoid the redundant
``osmosis deployment deploy`` phrasing:

    osmosis deploy    <checkpoint>             -> POST   /api/cli/deployments/[checkpointId]/deploy
    osmosis undeploy  <checkpoint>             -> POST   /api/cli/deployments/[checkpointId]/undeploy

``<checkpoint>`` accepts either a checkpoint UUID or a ``checkpoint_name``;
the platform resolves both forms.
"""

from __future__ import annotations

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage LoRA deployments (list, info, rename, delete).",
    no_args_is_help=True,
)


def _deployment_status_style(status: str) -> str | None:
    """Return the Rich style for a deployment status, or None if unstyled."""
    from osmosis_ai.platform.api.models import (
        DEPLOYMENT_STATUSES_ERROR,
        DEPLOYMENT_STATUSES_INACTIVE,
        DEPLOYMENT_STATUSES_SUCCESS,
    )

    if status in DEPLOYMENT_STATUSES_SUCCESS:
        return "green"
    if status in DEPLOYMENT_STATUSES_INACTIVE:
        return "dim"
    if status in DEPLOYMENT_STATUSES_ERROR:
        return "red"
    return None


def _format_deployment_status(status: str) -> str:
    """Render a deployment status token with Rich styling."""
    style = _deployment_status_style(status)
    label = f"[{status}]"
    if style:
        return console.format_styled(label, style)
    return console.escape(label)


@app.command("list")
def list_deployments(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of deployments to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all deployments."),
) -> None:
    """List LoRA deployments in the current workspace."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        format_dim_date,
        paginated_fetch,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _, credentials = _require_auth()

    with console.spinner("Fetching deployments..."):
        client = OsmosisClient()
        deployments, total_count, _has_more = paginated_fetch(
            lambda lim, off: client.list_deployments(
                limit=lim, offset=off, credentials=credentials
            ),
            items_attr="deployments",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    if not deployments:
        console.print("No deployments found.")
        return

    console.print(f"Deployments ({total_count}):", style="bold")
    for d in deployments:
        status_str = _format_deployment_status(d.status)
        name = console.escape(d.checkpoint_name) if d.checkpoint_name else "—"
        base_model = console.escape(d.base_model) if d.base_model else "—"
        step = f"step:{d.checkpoint_step}"
        date = format_dim_date(d.created_at)
        console.print(
            f"  {name}  {status_str}  {base_model}  {step}  {date}",
            highlight=False,
        )

    print_pagination_footer(len(deployments), total_count, "deployments")


@app.command("info")
def info(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> None:
    """Show deployment details for a checkpoint."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth, format_date

    _, credentials = _require_auth()

    client = OsmosisClient()
    d = client.get_deployment(checkpoint, credentials=credentials)

    rows: list[tuple[str, str]] = [
        ("Checkpoint", console.escape(d.checkpoint_name) if d.checkpoint_name else "—"),
        ("ID", d.id),
        ("Status", d.status),
        ("Base Model", console.escape(d.base_model) if d.base_model else "—"),
        ("Step", str(d.checkpoint_step)),
    ]
    if d.training_run_name:
        rows.append(("Training Run", console.escape(d.training_run_name)))
    elif d.training_run_id:
        rows.append(("Training Run", d.training_run_id))
    if d.creator_name:
        rows.append(("Creator", console.escape(d.creator_name)))
    if d.created_at:
        rows.append(("Created", format_date(d.created_at)))

    console.table(rows, title="Deployment")


def deploy(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> None:
    """Deploy (or reactivate) a LoRA checkpoint."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner(f'Deploying checkpoint "{checkpoint}"...'):
        result = client.deploy_checkpoint(checkpoint, credentials=credentials)

    status_str = _format_deployment_status(result.status)
    name = console.escape(result.checkpoint_name) if result.checkpoint_name else "—"
    console.print(f"Deployment {name} {status_str}", style="green")


def undeploy(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> None:
    """Undeploy a LoRA checkpoint (transition to ``inactive``)."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner(f'Undeploying checkpoint "{checkpoint}"...'):
        result = client.undeploy_checkpoint(checkpoint, credentials=credentials)

    status_str = _format_deployment_status(result.status)
    name = console.escape(result.checkpoint_name) if result.checkpoint_name else "—"
    console.print(f"Deployment {name} {status_str}", style="green")


@app.command("rename")
def rename(
    checkpoint: str = typer.Argument(
        ..., help="Current checkpoint UUID or name.", metavar="CHECKPOINT"
    ),
    new_name: str = typer.Argument(
        ..., help="New checkpoint name.", metavar="NEW_NAME"
    ),
) -> None:
    """Rename a LoRA checkpoint (re-registers inference if active)."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner("Renaming checkpoint..."):
        result = client.rename_checkpoint(checkpoint, new_name, credentials=credentials)

    old = console.escape(result.old_checkpoint_name)
    new = console.escape(result.checkpoint_name)
    console.print(f"Renamed {old} -> {new}", style="green")
    if result.status == "failed":
        console.print(
            "Warning: inference re-registration failed; deployment is marked as failed.",
            style="yellow",
        )


@app.command("delete")
def delete(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a deployment record (idempotent)."""
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()

    require_confirmation(
        f'Delete deployment for checkpoint "{checkpoint}"? This cannot be undone.',
        yes=yes,
    )

    client = OsmosisClient()
    client.delete_deployment(checkpoint, credentials=credentials)
    console.print(f'Deployment for "{checkpoint}" deleted.', style="green")

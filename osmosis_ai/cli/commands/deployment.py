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

from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage LoRA deployments (list, info, rename, delete).",
    no_args_is_help=True,
)


def _detail_fields(rows: list[tuple[str, str]]) -> list[Any]:
    from osmosis_ai.cli.output import DetailField

    return [DetailField(label=label, value=value) for label, value in rows]


def _require_confirmation(message: str, *, yes: bool) -> None:
    if yes:
        return

    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is not OutputFormat.rich or not output.interactive:
        err = CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )
        if output.format is OutputFormat.json:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            emit_structured_error_to_stderr(err)
            raise typer.Exit(1)
        raise err

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(message, yes=yes)


def _deployment_summary_resource(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "checkpoint_name": result.checkpoint_name,
        "status": result.status,
    }


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
) -> Any:
    """List LoRA deployments in the current workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_deployment,
    )
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        fetch_all_pages,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _, credentials = _require_auth()

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching deployments..."):
        if fetch_all:
            deployments, total_count = fetch_all_pages(
                lambda lim, off: client.list_deployments(
                    limit=lim, offset=off, credentials=credentials
                ),
                items_attr="deployments",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_deployments(
                limit=effective_limit, offset=0, credentials=credentials
            )
            deployments = page.deployments
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Deployments",
        items=[serialize_deployment(dep) for dep in deployments],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        columns=[
            ListColumn(key="checkpoint_name", label="Checkpoint"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="base_model", label="Base Model"),
            ListColumn(key="checkpoint_step", label="Step"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )


@app.command("info")
def info(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Show deployment details for a checkpoint."""
    from osmosis_ai.cli.output import (
        DetailResult,
        get_output_context,
        serialize_deployment,
    )
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth, format_date

    _, credentials = _require_auth()

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching deployment..."):
        d = client.get_deployment(checkpoint, credentials=credentials)

    rows: list[tuple[str, str]] = [
        ("Checkpoint", console.escape(d.checkpoint_name) if d.checkpoint_name else "—"),
        ("ID", d.id),
        ("Status", console.escape(d.status)),
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

    return DetailResult(
        title="Deployment",
        data=serialize_deployment(d),
        fields=_detail_fields(rows),
    )


def deploy(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Deploy (or reactivate) a LoRA checkpoint."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Deploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.deploy_checkpoint(checkpoint, credentials=credentials)

    op_status = "failed" if result.status == "failed" else "success"
    return OperationResult(
        operation="deploy",
        status=op_status,
        resource=_deployment_summary_resource(result),
        message=f"Deployment {result.checkpoint_name or '-'} {result.status}",
        display_next_steps=[
            f"Inspect with: osmosis deployment info {result.checkpoint_name}"
        ]
        if result.checkpoint_name
        else [],
        next_steps_structured=[
            {"action": "deployment_info", "checkpoint_name": result.checkpoint_name}
        ]
        if result.checkpoint_name
        else [],
        exit_code=1 if op_status == "failed" else 0,
    )


def undeploy(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Undeploy a LoRA checkpoint (transition to ``inactive``)."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Undeploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.undeploy_checkpoint(checkpoint, credentials=credentials)

    op_status = "failed" if result.status == "failed" else "success"
    return OperationResult(
        operation="undeploy",
        status=op_status,
        resource=_deployment_summary_resource(result),
        message=f"Deployment {result.checkpoint_name or '-'} {result.status}",
        display_next_steps=[
            f"Inspect with: osmosis deployment info {result.checkpoint_name}"
        ]
        if result.checkpoint_name
        else [],
        next_steps_structured=[
            {"action": "deployment_info", "checkpoint_name": result.checkpoint_name}
        ]
        if result.checkpoint_name
        else [],
        exit_code=1 if op_status == "failed" else 0,
    )


@app.command("rename")
def rename(
    checkpoint: str = typer.Argument(
        ..., help="Current checkpoint UUID or name.", metavar="CHECKPOINT"
    ),
    new_name: str = typer.Argument(
        ..., help="New checkpoint name.", metavar="NEW_NAME"
    ),
) -> Any:
    """Rename a LoRA checkpoint (re-registers inference if active)."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()
    client = OsmosisClient()
    output = get_output_context()

    with output.status("Renaming checkpoint..."):
        result = client.rename_checkpoint(checkpoint, new_name, credentials=credentials)

    display_next_steps = []
    if result.status == "failed":
        display_next_steps.append(
            "Warning: inference re-registration failed; deployment is marked as failed."
        )
    return OperationResult(
        operation="deployment.rename",
        status="failed" if result.status == "failed" else "success",
        resource={
            "id": result.id,
            "old_checkpoint_name": result.old_checkpoint_name,
            "checkpoint_name": result.checkpoint_name,
            "status": result.status,
        },
        message=f"Renamed {result.old_checkpoint_name} -> {result.checkpoint_name}",
        display_next_steps=display_next_steps,
        exit_code=1 if result.status == "failed" else 0,
    )


@app.command("delete")
def delete(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Delete a deployment record (idempotent)."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()

    _require_confirmation(
        f'Delete deployment for checkpoint "{checkpoint}"? This cannot be undone.',
        yes=yes,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Deleting deployment..."):
        client.delete_deployment(checkpoint, credentials=credentials)
    return OperationResult(
        operation="deployment.delete",
        status="success",
        resource={"checkpoint": checkpoint},
        message=f'Deployment for "{checkpoint}" deleted.',
    )

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
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat
from osmosis_ai.cli.prompts import Choice, Separator, select_list
from osmosis_ai.platform.cli.workspace_context import WorkspaceContext
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage LoRA deployments (list, info, rename, delete).",
    no_args_is_help=True,
)


def _detail_fields(rows: list[tuple[str, str]]) -> list[Any]:
    from osmosis_ai.cli.output import DetailField

    return [DetailField(label=label, value=value) for label, value in rows]


def _workspace_result_context(workspace: Any) -> dict[str, Any]:
    return {
        "workspace": {"id": workspace.workspace_id, "name": workspace.workspace_name},
        "project_root": str(workspace.project_root),
    }


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


def _select_checkpoint_for_deploy(workspace: WorkspaceContext) -> str:
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    runs = client.list_training_runs(
        credentials=workspace.credentials,
        workspace_id=workspace.workspace_id,
    ).training_runs
    if not runs:
        raise CLIError("No training runs with deployable checkpoints found.")

    while True:
        run_choices: list[str | Choice | Separator] = [
            Choice(r.name or r.id, value=r) for r in runs
        ]
        selected_run = select_list(
            "Choose a training run",
            items=run_choices,
            actions=[Choice("Cancel", value=None)],
        )
        if selected_run is None:
            raise CLIError("Deploy cancelled.")
        checkpoints = client.list_training_run_checkpoints(
            selected_run.id,
            credentials=workspace.credentials,
            workspace_id=workspace.workspace_id,
        ).checkpoints
        if not checkpoints:
            run_label = selected_run.name or selected_run.id
            raise CLIError(
                f'No deployable checkpoints found for training run "{run_label}".'
            )

        checkpoint_choices: list[str | Choice | Separator] = [
            Choice(cp.checkpoint_name or cp.id, value=cp) for cp in checkpoints
        ]
        selected_checkpoint = select_list(
            "Choose a checkpoint",
            items=checkpoint_choices,
            actions=[Choice("Back", value="__back__"), Choice("Cancel", value=None)],
        )
        if selected_checkpoint == "__back__":
            continue
        if selected_checkpoint is None:
            raise CLIError("Deploy cancelled.")
        return selected_checkpoint.checkpoint_name or selected_checkpoint.id


@app.command("list")
def list_deployments(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of deployments to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all deployments."),
) -> Any:
    """List LoRA deployments in the linked project workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_deployment,
    )
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        require_workspace_context,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching deployments..."):
        if fetch_all:
            deployments, total_count = fetch_all_pages(
                lambda lim, off: client.list_deployments(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    workspace_id=workspace_id,
                ),
                items_attr="deployments",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_deployments(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                workspace_id=workspace_id,
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
        extra=_workspace_result_context(workspace),
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
    from osmosis_ai.platform.cli.utils import format_date, require_workspace_context

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching deployment..."):
        d = client.get_deployment(
            checkpoint,
            credentials=credentials,
            workspace_id=workspace_id,
        )

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

    data = serialize_deployment(d)
    data.update(_workspace_result_context(workspace))
    return DetailResult(
        title="Deployment",
        data=data,
        fields=_detail_fields(rows),
    )


def deploy(
    checkpoint: str | None = typer.Argument(
        None, help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Deploy (or reactivate) a LoRA checkpoint."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import require_workspace_context

    output = get_output_context()
    if checkpoint is None and (
        output.format is not OutputFormat.rich or not output.interactive
    ):
        raise CLIError(
            "Checkpoint is required in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    if checkpoint is None:
        checkpoint = _select_checkpoint_for_deploy(workspace)

    client = OsmosisClient()

    with output.status(f'Deploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.deploy_checkpoint(
            checkpoint,
            credentials=credentials,
            workspace_id=workspace_id,
        )

    op_status = "failed" if result.status == "failed" else "success"
    resource = _deployment_summary_resource(result)
    resource.update(_workspace_result_context(workspace))
    return OperationResult(
        operation="deploy",
        status=op_status,
        resource=resource,
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
    from osmosis_ai.platform.cli.utils import require_workspace_context

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Undeploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.undeploy_checkpoint(
            checkpoint,
            credentials=credentials,
            workspace_id=workspace_id,
        )

    op_status = "failed" if result.status == "failed" else "success"
    resource = _deployment_summary_resource(result)
    resource.update(_workspace_result_context(workspace))
    return OperationResult(
        operation="undeploy",
        status=op_status,
        resource=resource,
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
    from osmosis_ai.platform.cli.utils import require_workspace_context

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    client = OsmosisClient()
    output = get_output_context()

    with output.status("Renaming checkpoint..."):
        result = client.rename_checkpoint(
            checkpoint,
            new_name,
            credentials=credentials,
            workspace_id=workspace_id,
        )

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
            **_workspace_result_context(workspace),
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
    from osmosis_ai.platform.cli.utils import require_workspace_context

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id

    _require_confirmation(
        f'Delete deployment for checkpoint "{checkpoint}"? This cannot be undone.',
        yes=yes,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Deleting deployment..."):
        client.delete_deployment(
            checkpoint,
            credentials=credentials,
            workspace_id=workspace_id,
        )
    return OperationResult(
        operation="deployment.delete",
        status="success",
        resource={"checkpoint": checkpoint, **_workspace_result_context(workspace)},
        message=f'Deployment for "{checkpoint}" deleted.',
    )

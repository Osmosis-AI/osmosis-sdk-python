"""Handlers for `osmosis deployment` subcommands and the top-level
``deploy`` / ``undeploy`` verbs.

Deployments track the lifecycle of LoRA checkpoints registered with
inference. The ``cli/commands/deployment.py`` shell delegates here:

    osmosis deployment list                    -> list_deployments()
    osmosis deployment info   <checkpoint>     -> info()
    osmosis deploy            <checkpoint>      -> deploy()
    osmosis undeploy          <checkpoint>      -> undeploy()

``deploy`` / ``undeploy`` are registered as top-level verbs in ``cli/main.py``
via thin shell wrappers to avoid the redundant ``osmosis deployment deploy``
phrasing. ``<checkpoint>`` accepts either a checkpoint UUID or a
``checkpoint_name``; the platform resolves both forms.
"""

from __future__ import annotations

from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    OperationResult,
    OutputFormat,
    get_output_context,
    serialize_deployment,
)
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    DEPLOYMENT_STATUSES_ERROR,
    DEPLOYMENT_STATUSES_INACTIVE,
    DEPLOYMENT_STATUSES_SUCCESS,
)
from osmosis_ai.platform.cli.utils import (
    format_date,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

_CANCEL_ACTION = "__cancel__"


def _detail_fields(rows: list[tuple[str, str]]) -> list[DetailField]:
    return [DetailField(label=label, value=value) for label, value in rows]


def _deployment_summary_resource(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "checkpoint_name": result.checkpoint_name,
        "status": result.status,
    }


def _deployment_status_style(status: str) -> str | None:
    """Return the Rich style for a deployment status, or None if unstyled."""
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


def _select_checkpoint_for_deploy(context: Any) -> str | None:
    from osmosis_ai.cli.prompts import Choice, Separator, confirm, select_list

    client = OsmosisClient()
    runs = client.list_training_runs(
        credentials=context.credentials,
        git_identity=context.git_identity,
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
            actions=[Choice("Cancel", value=_CANCEL_ACTION)],
        )
        if selected_run is None or selected_run == _CANCEL_ACTION:
            return None
        checkpoints = client.list_training_run_checkpoints(
            selected_run.id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        ).checkpoints
        if not checkpoints:
            run_label = selected_run.name or selected_run.id
            console.print(
                f'No deployable checkpoints found for training run "{run_label}".'
            )
            if not confirm("Choose another training run?"):
                return None
            continue

        checkpoint_choices: list[str | Choice | Separator] = [
            Choice(cp.checkpoint_name or cp.id, value=cp) for cp in checkpoints
        ]
        selected_checkpoint = select_list(
            "Choose a checkpoint",
            items=checkpoint_choices,
            actions=[
                Choice("Back", value="__back__"),
                Choice("Cancel", value=_CANCEL_ACTION),
            ],
        )
        if selected_checkpoint == "__back__":
            continue
        if selected_checkpoint is None or selected_checkpoint == _CANCEL_ACTION:
            return None
        return selected_checkpoint.checkpoint_name or selected_checkpoint.id


def list_deployments(*, limit: int, all_: bool) -> ListResult:
    """List LoRA deployments for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching deployments..."):
        deployments, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_deployments(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=git_identity,
            ),
            items_attr="deployments",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Deployments",
        items=[serialize_deployment(dep) for dep in deployments],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="checkpoint_name", label="Checkpoint"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="created_at", label="Deployed"),
            ListColumn(key="creator_name", label="Deployed By"),
        ],
    )


def info(checkpoint: str) -> DetailResult:
    """Show deployment details for a checkpoint."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching deployment..."):
        d = client.get_deployment(
            checkpoint,
            credentials=credentials,
            git_identity=git_identity,
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
    data.update(git_result_context(context))
    return DetailResult(
        title="Deployment",
        data=data,
        fields=_detail_fields(rows),
    )


def deploy(checkpoint: str | None = None) -> OperationResult:
    """Deploy (or reactivate) a LoRA checkpoint."""
    output = get_output_context()
    if checkpoint is None and (
        output.format is not OutputFormat.rich or not output.interactive
    ):
        raise CLIError(
            "Checkpoint is required in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    if checkpoint is None:
        checkpoint = _select_checkpoint_for_deploy(context)
        if checkpoint is None:
            return OperationResult(
                operation="deploy",
                status="cancelled",
                resource=git_result_context(context),
                message="Deploy cancelled.",
            )

    client = OsmosisClient()

    with output.status(f'Deploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.deploy_checkpoint(
            checkpoint,
            credentials=credentials,
            git_identity=git_identity,
        )

    op_status = "failed" if result.status == "failed" else "success"
    resource = _deployment_summary_resource(result)
    resource.update(git_result_context(context))
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


def undeploy(checkpoint: str) -> OperationResult:
    """Undeploy a LoRA checkpoint (transition to ``inactive``)."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Undeploying checkpoint "{console.escape(checkpoint)}"...'):
        result = client.undeploy_checkpoint(
            checkpoint,
            credentials=credentials,
            git_identity=git_identity,
        )

    op_status = "failed" if result.status == "failed" else "success"
    resource = _deployment_summary_resource(result)
    resource.update(git_result_context(context))
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

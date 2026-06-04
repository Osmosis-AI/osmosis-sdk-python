"""LoRA deployment management commands (thin shell delegating to platform/cli/deployment.py).

Deployments track the lifecycle of LoRA checkpoints registered with
inference. The ``osmosis deployment`` group covers noun-based lookup:

    osmosis deployment list                    -> GET    /api/cli/deployments
    osmosis deployment info   <checkpoint>     -> GET    /api/cli/deployments/[checkpointId]

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

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage LoRA deployments (list, info).",
    no_args_is_help=True,
)


@app.command("list")
def list_deployments(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of deployments to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all deployments."),
) -> Any:
    """List LoRA deployments for the current workspace directory."""
    from osmosis_ai.platform.cli.deployment import list_deployments as _list

    return _list(limit=limit, all_=all_)


@app.command("info")
def info(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Show deployment details for a checkpoint."""
    from osmosis_ai.platform.cli.deployment import info as _info

    return _info(checkpoint)


def deploy(
    checkpoint: str | None = typer.Argument(
        None, help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Deploy (or reactivate) a LoRA checkpoint."""
    from osmosis_ai.platform.cli.deployment import deploy as _deploy

    return _deploy(checkpoint)


def undeploy(
    checkpoint: str = typer.Argument(
        ..., help="Checkpoint UUID or checkpoint_name.", metavar="CHECKPOINT"
    ),
) -> Any:
    """Undeploy a LoRA checkpoint (transition to ``inactive``)."""
    from osmosis_ai.platform.cli.deployment import undeploy as _undeploy

    return _undeploy(checkpoint)

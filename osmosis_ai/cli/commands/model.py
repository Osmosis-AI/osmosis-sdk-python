"""Model management commands (thin shell delegating to platform/cli/model.py).

Models cover both base (foundation) models and LoRA models produced by
training runs:

    osmosis model list                      -> GET  /api/cli/models/base + /api/cli/models/lora
    osmosis model deploy   <lora-model>     -> POST /api/cli/models/[modelName]/deploy
    osmosis model undeploy <lora-model>     -> POST /api/cli/models/[modelName]/undeploy
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage models (list, deploy, undeploy).",
    no_args_is_help=True,
)


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of models to show per type.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all models of each type."),
    type_: str = typer.Option(
        "all",
        "--type",
        help="Filter by model type: 'all', 'base', or 'lora'.",
    ),
) -> Any:
    """List base models and LoRA models as two separate lists."""
    from osmosis_ai.platform.cli.model import list_models as _list_models

    return _list_models(limit=limit, all_=all_, type_=type_)


@app.command("deploy")
def deploy(
    lora_model_name: str = typer.Argument(
        ..., help="LoRA model name.", metavar="LORA_MODEL"
    ),
) -> Any:
    """Deploy (or reactivate) a LoRA model."""
    from osmosis_ai.platform.cli.model import deploy as _deploy

    return _deploy(lora_model_name)


@app.command("undeploy")
def undeploy(
    lora_model_name: str = typer.Argument(
        ..., help="LoRA model name.", metavar="LORA_MODEL"
    ),
) -> Any:
    """Undeploy a LoRA model (transition to inactive)."""
    from osmosis_ai.platform.cli.model import undeploy as _undeploy

    return _undeploy(lora_model_name)

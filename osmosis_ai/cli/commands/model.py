"""Base-model management commands (thin shell delegating to platform/cli/model.py).

LoRA deployments live under ``osmosis deployment`` — this group is
scoped to base (foundation) models only.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage base models (list).", no_args_is_help=True)


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of base models to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all base models."),
) -> Any:
    """List base models for the current workspace directory."""
    from osmosis_ai.platform.cli.model import list_models as _list_models

    return _list_models(limit=limit, all_=all_)

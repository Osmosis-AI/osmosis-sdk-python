"""Top-level ``osmosis init`` command."""

from __future__ import annotations

import typer


def init(
    name: str = typer.Argument(
        ..., help="Workspace name (used for directory and config)."
    ),
    here: bool = typer.Option(
        False,
        "--here",
        help="Initialize in current directory instead of creating a subdirectory.",
    ),
) -> None:
    """Initialize a new Osmosis workspace directory."""
    from osmosis_ai.platform.cli.init import init as do_init

    do_init(name=name, here=here)

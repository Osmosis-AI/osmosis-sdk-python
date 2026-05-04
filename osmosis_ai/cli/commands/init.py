"""Top-level ``osmosis init`` command."""

from __future__ import annotations

from typing import Any

import typer


def init(
    name: str = typer.Argument(
        ..., help="Project name (used for directory and config)."
    ),
    here: bool = typer.Option(
        False,
        "--here",
        help="Initialize in current directory instead of creating a subdirectory.",
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace ID or name to link after scaffolding.",
    ),
    no_link: bool = typer.Option(
        False,
        "--no-link",
        help="Create or adopt project without linking to a workspace.",
    ),
) -> Any:
    """Initialize a new Osmosis project directory."""
    from osmosis_ai.platform.cli.init import init as do_init

    return do_init(name=name, here=here, workspace=workspace, no_link=no_link)

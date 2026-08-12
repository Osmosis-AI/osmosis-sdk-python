"""``osmosis quickstart`` command shell.

Thin Typer wrapper that delegates to :mod:`osmosis_ai.platform.cli.quickstart`.
Heavy imports stay inside the command body per the CLI lazy-loading contract.

    osmosis quickstart -> run_quickstart()
"""

from __future__ import annotations

import typer

from osmosis_ai.cli.output import CommandResult

HELP = (
    "Interactive setup: sign in, clone your workspace repository, and get your first "
    "agent prompt."
)


def quickstart(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace to set up. Defaults to the clone you run this from.",
    ),
) -> CommandResult:
    from osmosis_ai.platform.cli.quickstart import run_quickstart

    return run_quickstart(workspace_name=workspace)

"""``osmosis quickstart`` command shell.

Thin Typer wrapper that delegates to :mod:`osmosis_ai.platform.cli.quickstart`.
Heavy imports stay inside the command body per the CLI lazy-loading contract.

    osmosis quickstart -> run_quickstart()
"""

from __future__ import annotations

from typing import Any

HELP = (
    "Interactive setup: sign in, clone your workspace, and get your first agent prompt."
)


def quickstart() -> Any:
    from osmosis_ai.platform.cli.quickstart import run_quickstart

    return run_quickstart()

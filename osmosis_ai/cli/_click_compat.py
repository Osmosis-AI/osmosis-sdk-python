"""Single import point for Typer's vendored Click internals.

Typer 0.26+ vendors Click but does not re-export these symbols publicly yet
(fastapi/typer#1868); the <0.27 pin in pyproject.toml keeps the private paths
stable. Delete this module once upstream exports them.

Never import the external ``click`` package in CLI code: Typer raises the
vendored classes, so external ``click`` types silently never match.
"""

import typer.core
from typer._click.core import Command, Context
from typer._click.exceptions import ClickException, NoArgsIsHelpError, UsageError
from typer._click.globals import get_current_context

# In a coherent install, Typer's own classes derive from the vendored Click
# classes imported above. If typer/ files were overwritten by a stale
# typer-slim (<0.22 ships its own copy), the CLI would degrade silently —
# fail loudly with a fix instead.
if not issubclass(typer.core.TyperGroup, Command):
    raise ImportError(
        "Corrupted typer install: typer's files were overwritten by another "
        "distribution (typically typer-slim<0.22). "
        "Fix: pip install --force-reinstall --no-deps typer"
    )

__all__ = [
    "ClickException",
    "Command",
    "Context",
    "NoArgsIsHelpError",
    "UsageError",
    "get_current_context",
]

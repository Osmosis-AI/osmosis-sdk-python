"""Single import point for Typer's vendored Click internals.

Typer 0.26+ vendors Click but does not re-export these symbols publicly yet
(fastapi/typer#1868); the <0.27 pin in pyproject.toml keeps the private paths
stable. Delete this module once upstream exports them.

Never import the external ``click`` package in CLI code: Typer raises the
vendored classes, so external ``click`` types silently never match.
"""

from typer._click.core import Command, Context
from typer._click.exceptions import ClickException, NoArgsIsHelpError, UsageError
from typer._click.globals import get_current_context

__all__ = [
    "ClickException",
    "Command",
    "Context",
    "NoArgsIsHelpError",
    "UsageError",
    "get_current_context",
]

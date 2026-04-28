from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CLIError(Exception):
    """Raised when the CLI encounters a recoverable error."""

    def __init__(
        self,
        message: str = "",
        *,
        code: str = "VALIDATION",
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}
        self.request_id = request_id


def not_implemented(group: str, cmd: str) -> None:
    """Print a 'not yet implemented' message and exit."""
    import typer

    from osmosis_ai.cli.console import console
    from osmosis_ai.cli.output.context import OutputFormat, get_output_context

    message = f"'osmosis {group} {cmd}' is not yet implemented."
    if get_output_context().format is not OutputFormat.rich:
        raise CLIError(message, code="NOT_IMPLEMENTED")

    console.print(message, style="yellow")
    raise typer.Exit(1)

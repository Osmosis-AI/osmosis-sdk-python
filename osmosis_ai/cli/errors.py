from __future__ import annotations


class CLIError(Exception):
    """Raised when the CLI encounters a recoverable error."""


def not_implemented(group: str, cmd: str) -> None:
    """Print a 'not yet implemented' message and exit."""
    import typer

    from osmosis_ai.cli.console import console

    console.print(f"'osmosis {group} {cmd}' is not yet implemented.", style="yellow")
    raise typer.Exit(1)

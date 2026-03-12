"""CLI commands for Osmosis rollout SDK.

This module provides CLI command handlers for the rollout subsystem,
including the 'serve' and 'validate' commands.

Example:
    osmosis serve --module my_agent:agent_loop --port 9000
"""

from __future__ import annotations

import sys
from typing import Literal

import typer

from osmosis_ai.cli.errors import CLIError

# Valid log levels for uvicorn
LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


def serve(
    module: str = typer.Option(
        ..., "-m", "--module", help="Module path 'module:attribute'."
    ),
    port: int = typer.Option(9000, "-p", "--port", help="Port to bind to."),
    host: str = typer.Option("0.0.0.0", "-H", "--host", help="Host to bind to."),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip agent loop validation."
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development."
    ),
    log_level: LogLevel = typer.Option(
        "info", "--log-level", help="Uvicorn log level."
    ),
    skip_register: bool = typer.Option(
        False, "--skip-register", help="Skip registering with Platform."
    ),
    local_debug: bool = typer.Option(
        False, "--local", "--local-debug", help="Local debug mode."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for TrainGate authentication."
    ),
    debug_dir: str | None = typer.Option(
        None, "--log", metavar="DIR", help="Write execution traces to DIR."
    ),
) -> None:
    """Start a RolloutServer for an agent loop."""
    from osmosis_ai.rollout.cli_utils import load_agent_loop
    from osmosis_ai.rollout.server.serve import serve_agent_loop

    try:
        agent_loop = load_agent_loop(module)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(1) from None

    try:
        serve_agent_loop(
            agent_loop,
            host=host,
            port=port,
            validate=not no_validate,
            log_level=log_level,
            reload=reload,
            skip_register=skip_register,
            api_key=api_key,
            local_debug=local_debug,
            debug_dir=debug_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(1) from None


def validate(
    module: str = typer.Option(
        ..., "-m", "--module", help="Module path 'module:attribute'."
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed validation output."
    ),
) -> None:
    """Validate a RolloutAgentLoop implementation."""
    from osmosis_ai.rollout.cli_utils import load_agent_loop
    from osmosis_ai.rollout.server.serve import validate_and_report

    try:
        agent_loop = load_agent_loop(module)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(1) from None

    result = validate_and_report(agent_loop, verbose=verbose)

    if not result.valid:
        raise typer.Exit(1)

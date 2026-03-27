"""Rollout commands: serve, validate, test."""

from __future__ import annotations

from typing import Literal

import typer

from osmosis_ai.cli.errors import not_implemented

app: typer.Typer = typer.Typer(
    help="Manage rollouts (serve, validate, test, list).",
    no_args_is_help=True,
)

# Valid log levels for uvicorn (defined locally to avoid importing the heavy
# serve module at CLI parse time).
LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


@app.command("serve")
def serve(
    module: str = typer.Option(
        ..., "-m", "--module", help="Module path 'module:attribute'."
    ),
    port: int = typer.Option(9000, "-p", "--port", help="Port to bind to."),
    host: str = typer.Option("0.0.0.0", "-H", "--host", help="Host to bind to."),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip agent loop validation."
    ),
    validate_only: bool = typer.Option(
        False, "--validate-only", help="Validate the agent loop and exit."
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

    agent_loop = load_agent_loop(module)

    if validate_only:
        from osmosis_ai.rollout.server.serve import validate_and_report

        result = validate_and_report(agent_loop, verbose=True)
        if not result.valid:
            raise typer.Exit(1)
        return

    from osmosis_ai.rollout.server.serve import serve_agent_loop

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


@app.command("validate")
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

    agent_loop = load_agent_loop(module)
    result = validate_and_report(agent_loop, verbose=verbose)

    if not result.valid:
        raise typer.Exit(1)


@app.command("test")
def test(
    module: str | None = typer.Option(
        None, "-m", "--module", "--agent", help="Module path 'module:attribute'."
    ),
    mcp: str | None = typer.Option(None, "--mcp", help="Path to MCP tools directory."),
    dataset: str = typer.Option(..., "-d", "--dataset", help="Path to dataset file."),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name to use."),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of rows to test."
    ),
    offset: int = typer.Option(0, "--offset", help="Number of rows to skip."),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the LLM provider."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for OpenAI-compatible APIs."
    ),
    max_turns: int = typer.Option(
        10, "--max-turns", help="Maximum agent turns per row."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens per completion."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="LLM temperature."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output."),
    output: str | None = typer.Option(
        None, "-o", "--output", help="Output results to JSON file."
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Suppress progress output."
    ),
    interactive: bool = typer.Option(
        False, "-i", "--interactive", help="Interactive mode."
    ),
    row: int | None = typer.Option(
        None, "--row", help="Initial row in interactive mode."
    ),
) -> None:
    """Test a RolloutAgentLoop against a dataset."""
    from osmosis_ai.rollout.eval.test_mode.cli import TestCommand

    rc = TestCommand().run(
        module=module,
        mcp=mcp,
        dataset=dataset,
        model=model,
        limit=limit,
        offset=offset,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        debug=debug,
        output=output,
        quiet=quiet,
        interactive=interactive,
        row=row,
    )
    if rc:
        raise typer.Exit(rc)


@app.command("list")
def list_rollouts() -> None:
    """List registered rollouts."""
    not_implemented("rollout", "list")

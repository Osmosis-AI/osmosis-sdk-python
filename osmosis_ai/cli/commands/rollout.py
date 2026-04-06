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


@app.command("test", hidden=True)
def test(
    module: str | None = typer.Option(
        None, "-m", "--module", "--agent", help="Module path."
    ),
    dataset: str = typer.Option(..., "-d", "--dataset", help="Path to dataset file."),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name."),
    limit: int | None = typer.Option(None, "--limit", help="Max rows."),
    offset: int = typer.Option(0, "--offset", help="Skip rows."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL."),
    debug: bool = typer.Option(False, "--debug", help="Debug output."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress output."),
) -> None:
    """Test an AgentWorkflow against a dataset (alias for eval run without grader)."""
    import tempfile
    from pathlib import Path

    from osmosis_ai.eval.evaluation.cli import EvalCommand

    if not module:
        from osmosis_ai.cli.console import Console

        Console().print_error("Error: --module (-m) is required.")
        raise typer.Exit(1)

    # Normalize model for LiteLLM
    llm_model = model if "/" in model else f"openai/{model}"

    # Generate temporary TOML (no [grader] = smoke test mode)
    toml_content = f'''[eval]
module = "{module}"
dataset = "{dataset}"

[llm]
model = "{llm_model}"
'''
    if base_url:
        toml_content += f'base_url = "{base_url}"\n'

    # --api-key → write to temp env var, reference in TOML via api_key_env
    _tmp_env_key = "_OSMOSIS_EVAL_TMP_API_KEY"
    if api_key:
        import os

        os.environ[_tmp_env_key] = api_key
        toml_content += f'api_key_env = "{_tmp_env_key}"\n'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        tmp_path = f.name

    try:
        cmd = EvalCommand()
        rc = cmd.run(
            config_path=tmp_path,
            fresh=False,
            retry_failed=False,
            limit=limit,
            offset=offset,
            quiet=quiet,
            debug=debug,
            output_path=None,
            batch_size_override=None,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up temp env var
        if api_key:
            import os

            os.environ.pop(_tmp_env_key, None)

    if rc:
        raise typer.Exit(rc)


@app.command("list")
def list_rollouts() -> None:
    """List registered rollouts."""
    not_implemented("rollout", "list")

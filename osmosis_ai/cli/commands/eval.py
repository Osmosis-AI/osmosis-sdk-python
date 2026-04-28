"""Eval commands: evaluate agent against dataset with eval functions."""

from __future__ import annotations

from typing import Any, Literal

import typer

app: typer.Typer = typer.Typer(
    help="Evaluate agent against dataset (run, rubric, cache).",
    no_args_is_help=True,
)

cache_app: typer.Typer = typer.Typer(help="Manage eval cache.")
app.add_typer(cache_app, name="cache")


@app.command("run")
def eval_run(
    config_path: str = typer.Argument(..., help="Path to TOML config file."),
    fresh: bool = typer.Option(False, "--fresh", help="Discard cached results."),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-run only failed."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max rows to evaluate."),
    offset: int | None = typer.Option(None, "--offset", help="Skip first N rows."),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Suppress progress output."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging + trace."),
    output_path: str | None = typer.Option(
        None, "-o", "--output-path", help="Override output directory."
    ),
    log_samples: bool = typer.Option(
        False, "--log-samples", help="Save full conversation logs to JSONL."
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Override concurrent batch size."
    ),
) -> Any:
    """Evaluate agent against dataset using TOML config."""
    from osmosis_ai.cli.output import CommandResult
    from osmosis_ai.eval.evaluation.cli import EvalCommand

    cmd = EvalCommand()
    result = cmd.run(
        config_path=config_path,
        fresh=fresh,
        retry_failed=retry_failed,
        limit=limit,
        offset=offset,
        quiet=quiet,
        debug=debug,
        output_path=output_path,
        log_samples=log_samples,
        batch_size_override=batch_size,
    )
    if isinstance(result, CommandResult):
        return result
    if result:
        raise typer.Exit(result)
    return None


@app.command("rubric")
def eval_rubric(
    data: str = typer.Option(
        ..., "-d", "--data", help="Path to JSONL file with conversations."
    ),
    rubric: str = typer.Option(
        ...,
        "-r",
        "--rubric",
        help="Rubric text (inline) or @file.txt to read from file.",
    ),
    model: str = typer.Option(
        ..., "--model", help="Judge model (LiteLLM format, e.g. openai/gpt-5.4)."
    ),
    number: int = typer.Option(
        1, "-n", "--number", help="Number of evaluation runs per record."
    ),
    output_path: str | None = typer.Option(
        None, "-o", "--output", help="Path to write evaluation results as JSON."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the judge model."
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
    score_min: float = typer.Option(0.0, "--score-min", help="Minimum score."),
    score_max: float = typer.Option(1.0, "--score-max", help="Maximum score."),
) -> Any:
    """Evaluate conversations against a rubric using LLM-as-judge."""
    from osmosis_ai.cli.output import CommandResult
    from osmosis_ai.eval.rubric.cli import RubricCommand

    result = RubricCommand().run(
        data=data,
        rubric=rubric,
        model=model,
        number=number,
        output_path=output_path,
        api_key=api_key,
        timeout=timeout,
        score_min=score_min,
        score_max=score_max,
    )
    if isinstance(result, CommandResult):
        return result
    if result:
        raise typer.Exit(result)
    return None


@cache_app.command("dir")
def eval_cache_dir() -> Any:
    """Print cache root directory path."""
    from osmosis_ai.cli.output import CommandResult
    from osmosis_ai.eval.evaluation.cli import EvalCommand

    result = EvalCommand()._run_cache_dir()
    if isinstance(result, CommandResult):
        return result
    if result:
        raise typer.Exit(result)
    return None


@cache_app.command("ls")
def eval_cache_ls(
    cache_model: str | None = typer.Option(
        None, "--model", help="Filter by model name."
    ),
    cache_dataset: str | None = typer.Option(
        None, "--dataset", help="Filter by dataset path."
    ),
    cache_status: Literal["in_progress", "completed"] | None = typer.Option(
        None, "--status", help="Filter by status (in_progress, completed)."
    ),
) -> Any:
    """List cached evaluations."""
    from osmosis_ai.cli.output import CommandResult
    from osmosis_ai.eval.evaluation.cli import EvalCommand

    result = EvalCommand()._run_cache_ls(
        cache_model=cache_model,
        cache_dataset=cache_dataset,
        cache_status=cache_status,
    )
    if isinstance(result, CommandResult):
        return result
    if result:
        raise typer.Exit(result)
    return None


@cache_app.command("rm")
def eval_cache_rm(
    task_id: str | None = typer.Argument(
        None, help="Task ID of the cache entry to delete."
    ),
    rm_all: bool = typer.Option(False, "--all", help="Delete all cached evaluations."),
    cache_model: str | None = typer.Option(
        None, "--model", help="Filter by model name."
    ),
    cache_dataset: str | None = typer.Option(
        None, "--dataset", help="Filter by dataset path."
    ),
    cache_status: Literal["in_progress", "completed"] | None = typer.Option(
        None, "--status", help="Filter by status (in_progress, completed)."
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
) -> Any:
    """Remove cached evaluations."""
    from osmosis_ai.cli.output import CommandResult
    from osmosis_ai.eval.evaluation.cli import EvalCommand

    result = EvalCommand()._run_cache_rm(
        task_id=task_id,
        rm_all=rm_all,
        cache_model=cache_model,
        cache_dataset=cache_dataset,
        cache_status=cache_status,
        yes=yes,
    )
    if isinstance(result, CommandResult):
        return result
    if result:
        raise typer.Exit(result)
    return None

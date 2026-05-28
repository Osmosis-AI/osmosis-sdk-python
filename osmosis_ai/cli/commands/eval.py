"""Eval commands (thin shell delegating to eval/rubric/ and platform/cli/eval.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage evaluation runs (submit, list, info, stop) and LLM-as-judge rubric scoring.",
    no_args_is_help=True,
)


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


@app.command("submit")
def eval_submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to evaluation config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit an evaluation run."""
    from osmosis_ai.platform.cli.eval import submit as _submit

    return _submit(config_path, yes=yes)


@app.command("list")
def eval_list(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of evaluation runs to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all evaluation runs."),
) -> Any:
    """List evaluation runs for the current workspace directory."""
    from osmosis_ai.platform.cli.eval import list_eval_runs as _list_eval_runs

    return _list_eval_runs(limit=limit, all_=all_)


@app.command("info")
def eval_info(
    name_or_id: str = typer.Argument(..., help="Evaluation run name or ID."),
) -> Any:
    """Show evaluation run details and results."""
    from osmosis_ai.platform.cli.eval import info as _info

    return _info(name_or_id)


@app.command("stop")
def eval_stop(
    name_or_id: str = typer.Argument(..., help="Evaluation run name or ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop an evaluation run."""
    from osmosis_ai.platform.cli.eval import stop as _stop

    return _stop(name_or_id, yes=yes)

"""Eval commands (thin shell delegating to eval/rubric/ and platform/cli/eval.py)."""

from __future__ import annotations

from pathlib import Path

import typer

from osmosis_ai.cli.options import (
    all_option,
    cursor_option,
    limit_option,
    log_limit_option,
)
from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help=(
        "Manage evaluation runs (submit, run, upload, list, info, download, stop) and"
        " LLM-as-judge rubric scoring."
    ),
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
        None,
        "--api-key",
        help=(
            "Deprecated. API key for the judge model; prefer the provider "
            "environment variable. Will be removed in a future release."
        ),
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
    score_min: float = typer.Option(0.0, "--score-min", help="Minimum score."),
    score_max: float = typer.Option(1.0, "--score-max", help="Maximum score."),
) -> CommandResult:
    """Evaluate conversations against a rubric using LLM-as-judge."""
    from osmosis_ai.eval.rubric.cli import RubricCommand

    if api_key is not None:
        from osmosis_ai.cli.console import console

        console.print_warning(
            "--api-key is deprecated and will be removed in a future release. "
            "Set the provider environment variable instead.",
            code="DEPRECATION",
        )

    return RubricCommand().run(
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
    secrets_file: str = typer.Option(
        None,
        "--secrets-file",
        help=(
            "Dotenv file supplying values for \\[secrets] names; - reads stdin. "
            "Values are never saved and are re-supplied on every run."
        ),
    ),
) -> CommandResult:
    """Submit an evaluation run."""
    from osmosis_ai.platform.cli.eval import submit as _submit

    return _submit(config_path, yes=yes, secrets_file=secrets_file)


@app.command("run")
def eval_run(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to evaluation config TOML file.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help=(
            "Stable run name. Re-running the same name resumes pending work. "
            "Omit to create a one-off run with a generated name."
        ),
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Run output root (default: .osmosis/evals/).",
    ),
    dataset_file: str | None = typer.Option(
        None,
        "--dataset-file",
        help="Local dataset file to run instead of the platform dataset.",
    ),
    secrets_file: str | None = typer.Option(
        None,
        "--secrets-file",
        help=(
            "Dotenv file supplying values for \\[secrets] names; - reads stdin. "
            "Values are never saved and are re-supplied on every run."
        ),
    ),
    rows: str | None = typer.Option(
        None,
        "--rows",
        help='Dataset rows to run, for example "3,7,10-20". Overrides limit.',
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Archive this run name's existing results and start clean.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Also re-run failed and skipped work items; successes are kept.",
    ),
    max_in_flight: int | None = typer.Option(
        None,
        "--max-in-flight",
        help="Concurrent rollouts (default: evaluation.batch_size, then 1).",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    rollout_port: int | None = typer.Option(
        None,
        "--rollout-port",
        help="Fixed rollout-server port (default: an ephemeral port).",
        rich_help_panel="Advanced",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Stream supervisor and rollout-server log lines to the terminal.",
        rich_help_panel="Advanced",
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload the completed local results to the platform.",
    ),
) -> CommandResult:
    """Run an evaluation locally against this workspace's rollout server."""
    from osmosis_ai.platform.cli.eval_run import run as _run

    return _run(
        config_path,
        name=name,
        output=output,
        dataset_file=dataset_file,
        secrets_file=secrets_file,
        rows=rows,
        fresh=fresh,
        retry_failed=retry_failed,
        max_in_flight=max_in_flight,
        yes=yes,
        rollout_port=rollout_port,
        verbose=verbose,
        upload=upload,
    )


@app.command("upload")
def eval_upload(
    run_dir: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=False,
        resolve_path=False,
        help="Completed local evaluation run directory.",
    ),
) -> CommandResult:
    """Upload a completed local evaluation run to the platform."""
    from osmosis_ai.platform.cli.eval_upload import upload as _upload

    return _upload(run_dir)


@app.command("list")
def eval_list(
    limit: int = limit_option("Maximum number of evaluation runs to show."),
    all_: bool = all_option("Show all evaluation runs."),
) -> CommandResult:
    """List evaluation runs for the current workspace directory."""
    from osmosis_ai.platform.cli.eval import list_eval_runs as _list_eval_runs

    return _list_eval_runs(limit=limit, all_=all_)


@app.command("logs")
def eval_logs(
    name: str = typer.Argument(..., help="Evaluation run name."),
    limit: int = log_limit_option(),
    cursor: str | None = cursor_option(),
) -> CommandResult:
    """Show recent logs for an evaluation run, oldest first."""
    from osmosis_ai.platform.cli.eval import logs as _logs

    return _logs(name, limit=limit, cursor=cursor)


@app.command("info")
def eval_info(
    name: str = typer.Argument(..., help="Evaluation run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Run output root (default in rich mode: .osmosis/evals/<run-name>/).",
    ),
) -> CommandResult:
    """Show evaluation run details, results, and metrics."""
    from osmosis_ai.platform.cli.eval import info as _info

    return _info(name, output=output)


@app.command("download")
def eval_download(
    name: str = typer.Argument(..., help="Evaluation run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Run output root (default: .osmosis/evals/<run-name>/).",
    ),
    types: str = typer.Option(
        "metrics,trajectories",
        "--type",
        help=(
            "Comma-separated selector: metrics, trajectories, artifacts, logs, all. "
            "Replaces the default selection."
        ),
    ),
    rows: str | None = typer.Option(
        None,
        "--rows",
        help='Rows for trajectories/artifacts, for example "3,7,10-20".',
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-download files that already exist locally.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip size confirmation.",
    ),
) -> CommandResult:
    """Download evaluation metrics, trajectories, artifacts, or logs."""
    from osmosis_ai.platform.cli.eval import download as _download

    return _download(
        name,
        output=output,
        types=types,
        rows=rows,
        overwrite=overwrite,
        yes=yes,
    )


@app.command("stop")
def eval_stop(
    name: str = typer.Argument(..., help="Evaluation run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> CommandResult:
    """Stop an evaluation run."""
    from osmosis_ai.platform.cli.eval import stop as _stop

    return _stop(name, yes=yes)

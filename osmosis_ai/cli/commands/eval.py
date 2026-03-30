"""Eval commands: evaluate agent against dataset with eval functions."""

from __future__ import annotations

from typing import Literal

import typer

app: typer.Typer = typer.Typer(
    help="Evaluate agent against dataset (run, rubric, cache).",
    no_args_is_help=True,
)

cache_app: typer.Typer = typer.Typer(help="Manage eval cache.")
app.add_typer(cache_app, name="cache")


@app.command("run")
def eval_run(
    module: str | None = typer.Option(
        None, "-m", "--module", help="Module path 'module:attribute'."
    ),
    mcp: str | None = typer.Option(None, "--mcp", help="Path to MCP tools directory."),
    dataset: str | None = typer.Option(
        None, "-d", "--dataset", help="Path to dataset file."
    ),
    model: str | None = typer.Option(None, "--model", help="Model to evaluate."),
    eval_fns: list[str] | None = typer.Option(
        None, "--eval-fn", help="Eval function 'module:function'. Can be repeated."
    ),
    n_runs: int = typer.Option(1, "--n", help="Number of runs per row for pass@n."),
    pass_threshold: float = typer.Option(
        1.0, "--pass-threshold", help="Score >= threshold counts as pass."
    ),
    max_turns: int = typer.Option(
        10, "--max-turns", help="Maximum agent turns per run."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="LLM sampling temperature."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens per completion."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the LLM provider."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for OpenAI-compatible APIs."
    ),
    baseline_model: str | None = typer.Option(
        None, "--baseline-model", help="Baseline model for comparison."
    ),
    baseline_base_url: str | None = typer.Option(
        None, "--baseline-base-url", help="Base URL for the baseline model."
    ),
    baseline_api_key: str | None = typer.Option(
        None, "--baseline-api-key", help="API key for the baseline model."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Suppress progress output."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of rows to evaluate."
    ),
    offset: int = typer.Option(0, "--offset", help="Number of rows to skip."),
    batch_size: int = typer.Option(
        1, "--batch-size", help="Number of concurrent runs."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Force restart, discarding cached results."
    ),
    log_samples: bool = typer.Option(
        False, "--log-samples", help="Store conversation messages to JSONL."
    ),
    output_path: str | None = typer.Option(
        None, "-o", "--output-path", help="Directory for structured output."
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-execute only failed runs."
    ),
) -> None:
    """Evaluate agent against dataset with eval functions."""
    from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

    cmd = EvalCommand()

    # Convert empty list to None for compatibility
    eval_fns_val = eval_fns if eval_fns else None

    # Validate required args for eval run
    if not dataset:
        cmd.console.print_error("Error: --dataset (-d) is required.")
        raise typer.Exit(1)
    if not model:
        cmd.console.print_error("Error: --model is required.")
        raise typer.Exit(1)
    if not eval_fns_val:
        cmd.console.print_error("Error: --eval-fn is required.")
        raise typer.Exit(1)

    rc = cmd.run(
        module=module,
        mcp=mcp,
        dataset=dataset,
        model=model,
        eval_fns=eval_fns_val,
        n_runs=n_runs,
        pass_threshold=pass_threshold,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        baseline_model=baseline_model,
        baseline_base_url=baseline_base_url,
        baseline_api_key=baseline_api_key,
        debug=debug,
        quiet=quiet,
        limit=limit,
        offset=offset,
        batch_size=batch_size,
        fresh=fresh,
        log_samples=log_samples,
        output_path=output_path,
        retry_failed=retry_failed,
    )
    if rc:
        raise typer.Exit(rc)


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
        ..., "--model", help="Judge model (LiteLLM format, e.g. openai/gpt-4o)."
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
) -> None:
    """Evaluate conversations against a rubric using LLM-as-judge."""
    from osmosis_ai.rollout.eval.rubric.cli import RubricCommand

    RubricCommand().run(
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


@cache_app.command("dir")
def eval_cache_dir() -> None:
    """Print cache root directory path."""
    from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

    rc = EvalCommand()._run_cache_dir()
    if rc:
        raise typer.Exit(rc)


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
) -> None:
    """List cached evaluations."""
    from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

    rc = EvalCommand()._run_cache_ls(
        cache_model=cache_model,
        cache_dataset=cache_dataset,
        cache_status=cache_status,
    )
    if rc:
        raise typer.Exit(rc)


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
) -> None:
    """Remove cached evaluations."""
    from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

    rc = EvalCommand()._run_cache_rm(
        task_id=task_id,
        rm_all=rm_all,
        cache_model=cache_model,
        cache_dataset=cache_dataset,
        cache_status=cache_status,
        yes=yes,
    )
    if rc:
        raise typer.Exit(rc)

"""Eval commands: evaluate agent against dataset with eval functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Evaluate agent against dataset (run, rubric, cache, submit, list, status, stop).",
    no_args_is_help=True,
)

cache_app: typer.Typer = typer.Typer(help="Manage eval cache.")
app.add_typer(cache_app, name="cache")


def _require_eval_local_project() -> None:
    from osmosis_ai.platform.cli.workspace_directory_context import (
        resolve_local_workspace_directory_context,
    )

    resolve_local_workspace_directory_context(require_scaffold=True)


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
    _require_eval_local_project()

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


@app.command("submit")
def eval_submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to eval config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a cloud eval run."""
    import tomllib

    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
    from osmosis_ai.platform.cli.workspace_directory_contract import (
        ensure_workspace_directory_config_path,
        validate_workspace_directory_contract,
    )
    from osmosis_ai.platform.cli.workspace_repo import summarize_local_git_state

    command_label = "`osmosis eval submit`"

    context = require_git_workspace_directory_context()
    workspace_directory = context.workspace_directory
    validate_workspace_directory_contract(workspace_directory)
    config_path = Path(config_path)
    resolved_config_path = (
        config_path if config_path.is_absolute() else workspace_directory / config_path
    )
    ensure_workspace_directory_config_path(
        resolved_config_path,
        workspace_directory,
        config_dir="configs/eval",
        command_label=command_label,
    )

    if not resolved_config_path.exists():
        raise CLIError(f"Config file not found: {resolved_config_path}")

    try:
        with open(resolved_config_path, "rb") as f:
            raw_config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {resolved_config_path}: {e}") from e

    eval_section = raw_config.get("eval", {})
    rollout_section = raw_config.get("rollout", {})
    rollout_env = rollout_section.get("env") if isinstance(rollout_section, dict) else None
    rollout_secret_refs = (
        rollout_section.get("secrets") if isinstance(rollout_section, dict) else None
    )

    # Get commit SHA from local git HEAD
    commit_sha: str | None = None
    state = summarize_local_git_state(workspace_directory)
    if state is not None:
        commit_sha = state.head_sha

    summary_rows: list[tuple[str, str]] = []
    if eval_section.get("rollout"):
        summary_rows.append(("Rollout", eval_section["rollout"]))
    if eval_section.get("dataset"):
        summary_rows.append(("Dataset", eval_section["dataset"]))
    if raw_config.get("llm", {}).get("model"):
        summary_rows.append(("Model", raw_config["llm"]["model"]))
    if commit_sha:
        summary_rows.append(("Commit", commit_sha[:8]))
    if rollout_env:
        env_keys = ", ".join(sorted(rollout_env))
        summary_rows.append((f"Rollout env ({len(rollout_env)})", env_keys))
    if rollout_secret_refs:
        secret_summary = ", ".join(
            f"{env_name}={secret_name}"
            for env_name, secret_name in sorted(rollout_secret_refs.items())
        )
        summary_rows.append(
            (f"Rollout secrets ({len(rollout_secret_refs)})", secret_summary)
        )

    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title="Cloud Eval Run",
    )

    if not yes:
        from osmosis_ai.cli.output import OutputFormat, get_output_context as _get_ctx

        output = _get_ctx()
        if output.format is not OutputFormat.rich or not output.interactive:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            err = CLIError(
                "Use --yes to confirm in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
                details={"prompt": "Submit this cloud eval run?", "summary": dict(summary_rows)},
            )
            if output.format is OutputFormat.json:
                emit_structured_error_to_stderr(err)
                raise typer.Exit(1)
            raise err

        from osmosis_ai.cli.prompts import require_confirmation

        require_confirmation("Submit this cloud eval run?", yes=yes)

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Submitting cloud eval run..."):
        result = client.submit_cloud_eval(
            eval_config=raw_config,
            commit_sha=commit_sha,
            rollout_env=rollout_env or None,
            rollout_secret_refs=rollout_secret_refs or None,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    return OperationResult(
        operation="eval.submit",
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "created_at": result.created_at,
            **({"url": result.platform_url} if result.platform_url else {}),
            **git_result_context(context),
        },
        message=f"Cloud eval run submitted: {result.name}",
        display_next_steps=[
            f"Status: {result.status}",
            f"Check status with: osmosis eval status {result.name}",
            "List all eval runs with: osmosis eval list",
        ],
        next_steps_structured=[
            {"action": "eval_status", "name": result.name},
            {"action": "eval_list"},
            *(
                [{"action": "open_url", "url": result.platform_url}]
                if result.platform_url
                else []
            ),
        ],
    )


@app.command("list")
def eval_list(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of eval runs to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all eval runs."),
) -> Any:
    """List cloud eval runs for the current workspace directory."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_eval_run,
    )
    from osmosis_ai.cli.output.display import (
        created_column_label,
        format_local_date,
    )
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        require_git_workspace_directory_context,
        validate_list_options,
    )
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching eval runs..."):
        if fetch_all:
            eval_runs, total_count = fetch_all_pages(
                lambda lim, off: client.list_eval_runs(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=context.git_identity,
                ),
                items_attr="data",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_eval_runs(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                git_identity=context.git_identity,
            )
            eval_runs = page.data
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    def _format_eval_status(run: Any) -> str:
        from osmosis_ai.platform.api.models import (
            EVAL_RUN_STATUSES_IN_PROGRESS,
            EVAL_RUN_STATUSES_TERMINAL,
        )

        status_info = f"[{run.status}]"
        if run.status in EVAL_RUN_STATUSES_IN_PROGRESS:
            return console.format_styled(status_info, "yellow")
        if run.status == "finished":
            return console.format_styled(status_info, "green")
        if run.status in EVAL_RUN_STATUSES_TERMINAL:
            return console.format_styled(status_info, "red")
        return console.escape(status_info)

    return ListResult(
        title="Cloud Eval Runs",
        items=[serialize_eval_run(r) for r in eval_runs],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="rollout", label="Rollout", ratio=2, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="model", label="Model", no_wrap=True, ratio=2),
            ListColumn(
                key="created_at",
                label=created_column_label(),
                no_wrap=True,
                ratio=1,
            ),
        ],
        display_items=[
            {
                **serialize_eval_run(run),
                "name": run.name,
                "rollout": run.rollout.get("name") if run.rollout else "—",
                "status": _format_eval_status(run),
                "model": run.model.get("name") if run.model else "—",
                "created_at": format_local_date(run.created_at),
            }
            for run in eval_runs
        ],
        display_hints=["Use osmosis eval status <name> for details."],
    )


@app.command("status")
def eval_status(
    name_or_id: str = typer.Argument(..., help="Eval run name or ID."),
) -> Any:
    """Show cloud eval run details and results."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        get_output_context,
    )
    from osmosis_ai.cli.output.display import format_local_datetime
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching eval run..."):
        detail = client.get_eval_run(
            name_or_id,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    eval_run = detail.eval_run
    rows: list[tuple[str, str]] = [
        ("Name", console.escape(eval_run.get("name", "(unnamed)"))),
        ("ID", eval_run.get("id", "")),
        ("Status", eval_run.get("status", "")),
    ]
    if detail.model and detail.model.get("name"):
        rows.append(("Model", console.escape(detail.model["name"])))
    if detail.dataset and detail.dataset.get("name"):
        rows.append(("Dataset", console.escape(detail.dataset["name"])))
    if detail.rollout and detail.rollout.get("name"):
        rows.append(("Rollout", console.escape(detail.rollout["name"])))
    if eval_run.get("creator_name"):
        rows.append(("Creator", console.escape(eval_run["creator_name"])))
    if eval_run.get("created_at"):
        rows.append(("Created", format_local_datetime(eval_run["created_at"])))
    if eval_run.get("started_at"):
        rows.append(("Started", format_local_datetime(eval_run["started_at"])))
    if eval_run.get("completed_at"):
        rows.append(("Completed", format_local_datetime(eval_run["completed_at"])))

    if detail.results:
        if detail.results.get("score") is not None:
            rows.append(("Score", f"{detail.results['score']:.4f}"))
        if detail.results.get("pass_rate") is not None:
            rows.append(("Pass Rate", f"{detail.results['pass_rate']:.1%}"))
        if detail.results.get("total_samples") is not None:
            rows.append(("Samples", str(detail.results["total_samples"])))

    fields = [DetailField(label=label, value=value) for label, value in rows]
    display_hints: list[str] = []

    from osmosis_ai.platform.api.models import EVAL_RUN_STATUSES_IN_PROGRESS

    if eval_run.get("status") in EVAL_RUN_STATUSES_IN_PROGRESS:
        fields.append(
            DetailField(
                label="Note",
                value="Eval is in progress. Results shown are a snapshot.",
            )
        )
        display_hints.append(
            f"Stop with: osmosis eval stop {eval_run.get('name', name_or_id)}"
        )

    return DetailResult(
        title="Cloud Eval Run",
        data={
            "eval_run": eval_run,
            "config": detail.config,
            "results": detail.results,
            "model": detail.model,
            "dataset": detail.dataset,
            "rollout": detail.rollout,
            **git_result_context(context),
        },
        fields=fields,
        display_hints=display_hints,
    )


@app.command("stop")
def eval_stop(
    name_or_id: str = typer.Argument(..., help="Eval run name or ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a cloud eval run."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.output import (
        OperationResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    if not yes:
        output = get_output_context()
        if output.format is not OutputFormat.rich or not output.interactive:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            err = CLIError(
                "Use --yes to confirm in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
                details={
                    "prompt": f'Stop eval run "{name_or_id}"?',
                    "summary": {"name": name_or_id},
                },
            )
            if output.format is OutputFormat.json:
                emit_structured_error_to_stderr(err)
                raise typer.Exit(1)
            raise err

        from osmosis_ai.cli.prompts import require_confirmation

        require_confirmation(f'Stop eval run "{name_or_id}"?', yes=yes)

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Stopping eval run..."):
        client.stop_eval_run(
            name_or_id,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    return OperationResult(
        operation="eval.stop",
        status="success",
        resource={"name": name_or_id, **git_result_context(context)},
        message=f'Eval run "{name_or_id}" stopped.',
    )


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
    _require_eval_local_project()

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
    _require_eval_local_project()

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

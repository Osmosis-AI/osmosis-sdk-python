"""Training run management commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError, not_implemented
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


def _detail_fields(rows: list[tuple[str, str]]) -> list[Any]:
    from osmosis_ai.cli.output import DetailField

    return [DetailField(label=label, value=value) for label, value in rows]


def _require_confirmation(message: str, *, yes: bool) -> None:
    if yes:
        return

    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is not OutputFormat.rich or not output.interactive:
        err = CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )
        if output.format is OutputFormat.json:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            emit_structured_error_to_stderr(err)
            raise typer.Exit(1)
        raise err

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(message, yes=yes)


@app.command("list")
def list_runs(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of runs to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all training runs."),
) -> Any:
    """List training runs in the current platform workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_training_run,
    )
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        fetch_all_pages,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching training runs..."):
        if fetch_all:
            training_runs, total_count = fetch_all_pages(
                lambda lim, off: client.list_training_runs(
                    limit=lim, offset=off, credentials=credentials
                ),
                items_attr="training_runs",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_training_runs(
                limit=effective_limit, offset=0, credentials=credentials
            )
            training_runs = page.training_runs
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Training Runs",
        items=[serialize_training_run(r) for r in training_runs],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        columns=[
            ListColumn(key="name", label="Name"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="model_name", label="Model"),
            ListColumn(key="eval_accuracy", label="Accuracy"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )


@app.command("status")
def status(
    name: str = typer.Argument(..., help="Training run name."),
) -> Any:
    """Show training run details."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        get_output_context,
        serialize_checkpoint,
        serialize_training_run,
    )
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.models import RUN_STATUSES_TERMINAL
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        build_run_detail_rows,
        format_date,
    )

    _, credentials = _require_auth()

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching training run..."):
        run = client.get_training_run(name, credentials=credentials)

    rows = build_run_detail_rows(run)
    if run.examples_processed_count is not None:
        rows.append(("Examples", str(run.examples_processed_count)))
    if run.notes:
        rows.append(("Notes", console.escape(run.notes)))
    if run.hf_status:
        rows.append(("HF Status", run.hf_status))
    if run.started_at:
        rows.append(("Started", format_date(run.started_at)))
    if run.completed_at:
        rows.append(("Completed", format_date(run.completed_at)))

    checkpoints = []

    if run.status in RUN_STATUSES_TERMINAL:
        try:
            with output.status("Fetching checkpoints..."):
                ckpts = client.list_training_run_checkpoints(
                    name, credentials=credentials
                )
        except PlatformAPIError:
            ckpts = None

        if ckpts is not None and ckpts.checkpoints:
            checkpoints = ckpts.checkpoints

    fields = _detail_fields(rows)
    for cp in checkpoints:
        cp_name = cp.checkpoint_name or "(unnamed)"
        fields.append(
            DetailField(
                label="Checkpoint",
                value=(
                    f"{cp_name}  step {cp.checkpoint_step}  [{cp.status}]"
                    f"  {cp.id[:8]}  {format_date(cp.created_at)}"
                ),
            )
        )
    if checkpoints:
        fields.append(
            DetailField(
                label="Deploy",
                value="osmosis deploy <checkpoint-name>",
            )
        )

    data = serialize_training_run(run)
    data.update(
        {
            "examples_processed_count": run.examples_processed_count,
            "notes": run.notes,
            "hf_status": run.hf_status,
            "checkpoints": [serialize_checkpoint(cp) for cp in checkpoints],
        }
    )

    return DetailResult(title="Training Run", data=data, fields=fields)


@app.command("info", hidden=True)
def info(
    name: str = typer.Argument(..., help="Training run name."),
) -> Any:
    """Deprecated alias for `osmosis train status`."""
    return status(name=name)


@app.command("submit")
def submit(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to training config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a new training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.project_contract import (
        ensure_project_config_path,
        resolve_project_root,
        validate_project_contract,
        validate_rollout_backend,
    )
    from osmosis_ai.platform.cli.training_config import load_training_config
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        platform_entity_url,
    )

    project_root = resolve_project_root(config_path)
    validate_project_contract(project_root)
    ensure_project_config_path(
        config_path,
        project_root,
        config_dir="configs/training",
        command_label="`osmosis train submit`",
    )
    config = load_training_config(config_path)
    validate_rollout_backend(
        project_root=project_root,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label="`osmosis train submit`",
    )
    ws_name, credentials = _require_auth()

    rows: list[tuple[str, str]] = [
        ("Rollout", console.escape(config.experiment_rollout)),
        ("Entrypoint", console.escape(config.experiment_entrypoint)),
        ("Model", console.escape(config.experiment_model_path)),
        ("Dataset", console.escape(config.experiment_dataset)),
    ]
    if config.experiment_commit_sha:
        rows.append(("Commit", console.escape(config.experiment_commit_sha)))
    console.table(rows, title="Training Run")

    _require_confirmation("Submit this training run?", yes=yes)

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Submitting training run..."):
        result = client.submit_training_run(
            model_path=config.experiment_model_path,
            dataset=config.experiment_dataset,
            rollout_name=config.experiment_rollout,
            entrypoint=config.experiment_entrypoint,
            commit_sha=config.experiment_commit_sha,
            config=config.to_api_config(),
            credentials=credentials,
        )

    url = platform_entity_url(ws_name, "training", result.id)
    return OperationResult(
        operation="train.submit",
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "created_at": result.created_at,
            "url": url,
            "config": {
                "rollout": config.experiment_rollout,
                "entrypoint": config.experiment_entrypoint,
                "model": config.experiment_model_path,
                "dataset": config.experiment_dataset,
                "commit_sha": config.experiment_commit_sha,
            },
        },
        message=f"Training run submitted: {result.name}",
        display_next_steps=[
            f"Status: {result.status}",
            f"View: {url}",
        ],
        next_steps_structured=[
            {"action": "train_status", "name": result.name},
            {"action": "open_url", "url": url},
        ],
    )


def _safe_name(name: str) -> str:
    """Sanitise a run name for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _default_filename(run_name: str | None, run_id: str) -> str:
    """Build the default JSON filename from run metadata."""
    short_id = run_id[:8]
    safe = _safe_name(run_name) if run_name else None
    return f"{safe}_{short_id}.json" if safe else f"{short_id}.json"


def _resolve_output_path(output: str, run_name: str | None, run_id: str) -> Path:
    """Resolve a user-supplied ``-o`` value into a concrete file path.

    Rules:
    * Trailing ``/`` or existing directory → directory mode (generate default
      filename inside the directory).
    * Has a file extension → use as-is.
    * No extension → auto-append ``.json``.

    Parent directories are created automatically.
    """
    from osmosis_ai.cli.paths import parse_cli_path

    parsed_output = parse_cli_path(output)
    path = parsed_output.path

    try:
        # Directory mode: trailing separator or existing directory
        if parsed_output.has_trailing_separator or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            return path / _default_filename(run_name, run_id)

        # Ensure .json extension (output is always JSON)
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(f"Cannot create output path: {exc}") from exc

    return path


def _resolve_default_output(
    run_name: str | None, run_id: str, *, cwd: Path | None = None
) -> Path:
    """Resolve the default output path under .osmosis/metrics/."""
    if cwd is None:
        cwd = Path.cwd()
    project_toml = cwd / ".osmosis" / "project.toml"
    if not project_toml.is_file():
        raise CLIError(
            "Not in an Osmosis project directory.\n"
            "  Run from a directory created by 'osmosis init',"
            " or use -o to specify an output path."
        )
    metrics_dir = cwd / ".osmosis" / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    return metrics_dir / _default_filename(run_name, run_id)


@app.command("metrics")
def metrics(
    name: str = typer.Argument(..., help="Training run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output path. Non-.json extensions are replaced with .json;"
            " a trailing '/' or existing directory generates a default"
            " filename inside it. (default: .osmosis/metrics/)"
        ),
    ),
) -> Any:
    """Export training run metrics to a JSON file."""
    import json

    from osmosis_ai.cli.metrics_export import build_export_dict
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        OutputFormat,
        get_output_context,
        serialize_training_run,
    )
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.models import RUN_STATUSES_IN_PROGRESS
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        platform_entity_url,
    )

    ws_name, credentials = _require_auth()
    client = OsmosisClient()
    output_ctx = get_output_context()

    with output_ctx.status("Fetching training run..."):
        run = client.get_training_run(name, credentials=credentials)

    if run.status == "pending":
        raise CLIError("Metrics are not yet available for pending training runs.")

    is_in_progress = run.status in RUN_STATUSES_IN_PROGRESS

    # ── Platform URL (no metrics dependency) ─────────────────────
    url = platform_entity_url(ws_name, "training", run.id)

    # ── Fetch metrics (best-effort) ──────────────────────────────
    metrics_data = None
    metrics_error: str | None = None
    try:
        with output_ctx.status("Fetching metrics..."):
            metrics_data = client.get_training_run_metrics(
                run.id, credentials=credentials
            )
    except (PlatformAPIError, KeyError) as exc:
        metrics_error = str(exc) or "Could not fetch metrics data."

    # ── Summary table ─────────────────────────────────────────────
    rows: list[tuple[str, str]] = []
    if run.name:
        rows.append(("Name", console.escape(run.name)))
    rows.append(("Status", run.status))
    if run.model_name:
        rows.append(("Model", console.escape(run.model_name)))

    if metrics_data is not None:
        if metrics_data.overview.duration_formatted:
            rows.append(("Duration", metrics_data.overview.duration_formatted))
        if metrics_data.overview.reward is not None:
            rows.append(("Final Reward", f"{metrics_data.overview.reward:.4f}"))
        if metrics_data.overview.reward_delta is not None:
            rows.append(("Reward Delta", f"{metrics_data.overview.reward_delta:+.4f}"))
        if metrics_data.overview.examples_processed_count is not None:
            rows.append(
                ("Examples", f"{metrics_data.overview.examples_processed_count:,}")
            )
        total_steps = max(
            (dp.step for m in metrics_data.metrics for dp in m.data_points),
            default=0,
        )
        if total_steps:
            rows.append(("Steps", f"{total_steps:,}"))

    fields = _detail_fields(rows)
    fields.insert(0, DetailField(label="View", value=url))

    # ── Metric trends ─────────────────────────────────────────────
    if metrics_data is not None and metrics_data.metrics:
        from osmosis_ai.cli.metrics_graph import (
            render_metric_trends,
            should_render_metric_trends,
        )

        if output_ctx.format is OutputFormat.rich and should_render_metric_trends(
            is_tty=console.is_tty,
            terminal_width=console.width,
            metrics=metrics_data.metrics,
        ):
            trends = render_metric_trends(
                metrics_data.metrics, terminal_width=console.width
            )
            if trends:
                fields.append(DetailField(label="Metric Trends", value=trends))
    elif metrics_data is not None:
        fields.append(DetailField(label="Metrics", value="No metric data found."))

    if metrics_error is not None:
        fields.append(
            DetailField(label="Metrics", value="Could not fetch metrics data.")
        )

    if is_in_progress:
        fields.append(
            DetailField(
                label="Note",
                value="Training is in progress. Metrics shown are a snapshot.",
            )
        )

    # ── Save to file (best-effort) ────────────────────────────────
    export: dict[str, Any] | None = None
    output_path: str | None = None
    save_warning: str | None = None
    if metrics_data is not None:
        export = build_export_dict(run, metrics_data)
        should_write_file = output is not None or output_ctx.format is OutputFormat.rich
        if should_write_file:
            try:
                out_path = (
                    _resolve_output_path(output, run.name, run.id)
                    if output
                    else _resolve_default_output(run.name, run.id)
                )
                out_path.write_text(
                    json.dumps(export, indent=2, ensure_ascii=False) + "\n"
                )
                output_path = str(out_path)
                fields.append(DetailField(label="Saved", value=output_path))
            except (CLIError, OSError) as exc:
                save_warning = f"Could not save metrics: {exc}"
                fields.append(DetailField(label="Warning", value=save_warning))

    return DetailResult(
        title="Training Run Metrics",
        data={
            "training_run": serialize_training_run(run),
            "platform_url": url,
            "in_progress": is_in_progress,
            "metrics_available": metrics_data is not None,
            "metrics_error": metrics_error,
            "metrics": export,
            "output_path": output_path,
            "save_warning": save_warning,
        },
        fields=fields,
    )


@app.command("traces")
def traces() -> None:
    """Show training run traces."""
    not_implemented("train", "traces")


@app.command("stop")
def stop(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    _require_confirmation(f'Stop training run "{name}"?', yes=yes)

    output = get_output_context()
    with output.status("Stopping training run..."):
        client.stop_training_run(name, credentials=credentials)
    return OperationResult(
        operation="train.stop",
        status="success",
        resource={"name": name},
        message=f'Training run "{name}" stopped.',
    )


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Delete a training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.utils import _require_auth

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    _require_confirmation(
        f'Delete training run "{name}"? This cannot be undone.', yes=yes
    )

    output = get_output_context()
    with output.status("Deleting training run..."):
        client.delete_training_run(name, credentials=credentials)
    return OperationResult(
        operation="train.delete",
        status="success",
        resource={"name": name},
        message=f'Training run "{name}" deleted.',
    )

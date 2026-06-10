"""Handlers for `osmosis train` subcommands (mirror of platform/cli/eval.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.metrics_export import (
    build_export_dict,
    resolve_default_metrics_output,
    resolve_metrics_output_path,
)
from osmosis_ai.cli.output import (
    DetailField,
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    OperationResult,
    OutputFormat,
    detail_fields,
    get_output_context,
    serialize_checkpoint,
    serialize_training_run,
)
from osmosis_ai.cli.output.display import (
    format_duration_ms,
    format_local_date,
    format_local_datetime,
)
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    RUN_STATUSES_PENDING,
    RUN_STATUSES_TERMINAL,
    SubmitRunResult,
)
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.shared_submit import CloudSubmitSpec, run_cloud_submit
from osmosis_ai.platform.cli.training_config import (
    TrainSubmitConfig,
    load_train_submit_config,
    validate_train_submit_context_paths,
)
from osmosis_ai.platform.cli.utils import (
    build_run_detail_rows,
    format_env_config,
    format_progress,
    format_run_status,
    format_secret_scopes,
    jsonish,
    kv_section,
    make_progress,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context


def _format_train_reward(run: Any) -> str:
    if run.reward is None:
        return "—"
    return f"{run.reward:.2f}"


def _format_train_config(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None
    parts: list[str] = []
    for key in sorted(config):
        if key == "model_path":
            continue
        value = config[key]
        if value is None:
            continue
        if isinstance(value, dict):
            for sub in sorted(value):
                sub_value = value[sub]
                if sub_value is not None:
                    parts.append(f"{key}.{sub}={jsonish(sub_value)}")
        else:
            parts.append(f"{key}={jsonish(value)}")
    return ", ".join(parts) if parts else None


def _insert_after(
    rows: list[tuple[str, str]],
    anchors: tuple[str, ...],
    item: tuple[str, str],
) -> None:
    """Insert ``item`` right after the first row matching any of ``anchors``.

    Falls back to appending when none of the anchor labels are present.
    """
    for anchor in anchors:
        for index, (label, _value) in enumerate(rows):
            if label == anchor:
                rows.insert(index + 1, item)
                return
    rows.append(item)


def _train_summary(run: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if run.reward is not None:
        summary["reward"] = run.reward
    progress = make_progress(run.current_step, run.total_steps, "steps")
    if progress is not None:
        summary["progress"] = progress
    return summary


def _submit_training(
    client: OsmosisClient,
    config: TrainSubmitConfig,
    credentials: Any,
    git_identity: str,
) -> SubmitRunResult:
    return client.submit_training_run(
        experiment_config=config.experiment_config,
        training_config=config.training_config or None,
        sampling_config=config.sampling_config or None,
        checkpoints_config=config.checkpoints_config or None,
        advanced_config=config.advanced_config or None,
        env_config=config.env or None,
        secrets=config.secrets or None,
        credentials=credentials,
        git_identity=git_identity,
    )


def _train_next_steps(
    result: SubmitRunResult, config: TrainSubmitConfig
) -> tuple[list[str], list[dict[str, Any]]]:
    display = [
        f"Status: {result.status}",
        f"Rollout: {config.experiment_rollout}",
        f"Model: {config.experiment_model_path}",
        f"Dataset: {config.experiment_dataset}",
        (
            f"View: {result.platform_url}"
            if result.platform_url
            else f"Check status with: osmosis train info {result.name}"
        ),
    ]
    structured: list[dict[str, Any]] = [
        {"action": "train_info", "name": result.name},
        {"action": "train_list"},
    ]
    if result.platform_url:
        structured.append({"action": "open_url", "url": result.platform_url})
    return display, structured


_TRAIN_SUBMIT_SPEC: CloudSubmitSpec[TrainSubmitConfig] = CloudSubmitSpec(
    config_dir="configs/training",
    command_label="`osmosis train submit`",
    table_title="Training Run",
    confirm_prompt="Submit this training run?",
    status_message="Submitting training run...",
    operation="train.submit",
    success_message_format="Training run submitted: {name}",
    load_config=load_train_submit_config,
    validate_context=validate_train_submit_context_paths,
    submit=_submit_training,
    build_next_steps=_train_next_steps,
)


def submit(config_path: Path, *, yes: bool) -> OperationResult:
    """Submit a new training run."""
    return run_cloud_submit(config_path, yes=yes, spec=_TRAIN_SUBMIT_SPEC)


def list_training_runs(*, limit: int, all_: bool) -> ListResult:
    """List training runs for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching training runs..."):
        training_runs, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_training_runs(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=context.git_identity,
            ),
            items_attr="training_runs",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Training Runs",
        items=[
            {**serialize_training_run(r), "summary": _train_summary(r)}
            for r in training_runs
        ],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="model_name", label="Base Model", ratio=2, overflow="fold"),
            ListColumn(key="rollout_name", label="Rollout", ratio=2, overflow="fold"),
            ListColumn(key="reward", label="Reward", no_wrap=True, ratio=1),
            ListColumn(key="created_at", label="Submitted", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Submitted By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_training_run(run),
                "name": run.name or "(unnamed)",
                "status": format_run_status(run),
                "model_name": run.model_name or "—",
                "rollout_name": run.rollout_name or "—",
                "reward": _format_train_reward(run),
                "created_at": format_local_date(run.created_at),
                "creator_name": run.creator_name or "—",
            }
            for run in training_runs
        ],
        display_hints=["Use osmosis train info <name> for details."],
    )


def info(name: str, *, output: str | None) -> DetailResult:
    """Show training run details, checkpoints, and metrics."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output_ctx = get_output_context()
    with output_ctx.status("Fetching training run..."):
        run = client.get_training_run(
            name,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    rows = build_run_detail_rows(run)
    summary = _train_summary(run)
    if run.status in RUN_STATUSES_PENDING:
        rows.insert(3, ("Progress", "Waiting to start..."))
    else:
        progress = format_progress(summary.get("progress"))
        if progress:
            rows.insert(3, ("Progress", progress))
    if run.started_at:
        rows.append(("Started", format_local_datetime(run.started_at)))
    if run.completed_at:
        rows.append(("Completed", format_local_datetime(run.completed_at)))
    if run.notes:
        rows.append(("Notes", console.escape(run.notes)))

    checkpoints: list[Any] = []
    sections: list[DetailSection] = []
    display_hints: list[str] = []

    config_rows: list[tuple[str, str]] = []
    if run.entrypoint:
        config_rows.append(("Entrypoint", run.entrypoint))
    train_config = _format_train_config(run.config)
    if train_config:
        config_rows.append(("Config", train_config))
    if run.commit_sha:
        config_rows.append(("Commit", run.commit_sha[:7]))
    secret_scopes = format_secret_scopes(run.resolved_secret_scopes)
    if secret_scopes:
        config_rows.append(("Secrets", secret_scopes))
    env_config = format_env_config(run.env_config)
    if env_config:
        config_rows.append(("Environment Variables", env_config))
    config_section = kv_section("Configuration", config_rows)
    if config_section is not None:
        sections.append(config_section)

    if run.status in RUN_STATUSES_TERMINAL:
        try:
            with output_ctx.status("Fetching checkpoints..."):
                ckpts = client.list_training_run_checkpoints(
                    name,
                    credentials=credentials,
                    git_identity=context.git_identity,
                )
        except PlatformAPIError:
            ckpts = None

        if ckpts is not None:
            fetched_checkpoints = getattr(ckpts, "checkpoints", None)
            if isinstance(fetched_checkpoints, list) and fetched_checkpoints:
                checkpoints = fetched_checkpoints

    metrics_data = None
    metrics_error: str | None = None
    is_in_progress = (
        run.status not in RUN_STATUSES_TERMINAL
        and run.status not in RUN_STATUSES_PENDING
    )
    if run.status in RUN_STATUSES_PENDING:
        metrics_error = (
            "Metrics are not yet available for pending or queued training runs."
        )
    else:
        try:
            with output_ctx.status("Fetching metrics..."):
                metrics_data = client.get_training_run_metrics(
                    run.id,
                    credentials=credentials,
                    git_identity=context.git_identity,
                )
        except (PlatformAPIError, KeyError) as exc:
            metrics_error = str(exc) or "Could not fetch metrics data."

    export: dict[str, Any] | None = None
    output_path: str | None = None
    save_warning: str | None = None
    if metrics_data is not None:
        export = build_export_dict(run, metrics_data)
        if "progress" not in summary and metrics_data.overview.total_steps is not None:
            latest = metrics_data.overview.latest_step or 0
            total = metrics_data.overview.total_steps
            rows.insert(3, ("Progress", f"{latest} / {total} rollout steps"))
        if metrics_data.overview.duration_ms is not None:
            _insert_after(
                rows,
                ("Progress", "Status"),
                ("Duration", format_duration_ms(metrics_data.overview.duration_ms)),
            )

    if run.examples_processed_count is not None:
        _insert_after(
            rows,
            ("Duration", "Progress", "Status"),
            ("Examples Processed", f"{run.examples_processed_count:,}"),
        )

    fields = detail_fields(rows)
    if run.platform_url:
        display_hints.append(f"View: {run.platform_url}")
    if checkpoints:
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=True, header_style="bold", expand=False)
        table.add_column("Checkpoint", overflow="fold")
        table.add_column("Step", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("ID", no_wrap=True)
        plain_lines = []
        for cp in checkpoints:
            cp_name = cp.checkpoint_name or "(unnamed)"
            table.add_row(
                Text(cp_name),
                str(cp.checkpoint_step),
                cp.status,
                cp.id[:8],
            )
            plain_lines.append(
                f"Checkpoint: {cp_name} step {cp.checkpoint_step} "
                f"[{cp.status}] {cp.id[:8]}"
            )
        sections.append(DetailSection(rich=table, plain_lines=plain_lines))
        display_hints.append("Deploy with: osmosis model deploy <lora-model-name>")

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
                sections.append(DetailSection(rich=trends))
    elif metrics_data is not None:
        fields.append(DetailField(label="Metrics", value="No metric data found."))

    if metrics_error is not None:
        fields.append(DetailField(label="Metrics", value=metrics_error))

    if is_in_progress and metrics_data is not None:
        fields.append(
            DetailField(
                label="Note",
                value="Training is in progress. Metrics shown are a snapshot.",
            )
        )

    if metrics_data is not None:
        should_write_file = output is not None or output_ctx.format is OutputFormat.rich
        if should_write_file:
            try:
                out_path = (
                    resolve_metrics_output_path(output, run.name, run.id)
                    if output
                    else resolve_default_metrics_output(
                        run.name,
                        run.id,
                        workspace_directory=context.workspace_directory,
                    )
                )
                out_path.write_text(
                    json.dumps(export, indent=2, ensure_ascii=False) + "\n"
                )
                output_path = str(out_path)
                display_hints.append(f"Saved metrics to {output_path}")
            except (CLIError, OSError) as exc:
                save_warning = f"Could not save metrics: {exc}"
                display_hints.append(save_warning)

    return DetailResult(
        title="Training Run Info",
        data={
            "training_run": {
                **serialize_training_run(run),
                "examples_processed_count": run.examples_processed_count,
                "notes": run.notes,
            },
            "summary": summary,
            **({"platform_url": run.platform_url} if run.platform_url else {}),
            "checkpoints": [serialize_checkpoint(cp) for cp in checkpoints],
            "in_progress": is_in_progress,
            "metrics_available": metrics_data is not None,
            "metrics_error": metrics_error,
            "metrics": export,
            "output_path": output_path,
            "save_warning": save_warning,
            **git_result_context(context),
        },
        fields=fields,
        sections=sections,
        display_hints=display_hints,
    )


def stop(name: str, *, yes: bool) -> OperationResult:
    """Stop a training run."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials

    require_confirmation(
        f'Stop training run "{name}"?',
        yes=yes,
        default=False,
        summary=[("Name", name)],
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Stopping training run..."):
        client.stop_training_run(
            name,
            credentials=credentials,
            git_identity=context.git_identity,
        )
    return OperationResult(
        operation="train.stop",
        status="success",
        resource={"name": name, **git_result_context(context)},
        message=f'Training run "{name}" stopped.',
    )


__all__ = ["info", "list_training_runs", "stop", "submit"]

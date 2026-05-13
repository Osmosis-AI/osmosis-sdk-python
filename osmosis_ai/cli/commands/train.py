"""Training run management commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


def _workspace_result_context(workspace: Any) -> dict[str, Any]:
    return {
        "workspace": {"id": workspace.workspace_id, "name": workspace.workspace_name},
        "project_root": str(workspace.project_root),
    }


def _detail_fields(rows: list[tuple[str, str]]) -> list[Any]:
    from osmosis_ai.cli.output import DetailField

    return [DetailField(label=label, value=value) for label, value in rows]


def _require_confirmation(
    message: str,
    *,
    yes: bool,
    summary: list[tuple[str, str]] | None = None,
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Confirm a destructive action, surfacing context in non-interactive modes.

    In rich+interactive mode, falls back to the questionary prompt. In JSON
    and plain modes (or any non-interactive shell), raises an
    ``INTERACTIVE_REQUIRED`` error that carries the prompt question, the
    summary of what would be acted on, and any notes/warnings — so AI
    agents and CI scripts can see exactly what they're being asked to
    confirm without first having to retry with ``--yes``.
    """
    if yes:
        return

    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is not OutputFormat.rich or not output.interactive:
        details: dict[str, Any] = {"prompt": message}
        if summary:
            details["summary"] = {label: value for label, value in summary}
        if notes:
            details["notes"] = list(notes)
        if warnings:
            details["warnings"] = list(warnings)

        if output.format is OutputFormat.plain:
            import sys

            lines: list[str] = [f"Confirmation required: {message}"]
            if summary:
                for label, value in summary:
                    lines.append(f"  {label}: {value}")
            if notes:
                lines.append("Notes:")
                for note in notes:
                    lines.append(f"  - {note}")
            if warnings:
                lines.append("Warnings:")
                for warning in warnings:
                    lines.append(f"  - {warning}")
            sys.stderr.write("\n".join(lines) + "\n")
            sys.stderr.flush()

        err = CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
            details=details,
        )
        if output.format is OutputFormat.json:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            emit_structured_error_to_stderr(err)
            raise typer.Exit(1)
        raise err

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(message, yes=yes)


def _print_remote_fetch_notice(
    project_root: Path,
    *,
    pinned_commit_sha: str | None,
) -> tuple[list[str], list[str]]:
    """Remind the user that submit pulls *code* from the connected Git remote
    while reading *config values* from the local TOML file.

    The platform clones the workspace's connected Git repository (or
    fetches a pinned commit) before training, so local *code* changes
    that haven't been pushed will silently be ignored. The config TOML
    passed to ``osmosis train submit``, by contrast, is read from disk
    and its values are sent verbatim in the submit payload — local
    edits to the config take effect immediately, even if they are
    uncommitted.

    Returns ``(notes, warnings)`` as plain-text lists so callers can
    surface the same context in non-rich modes (e.g. the JSON error
    envelope when ``--yes`` is missing). The Rich panel is rendered
    only when the output format is Rich.
    """
    from osmosis_ai.cli.output import OutputFormat, get_output_context
    from osmosis_ai.platform.cli.workspace_repo import summarize_local_git_state

    state = summarize_local_git_state(project_root)

    warnings: list[str] = []
    if state is not None:
        if state.is_dirty:
            warnings.append(
                "Uncommitted changes detected — code edits won't be picked up "
                "(only the config file above is read locally)."
            )
        if state.has_upstream and state.ahead > 0:
            commits_word = "commit" if state.ahead == 1 else "commits"
            warnings.append(
                f"{state.ahead} unpushed {commits_word} ahead of upstream — "
                "push code before submitting."
            )
        elif state.branch is not None and not state.has_upstream:
            warnings.append(
                f"Branch '{state.branch}' has no upstream — "
                "push code and set tracking before submitting."
            )

    notes: list[str] = []
    if pinned_commit_sha:
        notes.append(
            f"Osmosis will fetch commit {pinned_commit_sha} from the "
            "workspace's connected Git repository for training code."
        )
        notes.append("Make sure that commit is already pushed to the remote.")
    else:
        notes.append(
            "Osmosis will fetch the latest training code from the workspace's "
            "connected Git repository."
        )
        if state is not None and state.branch and state.head_sha:
            notes.append(f"Local branch: {state.branch} @ {state.head_sha[:8]}")
        notes.append("Make sure your code changes are committed and pushed.")
    notes.append(
        "Config values come from your local TOML file and are submitted "
        "as-is — uncommitted edits to the config still apply."
    )

    if get_output_context().format is OutputFormat.rich:
        body_lines: list[str] = []
        if pinned_commit_sha:
            body_lines.append(
                f"Osmosis will fetch commit [bold]{console.escape(pinned_commit_sha)}[/bold] "
                "from the workspace's connected Git repository for training code."
            )
            body_lines.append("Make sure that commit is already pushed to the remote.")
        else:
            body_lines.append(
                "Osmosis will fetch the latest training code from the workspace's "
                "connected Git repository."
            )
            if state is not None and state.branch and state.head_sha:
                body_lines.append(
                    f"Local: [bold]{console.escape(state.branch)}[/bold] @ "
                    f"[dim]{console.escape(state.head_sha[:8])}[/dim]"
                )
            body_lines.append("Make sure your code changes are committed and pushed.")

        body_lines.append("")
        body_lines.append(
            "[dim]Config values above come from your local TOML file and are "
            "submitted as-is — uncommitted edits to the config still apply.[/dim]"
        )

        if warnings:
            body_lines.append("")
            for warning in warnings:
                body_lines.append(f"[yellow]• {console.escape(warning)}[/yellow]")

        style = "yellow" if warnings else "blue"
        title = "Push before submitting" if warnings else "Before you submit"
        console.panel(title, "\n".join(body_lines), style=style)

    return notes, warnings


@app.command("list")
def list_runs(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of runs to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all training runs."),
) -> Any:
    """List training runs in the linked project workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_training_run,
    )
    from osmosis_ai.cli.output.display import (
        created_column_label,
        format_local_date,
        format_reward,
    )
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        format_run_status,
        require_workspace_context,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    workspace = require_workspace_context()
    credentials = workspace.credentials

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching training runs..."):
        if fetch_all:
            training_runs, total_count = fetch_all_pages(
                lambda lim, off: client.list_training_runs(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    workspace_id=workspace.workspace_id,
                ),
                items_attr="training_runs",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_training_runs(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                workspace_id=workspace.workspace_id,
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
        extra=_workspace_result_context(workspace),
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="reward", label="Reward", no_wrap=True, ratio=1),
            ListColumn(
                key="created_at",
                label=created_column_label(),
                no_wrap=True,
                ratio=1,
            ),
        ],
        display_items=[
            {
                **serialize_training_run(run),
                "name": run.name or "(unnamed)",
                "status": format_run_status(run),
                "reward": format_reward(run.reward),
                "created_at": format_local_date(run.created_at),
            }
            for run in training_runs
        ],
        display_hints=[
            "Use osmosis train status <name> or osmosis train metrics <name> for details."
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
        build_run_detail_rows,
        format_date,
        require_workspace_context,
    )

    workspace = require_workspace_context()
    credentials = workspace.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching training run..."):
        run = client.get_training_run(
            name,
            credentials=credentials,
            workspace_id=workspace.workspace_id,
        )

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
                    name,
                    credentials=credentials,
                    workspace_id=workspace.workspace_id,
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
            **_workspace_result_context(workspace),
        }
    )

    return DetailResult(title="Training Run", data=data, fields=fields)


@app.command("submit")
def submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to training config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a new training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.project_contract import (
        ensure_project_config_path,
        resolve_project_root_from_cwd,
        validate_project_contract,
        validate_rollout_backend,
    )
    from osmosis_ai.platform.cli.training_config import (
        load_training_config,
        validate_training_context_paths,
    )
    from osmosis_ai.platform.cli.utils import (
        platform_entity_url,
        require_workspace_context,
    )
    from osmosis_ai.platform.cli.workspace_repo import (
        require_git_top_level,
        validate_workspace_repo,
    )

    command_label = "`osmosis train submit`"

    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)
    config_path = Path(config_path)
    resolved_config_path = (
        config_path if config_path.is_absolute() else project_root / config_path
    )
    ensure_project_config_path(
        resolved_config_path,
        project_root,
        config_dir="configs/training",
        command_label=command_label,
    )
    config = load_training_config(resolved_config_path)
    validate_training_context_paths(config, project_root)
    validate_rollout_backend(
        project_root=project_root,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=command_label,
    )
    workspace = require_workspace_context()
    credentials = workspace.credentials
    require_git_top_level(project_root, command_label)
    validate_workspace_repo(
        project_root=project_root,
        workspace_id=workspace.workspace_id,
        workspace_name=workspace.workspace_name,
        credentials=credentials,
        command_label=command_label,
    )

    summary_rows: list[tuple[str, str]] = [
        ("Rollout", config.experiment_rollout),
        ("Entrypoint", config.experiment_entrypoint),
        ("Model", config.experiment_model_path),
        ("Dataset", config.experiment_dataset),
    ]
    if config.experiment_commit_sha:
        summary_rows.append(("Commit", config.experiment_commit_sha))
    if config.rollout_env:
        env_keys = ", ".join(sorted(config.rollout_env))
        summary_rows.append((f"Rollout env ({len(config.rollout_env)})", env_keys))
    if config.rollout_secret_refs:
        # Show env-var name → secret-record name (no secret values, ever).
        secret_summary = ", ".join(
            f"{env_name}={secret_name}"
            for env_name, secret_name in sorted(config.rollout_secret_refs.items())
        )
        summary_rows.append(
            (
                f"Rollout secrets ({len(config.rollout_secret_refs)})",
                secret_summary,
            )
        )
    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title="Training Run",
    )

    notes, warnings = _print_remote_fetch_notice(
        project_root,
        pinned_commit_sha=config.experiment_commit_sha,
    )

    _require_confirmation(
        "Submit this training run?",
        yes=yes,
        summary=summary_rows,
        notes=notes,
        warnings=warnings,
    )

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
            rollout_env=config.rollout_env or None,
            rollout_secret_refs=config.rollout_secret_refs or None,
            credentials=credentials,
            workspace_id=workspace.workspace_id,
        )

    url = platform_entity_url(workspace.workspace_name, "training", result.id)
    return OperationResult(
        operation="train.submit",
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "created_at": result.created_at,
            "url": url,
            **_workspace_result_context(workspace),
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
            "  Run from an existing or cloned Osmosis project repo linked with "
            "'osmosis project link', or use -o to specify an output path."
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
        platform_entity_url,
        require_workspace_context,
    )

    workspace = require_workspace_context()
    credentials = workspace.credentials
    client = OsmosisClient()
    output_ctx = get_output_context()

    with output_ctx.status("Fetching training run..."):
        run = client.get_training_run(
            name,
            credentials=credentials,
            workspace_id=workspace.workspace_id,
        )

    if run.status == "pending":
        raise CLIError("Metrics are not yet available for pending training runs.")

    is_in_progress = run.status in RUN_STATUSES_IN_PROGRESS

    # ── Platform URL (no metrics dependency) ─────────────────────
    url = platform_entity_url(workspace.workspace_name, "training", run.id)

    # ── Fetch metrics (best-effort) ──────────────────────────────
    metrics_data = None
    metrics_error: str | None = None
    try:
        with output_ctx.status("Fetching metrics..."):
            metrics_data = client.get_training_run_metrics(
                run.id,
                credentials=credentials,
                workspace_id=workspace.workspace_id,
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
                    else _resolve_default_output(
                        run.name,
                        run.id,
                        cwd=workspace.project_root,
                    )
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
            **_workspace_result_context(workspace),
        },
        fields=fields,
    )


@app.command("stop")
def stop(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.platform.cli.utils import require_workspace_context

    workspace = require_workspace_context()
    credentials = workspace.credentials

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    _require_confirmation(
        f'Stop training run "{name}"?',
        yes=yes,
        summary=[("Name", name)],
    )

    output = get_output_context()
    with output.status("Stopping training run..."):
        client.stop_training_run(
            name,
            credentials=credentials,
            workspace_id=workspace.workspace_id,
        )
    return OperationResult(
        operation="train.stop",
        status="success",
        resource={"name": name, **_workspace_result_context(workspace)},
        message=f'Training run "{name}" stopped.',
    )

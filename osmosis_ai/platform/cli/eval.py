"""Handler for `osmosis eval` remote subcommands (submit/list/status/stop)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.output import (
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    OperationResult,
    get_output_context,
    serialize_eval_run,
)
from osmosis_ai.cli.output.display import (
    created_column_label,
    format_local_date,
    format_local_datetime,
)
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    EVAL_RUN_STATUSES_IN_PROGRESS,
    EVAL_RUN_STATUSES_TERMINAL,
)
from osmosis_ai.platform.cli.eval_config import (
    EvalSubmitConfig,
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)
from osmosis_ai.platform.cli.utils import (
    fetch_all_pages,
    print_remote_fetch_notice,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_rollout_backend,
    validate_workspace_directory_contract,
)


def _build_submit_summary(
    config: EvalSubmitConfig,
    *,
    env: dict[str, str],
    secret_refs: dict[str, str],
) -> list[tuple[str, str]]:
    """Build the confirmation-table rows shown before submitting."""
    rows: list[tuple[str, str]] = [
        ("Rollout", config.experiment_rollout),
        ("Entrypoint", config.experiment_entrypoint),
        ("Model", config.llm_model_path),
        ("Dataset", config.experiment_dataset),
    ]
    if config.experiment_commit_sha:
        rows.append(("Commit", config.experiment_commit_sha))
    if env:
        env_keys = ", ".join(sorted(env))
        rows.append((f"Rollout env ({len(env)})", env_keys))
    if secret_refs:
        secret_summary = ", ".join(
            f"{env_name}={secret_name}"
            for env_name, secret_name in sorted(secret_refs.items())
        )
        rows.append((f"Rollout secrets ({len(secret_refs)})", secret_summary))
    return rows


def _format_eval_status(run: Any) -> str:
    """Format an eval run status string with Rich color styling."""
    status_info = f"[{run.status}]"
    if run.status in EVAL_RUN_STATUSES_IN_PROGRESS:
        return console.format_styled(status_info, "yellow")
    if run.status == "finished":
        return console.format_styled(status_info, "green")
    if run.status in EVAL_RUN_STATUSES_TERMINAL:
        return console.format_styled(status_info, "red")
    return console.escape(status_info)


def submit(config_path: Path, *, yes: bool) -> OperationResult:
    """Submit a cloud eval run."""
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

    config = load_eval_submit_config(resolved_config_path)
    validate_eval_submit_context_paths(config, workspace_directory)
    validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=command_label,
    )
    env = config.env
    secret_refs = config.secrets

    summary_rows = _build_submit_summary(
        config,
        env=env,
        secret_refs=secret_refs,
    )
    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title="Cloud Eval Run",
    )

    notes, warnings = print_remote_fetch_notice(
        workspace_directory,
        pinned_commit_sha=config.experiment_commit_sha,
    )

    require_confirmation(
        "Submit this cloud eval run?",
        yes=yes,
        summary=summary_rows,
        notes=notes,
        warnings=warnings,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Submitting cloud eval run..."):
        result = client.submit_cloud_eval(
            experiment_config=config.experiment_config,
            llm_config=config.llm_config,
            evaluation_config=config.evaluation_config or None,
            advanced_config=config.advanced_config or None,
            env_config=env or None,
            secret_refs_config=secret_refs or None,
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
            "config": {
                "rollout": config.experiment_rollout,
                "entrypoint": config.experiment_entrypoint,
                "model": config.llm_model_path,
                "dataset": config.experiment_dataset,
                "commit_sha": config.experiment_commit_sha,
            },
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


def list_eval_runs(*, limit: int, all_: bool) -> ListResult:
    """List cloud eval runs for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching eval runs..."):
        if fetch_all:
            eval_runs, total_count = fetch_all_pages(
                lambda lim, off: client.list_eval_runs(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=context.git_identity,
                ),
                items_attr="eval_runs",
            )
            has_more = False
            next_offset: int | None = None
        else:
            page = client.list_eval_runs(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                git_identity=context.git_identity,
            )
            eval_runs = page.eval_runs
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

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


def status(name_or_id: str) -> DetailResult:
    """Show cloud eval run details and results."""
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

    if eval_run.get("status") in EVAL_RUN_STATUSES_IN_PROGRESS:
        fields.append(
            DetailField(
                label="Note",
                value="Eval is in progress. Results shown are a snapshot.",
            )
        )
        display_hints.append(
            f"Stop with: osmosis eval stop {eval_run.get('name') or name_or_id}"
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


def stop(name_or_id: str, *, yes: bool) -> OperationResult:
    """Stop a cloud eval run."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials

    require_confirmation(
        f'Stop eval run "{name_or_id}"?',
        yes=yes,
        summary=[("Name", name_or_id)],
    )

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

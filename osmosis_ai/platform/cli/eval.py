"""Handler for `osmosis eval` remote subcommands (submit/list/info/stop)."""

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
    detail_fields,
    get_output_context,
    serialize_eval_run,
)
from osmosis_ai.cli.output.display import (
    format_local_date,
    format_local_datetime,
)
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    EVAL_RUN_STATUSES_IN_PROGRESS,
    EVAL_RUN_STATUSES_TERMINAL,
    SubmitRunResult,
)
from osmosis_ai.platform.cli.eval_config import (
    EvalSubmitConfig,
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)
from osmosis_ai.platform.cli.shared_submit import CloudSubmitSpec, run_cloud_submit
from osmosis_ai.platform.cli.utils import (
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context


def _format_eval_status(run: Any) -> str:
    """Format an evaluation run status string with Rich color styling."""
    status_info = f"[{run.status}]"
    if run.status in EVAL_RUN_STATUSES_IN_PROGRESS:
        return console.format_styled(status_info, "yellow")
    if run.status == "finished":
        return console.format_styled(status_info, "green")
    if run.status in EVAL_RUN_STATUSES_TERMINAL:
        return console.format_styled(status_info, "red")
    return console.escape(status_info)


def _submit_eval(
    client: OsmosisClient,
    config: EvalSubmitConfig,
    credentials: Any,
    git_identity: str,
) -> SubmitRunResult:
    return client.submit_evaluation_run(
        experiment_config=config.experiment_config,
        evaluation_config=config.evaluation_config or None,
        advanced_config=config.advanced_config or None,
        env_config=config.env or None,
        secrets=config.secrets or None,
        credentials=credentials,
        git_identity=git_identity,
    )


def _eval_next_steps(
    result: SubmitRunResult, _config: EvalSubmitConfig
) -> tuple[list[str], list[dict[str, Any]]]:
    display = [
        f"Status: {result.status}",
        f"Rollout: {_config.experiment_rollout}",
        f"Model: {_config.experiment_model_path}",
        f"Dataset: {_config.experiment_dataset}",
        (
            f"View: {result.platform_url}"
            if result.platform_url
            else f"Check status with: osmosis eval info {result.name}"
        ),
    ]
    structured: list[dict[str, Any]] = [
        {"action": "eval_info", "name": result.name},
        {"action": "eval_list"},
    ]
    if result.platform_url:
        structured.append({"action": "open_url", "url": result.platform_url})
    return display, structured


_EVAL_SUBMIT_SPEC: CloudSubmitSpec[EvalSubmitConfig] = CloudSubmitSpec(
    config_dir="configs/eval",
    command_label="`osmosis eval submit`",
    table_title="Evaluation Run",
    confirm_prompt="Submit this evaluation run?",
    status_message="Submitting evaluation run...",
    operation="eval.submit",
    success_message_format="Evaluation run submitted: {name}",
    load_config=load_eval_submit_config,
    validate_context=validate_eval_submit_context_paths,
    submit=_submit_eval,
    build_next_steps=_eval_next_steps,
)


def submit(config_path: Path, *, yes: bool) -> OperationResult:
    """Submit an evaluation run."""
    return run_cloud_submit(config_path, yes=yes, spec=_EVAL_SUBMIT_SPEC)


def list_eval_runs(*, limit: int, all_: bool) -> ListResult:
    """List evaluation runs for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching evaluation runs..."):
        eval_runs, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_eval_runs(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=context.git_identity,
            ),
            items_attr="eval_runs",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Evaluation Runs",
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
            ListColumn(key="created_at", label="Submitted", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Submitted By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_eval_run(run),
                "name": run.name,
                "rollout": run.rollout.get("name") if run.rollout else "—",
                "status": _format_eval_status(run),
                "model": run.model.get("name") if run.model else "—",
                "creator_name": run.creator_name or "—",
                "created_at": format_local_date(run.created_at),
            }
            for run in eval_runs
        ],
        display_hints=["Use osmosis eval info <name> for details."],
    )


def info(name_or_id: str) -> DetailResult:
    """Show evaluation run details and results."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching evaluation run..."):
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

    fields = detail_fields(rows)
    display_hints: list[str] = []

    if eval_run.get("platform_url"):
        display_hints.append(f"View: {eval_run['platform_url']}")

    if eval_run.get("status") in EVAL_RUN_STATUSES_IN_PROGRESS:
        fields.append(
            DetailField(
                label="Note",
                value="Evaluation run is in progress. Results shown are a snapshot.",
            )
        )
        display_hints.append(
            f"Stop with: osmosis eval stop {eval_run.get('name') or name_or_id}"
        )

    return DetailResult(
        title="Evaluation Run",
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
    """Stop an evaluation run."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials

    require_confirmation(
        f'Stop evaluation run "{name_or_id}"?',
        yes=yes,
        default=False,
        summary=[("Name", name_or_id)],
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Stopping evaluation run..."):
        client.stop_eval_run(
            name_or_id,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    return OperationResult(
        operation="eval.stop",
        status="success",
        resource={"name": name_or_id, **git_result_context(context)},
        message=f'Evaluation run "{name_or_id}" stopped.',
    )

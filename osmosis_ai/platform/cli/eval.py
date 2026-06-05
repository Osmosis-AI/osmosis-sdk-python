"""Handler for `osmosis eval` remote subcommands (submit/list/info/stop)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.output import (
    DetailResult,
    DetailSection,
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
    SubmitRunResult,
)
from osmosis_ai.platform.cli.eval_config import (
    EvalSubmitConfig,
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)
from osmosis_ai.platform.cli.shared_submit import CloudSubmitSpec, run_cloud_submit
from osmosis_ai.platform.cli.utils import (
    format_eval_status,
    format_progress,
    make_progress,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context


def _ref_name(ref: dict[str, Any] | None) -> str | None:
    if not ref:
        return None
    value = ref.get("name") or ref.get("file_name") or ref.get("model_name")
    return value if isinstance(value, str) else None


def _results_number(results: dict[str, Any] | None, key: str) -> int | float | None:
    if not isinstance(results, dict):
        return None
    value = results.get(key)
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return value


def _config_positive_int(config: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(config, dict):
        return None
    value = config.get(key)
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _eval_progress(detail: Any) -> dict[str, Any] | None:
    """Single source of truth for eval progress (``completed`` / ``total``).

    Total is ``sampled_rows * n`` but widened to ``completed`` when ``n`` is
    unknown and more runs have completed than rows — ``total`` is only a lower
    bound in that case, so widening (rather than clamping ``completed`` down)
    avoids hiding real progress. Both the rendered string and the JSON
    ``summary`` derive from this so they cannot diverge.
    """
    completed = _results_number(detail.results, "total_runs")
    sampled_rows = _results_number(detail.results, "sampled_rows")
    if completed is None or sampled_rows is None:
        return None
    n = _config_positive_int(getattr(detail, "config", None), "n") or 1
    total = max(int(sampled_rows) * n, int(completed))
    return make_progress(completed, total, "samples")


def _format_eval_progress(detail: Any) -> str | None:
    progress = format_progress(_eval_progress(detail))
    if progress is not None:
        return progress

    completed = _results_number(detail.results, "total_runs")
    if completed is None:
        return "Waiting to start..." if detail.status == "pending" else None
    return f"{int(completed):,} samples"


def _format_avg_reward(results: dict[str, Any] | None, *, precision: int) -> str:
    avg_reward = _results_number(results, "score")
    if avg_reward is None:
        return "—"
    return f"{avg_reward:.{precision}f}"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _format_duration_ms(duration_ms: float) -> str:
    duration_ms = max(0.0, duration_ms)
    total_seconds = duration_ms / 1000
    if total_seconds < 60:
        return (
            f"{total_seconds:.1f}s" if total_seconds % 1 else f"{int(total_seconds)}s"
        )

    total_seconds_int = int(total_seconds)
    minutes, seconds = divmod(total_seconds_int, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def _format_eval_duration(detail: Any) -> str | None:
    started_at = _parse_iso_datetime(detail.started_at)
    if started_at is None:
        return None
    completed_at = _parse_iso_datetime(detail.completed_at)
    end = completed_at or datetime.now(UTC)
    return _format_duration_ms((end - started_at).total_seconds() * 1000)


def _format_decimal(value: int | float, *, precision: int = 4) -> str:
    return f"{float(value):.{precision}f}"


def _format_optional_int(value: int | float | None) -> str | None:
    if value is None:
        return None
    return f"{int(value):,}"


def _format_results_counts(results: dict[str, Any] | None) -> str | None:
    graded = _results_number(results, "graded")
    passed = _results_number(results, "passed")
    failed = _results_number(results, "failed")
    skipped = _results_number(results, "skipped")
    values = [graded, passed, failed, skipped]
    if all(value is None for value in values):
        return None
    return (
        f"{int(graded or 0):,} graded, "
        f"{int(passed or 0):,} passed, "
        f"{int(failed or 0):,} failed, "
        f"{int(skipped or 0):,} skipped"
    )


def _reward_stats(results: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(results, dict):
        return None
    reward_stats = results.get("reward_stats")
    return reward_stats if isinstance(reward_stats, dict) else None


def _format_reward_stats(results: dict[str, Any] | None) -> str | None:
    reward_stats = _reward_stats(results)
    if reward_stats is None:
        return None
    parts: list[str] = []
    for key in ("min", "median", "max", "std"):
        value = reward_stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(value):
            continue
        parts.append(f"{key} {_format_decimal(value)}")
    return ", ".join(parts) if parts else None


def _format_pass_at_k(results: dict[str, Any] | None) -> str | None:
    reward_stats = _reward_stats(results)
    if reward_stats is None:
        return None
    pass_at_k = reward_stats.get("pass_at_k")
    if not isinstance(pass_at_k, dict):
        return None

    formatted: list[str] = []
    for key in sorted(
        pass_at_k,
        key=lambda value: (
            (0, int(value), "") if str(value).isdigit() else (1, 0, str(value))
        ),
    ):
        value = pass_at_k.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(value):
            continue
        formatted.append(f"{key}: {value:.1%}")
    return ", ".join(formatted) if formatted else None


def _format_secret_scopes(scopes: dict[str, Any] | None) -> str | None:
    if not scopes:
        return None

    parts: list[str] = []
    for name, raw_scope in sorted(scopes.items()):
        if not isinstance(name, str) or not isinstance(raw_scope, str):
            continue
        if raw_scope == "workspace":
            scope = "workspace"
        elif raw_scope == "user_override":
            scope = "personal, overrides workspace"
        elif raw_scope == "user":
            scope = "personal"
        else:
            scope = raw_scope
        parts.append(f"{name} ({scope})")
    return ", ".join(parts) if parts else None


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_env_config(env_config: dict[str, Any] | None) -> str | None:
    if not env_config:
        return None
    parts = [
        f"{key}={_jsonish(value)}"
        for key, value in sorted(env_config.items())
        if isinstance(key, str)
    ]
    return ", ".join(parts) if parts else None


def _format_eval_config(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None

    ordered_keys = (
        "limit",
        "n",
        "batch_size",
        "pass_threshold",
        "agent_workflow_timeout_s",
        "grader_timeout_s",
    )
    keys = [
        key
        for key in ordered_keys
        if key in config and key != "model_path" and config[key] is not None
    ]
    keys.extend(
        sorted(
            key
            for key in config
            if key not in ordered_keys
            and key != "model_path"
            and config[key] is not None
        )
    )
    parts = [f"{key}={_jsonish(config[key])}" for key in keys]
    return ", ".join(parts) if parts else None


def _kv_section(title: str, rows: list[tuple[str, str]]) -> DetailSection | None:
    """Build a titled key/value section mirroring the main detail table.

    Values are passed through as plain text (never markup) so brackets and other
    Rich-significant characters render literally. Returns ``None`` when there is
    nothing to show so callers can append unconditionally.
    """
    if not rows:
        return None

    from rich import box
    from rich.table import Table
    from rich.text import Text

    table = Table(title=title, box=box.ROUNDED, show_header=False, title_justify="left")
    table.add_column("", style="cyan")
    table.add_column("")
    plain_lines = [f"{title}:"]
    for label, value in rows:
        table.add_row(label, Text(value))
        plain_lines.append(f"{label}: {value}")
    return DetailSection(rich=table, plain_lines=plain_lines)


def _eval_summary(detail: Any, *, include_details: bool) -> dict[str, Any]:
    results = detail.results if isinstance(detail.results, dict) else None
    summary: dict[str, Any] = {}

    avg_reward = _results_number(results, "score")
    if avg_reward is not None:
        summary["avg_reward"] = avg_reward

    pass_rate = _results_number(results, "pass_rate")
    if pass_rate is not None:
        summary["pass_rate"] = pass_rate

    for key in ("graded", "passed", "failed", "skipped"):
        value = _results_number(results, key)
        if value is not None:
            summary[key] = int(value)

    progress = _eval_progress(detail)
    if progress is not None:
        summary["progress"] = progress

    if include_details:
        pass_threshold = _results_number(results, "pass_threshold")
        if pass_threshold is not None:
            summary["pass_threshold"] = pass_threshold

        total_tokens = _results_number(results, "total_tokens")
        if total_tokens is not None:
            summary["total_tokens"] = int(total_tokens)

        dataset_rows = _results_number(results, "total_dataset_rows")
        if dataset_rows is not None:
            summary["dataset_rows"] = int(dataset_rows)

        dominant_error_type = results.get("dominant_error_type") if results else None
        if isinstance(dominant_error_type, str) and dominant_error_type:
            summary["dominant_error_type"] = dominant_error_type

        resumed_count = _results_number(results, "resumed_count")
        if resumed_count is not None:
            summary["resumed_count"] = int(resumed_count)

    return summary


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
        items=[
            {
                **serialize_eval_run(r),
                "summary": _eval_summary(r, include_details=False),
            }
            for r in eval_runs
        ],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="rollout", label="Rollout", ratio=2, overflow="fold"),
            ListColumn(key="avg_reward", label="Avg. Reward", no_wrap=True, ratio=1),
            ListColumn(key="created_at", label="Submitted", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Submitted By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_eval_run(run),
                "name": run.name,
                "status": format_eval_status(run),
                "rollout": _ref_name(run.rollout) or "—",
                "avg_reward": _format_avg_reward(run.results, precision=2),
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

    # Main info — mirrors the Platform detail page sidebar (identity + timing +
    # what was run). Configuration and results live in their own sections below.
    rows: list[tuple[str, str]] = [
        ("Name", console.escape(detail.name or "(unnamed)")),
        ("ID", detail.id),
        ("Status", detail.status),
    ]

    progress = _format_eval_progress(detail)
    if progress:
        rows.append(("Progress", progress))

    duration = _format_eval_duration(detail)
    if duration:
        rows.append(("Duration", duration))

    if detail.created_at:
        rows.append(("Submitted", format_local_datetime(detail.created_at)))
    if detail.creator_name:
        rows.append(("Submitted By", console.escape(detail.creator_name)))
    if detail.started_at:
        rows.append(("Started", format_local_datetime(detail.started_at)))
    if detail.completed_at:
        rows.append(("Completed", format_local_datetime(detail.completed_at)))

    dataset_name = _ref_name(detail.dataset)
    if dataset_name:
        rows.append(("Dataset", console.escape(dataset_name)))
    model_name = _ref_name(detail.model)
    if model_name:
        rows.append(("Model", console.escape(model_name)))
    rollout_name = _ref_name(detail.rollout)
    if rollout_name:
        rows.append(("Rollout", console.escape(rollout_name)))

    # Configuration section — what the run was launched with.
    config_rows: list[tuple[str, str]] = []
    if detail.entrypoint:
        config_rows.append(("Entrypoint", detail.entrypoint))
    config = _format_eval_config(detail.config)
    if config:
        config_rows.append(("Config", config))
    if detail.commit_sha:
        config_rows.append(("Commit", detail.commit_sha[:7]))
    secret_scopes = _format_secret_scopes(detail.resolved_secret_scopes)
    if secret_scopes:
        config_rows.append(("Required Secrets", secret_scopes))
    env_config = _format_env_config(detail.env_config)
    if env_config:
        config_rows.append(("Environment", env_config))

    # Results section — scoring outcome.
    result_rows: list[tuple[str, str]] = []
    if detail.results:
        results_counts = _format_results_counts(detail.results)
        if results_counts:
            result_rows.append(("Results", results_counts))

        avg_reward = _format_avg_reward(detail.results, precision=4)
        if avg_reward != "—":
            result_rows.append(("Avg. Reward", avg_reward))
        pass_rate = _results_number(detail.results, "pass_rate")
        if pass_rate is not None:
            result_rows.append(("Pass Rate", f"{pass_rate:.1%}"))
        pass_threshold = _results_number(detail.results, "pass_threshold")
        if pass_threshold is not None:
            result_rows.append(("Pass Threshold", _format_decimal(pass_threshold)))
        reward_stats = _format_reward_stats(detail.results)
        if reward_stats:
            result_rows.append(("Reward Stats", reward_stats))
        pass_at_k = _format_pass_at_k(detail.results)
        if pass_at_k:
            result_rows.append(("Pass@k", pass_at_k))
        total_tokens = _format_optional_int(
            _results_number(detail.results, "total_tokens")
        )
        if total_tokens:
            result_rows.append(("Total Tokens", total_tokens))
        dataset_rows = _format_optional_int(
            _results_number(detail.results, "total_dataset_rows")
        )
        if dataset_rows:
            result_rows.append(("Dataset Rows", dataset_rows))

    fields = detail_fields(rows)
    sections: list[DetailSection] = []
    for section in (
        _kv_section("Configuration", config_rows),
        _kv_section("Results", result_rows),
    ):
        if section is not None:
            sections.append(section)

    display_hints: list[str] = []

    if detail.platform_url:
        display_hints.append(f"View: {detail.platform_url}")

    if detail.status in EVAL_RUN_STATUSES_IN_PROGRESS:
        display_hints.append(
            f"Stop with: osmosis eval stop {detail.name or name_or_id}"
        )

    return DetailResult(
        title="Evaluation Run",
        data={
            "eval_run": serialize_eval_run(detail),
            "summary": _eval_summary(detail, include_details=True),
            "config": detail.config,
            "results": detail.results,
            "model": detail.model,
            "dataset": detail.dataset,
            "rollout": detail.rollout,
            "entrypoint": detail.entrypoint,
            "commit_sha": detail.commit_sha,
            "env_config": detail.env_config,
            "resolved_secret_scopes": detail.resolved_secret_scopes,
            "dataset_df_stats": detail.dataset_df_stats,
            "recent_logs": detail.recent_logs or [],
            **git_result_context(context),
        },
        fields=fields,
        sections=sections,
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

"""Handlers for benchmark catalog discovery and run management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    OperationResult,
    detail_fields,
    get_output_context,
    serialize_benchmark_run,
)
from osmosis_ai.cli.output.display import format_local_date, format_local_datetime
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    BENCHMARK_RUN_STATUSES_ERROR,
    BENCHMARK_RUN_STATUSES_PENDING,
    BENCHMARK_RUN_STATUSES_TERMINAL,
    BenchmarkCatalogDetail,
    BenchmarkCatalogEntry,
    BenchmarkRun,
    BenchmarkRunDetail,
    BenchmarkTaskSet,
    SubmitBenchmarkRunResult,
)
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.benchmark_config import (
    BenchmarkSubmitConfig,
    load_benchmark_submit_config,
)
from osmosis_ai.platform.cli.shared_config import (
    build_env_table_rows,
    build_secret_table_rows,
)
from osmosis_ai.platform.cli.shared_submit import (
    _enrich_missing_secret_error,
    _fetch_secret_scopes,
    _missing_secret_message,
)
from osmosis_ai.platform.cli.utils import (
    build_logs_result,
    format_benchmark_status,
    format_env_config,
    format_progress,
    format_secret_scopes,
    jsonish,
    kv_section,
    make_progress,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_workspace_directory_contract,
)

_HLE_PARITY_WARNING = (
    'For HLE, we recommend [tasks] task_set = "parity" so results are '
    "comparable with published scores. This submission uses the full or a "
    "custom task selection."
)

_BENCHMARK_COLUMNS = [
    ListColumn(key="name", label="Name", ratio=3, overflow="fold"),
    ListColumn(key="key", label="Key", no_wrap=True, min_width=20),
    ListColumn(key="task_count", label="Tasks", no_wrap=True, ratio=1),
    ListColumn(key="category_count", label="Categories", no_wrap=True, ratio=1),
    ListColumn(key="task_sets", label="Named Task Sets", ratio=2, overflow="fold"),
]


def _task_set_resource(task_set: BenchmarkTaskSet) -> dict[str, Any]:
    return {
        "name": task_set.name,
        "task_count": task_set.task_count,
        "recommended": task_set.recommended,
        "description": task_set.description,
    }


def _benchmark_resource(
    benchmark: BenchmarkCatalogEntry | BenchmarkCatalogDetail,
) -> dict[str, Any]:
    return {
        "id": benchmark.id,
        "name": benchmark.name,
        "key": benchmark.source_ref,
        "description": benchmark.description,
        "source_type": benchmark.source_type,
        "source_ref": benchmark.source_ref,
        "task_count": benchmark.task_count,
        "category_count": benchmark.category_count,
        "task_sets": [_task_set_resource(task_set) for task_set in benchmark.task_sets],
        "sync_status": benchmark.sync_status,
        "synced_task_count": benchmark.synced_task_count,
        "sync_error": benchmark.sync_error,
        "platform_url": benchmark.platform_url,
    }


def _task_count_display(
    benchmark: BenchmarkCatalogEntry | BenchmarkCatalogDetail,
) -> str:
    """Task total, or sync progress while the registry manifest pages in."""
    if benchmark.is_ready:
        return f"{benchmark.task_count:,}"
    if benchmark.sync_status == "failed":
        return "unavailable"
    return f"{benchmark.synced_task_count:,} / {benchmark.task_count:,} syncing"


def _sync_hints(
    benchmark: BenchmarkCatalogEntry | BenchmarkCatalogDetail,
) -> list[str]:
    if benchmark.is_ready:
        return []
    location = (
        f" Retry its sync at {benchmark.platform_url}."
        if benchmark.platform_url
        else ""
    )
    if benchmark.sync_status == "failed":
        reason = benchmark.sync_error or "Its task list failed to sync."
        return [f"{benchmark.name} is not runnable: {reason}{location}"]
    return [
        f"{benchmark.name} is still syncing its task list "
        f"({benchmark.synced_task_count:,} of {benchmark.task_count:,} tasks); "
        "submit once it is ready."
    ]


def _task_set_display(task_sets: list[BenchmarkTaskSet]) -> str:
    if not task_sets:
        return "–"
    labels = []
    for task_set in task_sets:
        suffix = ", recommended" if task_set.recommended else ""
        labels.append(f"{task_set.name} ({task_set.task_count:,}{suffix})")
    return ", ".join(labels)


def list_benchmarks(*, limit: int, all_: bool) -> ListResult:
    """List benchmarks available in the current workspace."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()

    with output.status("Fetching benchmarks..."):
        benchmarks, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_benchmarks(
                limit=lim,
                offset=off,
                credentials=context.credentials,
                git_identity=context.git_identity,
            ),
            items_attr="benchmarks",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Benchmarks",
        items=[_benchmark_resource(benchmark) for benchmark in benchmarks],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=_BENCHMARK_COLUMNS,
        display_items=[
            {
                **_benchmark_resource(benchmark),
                "task_count": _task_count_display(benchmark),
                "category_count": f"{benchmark.category_count:,}",
                "task_sets": _task_set_display(benchmark.task_sets),
                "source": (
                    "Managed"
                    if benchmark.source_type == "osmosis_managed"
                    else "Harbor"
                ),
            }
            for benchmark in benchmarks
        ],
        display_hints=[
            "Use osmosis benchmark catalog info <key> for task sets, "
            "categories, and tasks.",
            *[hint for benchmark in benchmarks for hint in _sync_hints(benchmark)],
        ],
    )


def catalog_info(name_or_id: str) -> DetailResult:
    """Show benchmark metadata and task-selection options."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Fetching benchmark "{console.escape(name_or_id)}"...'):
        benchmark = client.get_benchmark(
            name_or_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    harness = (
        "Required"
        if benchmark.requires_harness
        else "Optional"
        if benchmark.supports_harness
        else "Not supported"
    )
    if benchmark.default_harness:
        harness += f" (default: {benchmark.default_harness})"
    judge = "Not required"
    if benchmark.requires_judge_model:
        judge = "Required"
        if benchmark.judge_model_default:
            judge += f" (default: {benchmark.judge_model_default})"

    category_display = ", ".join(
        f"{category.name} ({category.task_count:,})"
        for category in benchmark.categories
    )
    rows = [
        ("Name", console.escape(benchmark.name)),
        ("Key", console.escape(benchmark.source_ref)),
        ("Description", console.escape(benchmark.description or "–")),
        ("Source", f"{benchmark.source_type}: {benchmark.source_ref}"),
        ("Runner", benchmark.runner_family),
        ("Tasks", _task_count_display(benchmark)),
        ("Categories", category_display or "–"),
        ("Named Task Sets", _task_set_display(benchmark.task_sets)),
        ("Harness", harness),
        ("LLM Judge", judge),
        (
            "Required Secret Records",
            ", ".join(benchmark.required_secret_names) or "–",
        ),
        ("Pass Threshold", f"{benchmark.pass_threshold:g}"),
    ]

    benchmark_data = {
        **_benchmark_resource(benchmark),
        "runner_family": benchmark.runner_family,
        "supports_harness": benchmark.supports_harness,
        "requires_harness": benchmark.requires_harness,
        "default_harness": benchmark.default_harness,
        "requires_judge_model": benchmark.requires_judge_model,
        "judge_model_default": benchmark.judge_model_default,
        "required_secret_names": benchmark.required_secret_names,
        "pass_threshold": benchmark.pass_threshold,
        "categories": [
            {"name": category.name, "task_count": category.task_count}
            for category in benchmark.categories
        ],
        "tasks": benchmark.tasks,
        "unavailable_tasks": benchmark.unavailable_tasks,
    }

    display_hints = [
        f"Omit [tasks] to select all {benchmark.task_count:,} tasks.",
        "Use task_names or categories under [tasks] for a custom subset.",
        "Use osmosis --json benchmark catalog info <key> to inspect the full "
        "task list.",
        *_sync_hints(benchmark),
    ]
    if benchmark.default_harness:
        display_hints.insert(
            0,
            f"{benchmark.name}'s published scores were measured on "
            f'harness = "{benchmark.default_harness}"; another harness is '
            "not comparable with them.",
        )
    for task_set in benchmark.task_sets:
        if task_set.recommended:
            display_hints.insert(
                0,
                f"For {benchmark.name}, we recommend [tasks] task_set = "
                f'"{task_set.name}" ({task_set.task_count:,} tasks).',
            )

    return DetailResult(
        title="Benchmark Info",
        data={"benchmark": benchmark_data, **git_result_context(context)},
        fields=detail_fields(rows),
        display_hints=display_hints,
    )


def _benchmark_progress(run: BenchmarkRun) -> dict[str, Any] | None:
    return make_progress(run.ingested_results, run.expected_results, "results")


def _format_pass_at_1(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def list_benchmark_runs(*, limit: int, all_: bool) -> ListResult:
    """List benchmark runs for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching benchmark runs..."):
        runs, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_benchmark_runs(
                limit=lim,
                offset=off,
                credentials=context.credentials,
                git_identity=context.git_identity,
            ),
            items_attr="benchmark_runs",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Benchmark Runs",
        items=[
            {
                **serialize_benchmark_run(run),
                "progress": _benchmark_progress(run),
            }
            for run in runs
        ],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name", ratio=3, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="benchmark", label="Benchmark", ratio=2, overflow="fold"),
            ListColumn(key="progress", label="Progress", no_wrap=True, ratio=2),
            ListColumn(key="best_pass_at_1", label="Best Pass@1", no_wrap=True),
            ListColumn(key="created_at", label="Submitted", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Submitted By", no_wrap=True),
        ],
        display_items=[
            {
                **serialize_benchmark_run(run),
                "status": format_benchmark_status(run),
                "benchmark": run.benchmark_name or "–",
                "progress": format_progress(_benchmark_progress(run)) or "–",
                "best_pass_at_1": _format_pass_at_1(run.best_pass_at_1),
                "created_at": format_local_date(run.created_at),
                "creator_name": run.creator_name or "–",
            }
            for run in runs
        ],
        display_hints=["Use osmosis benchmark info <name-or-id> for details."],
    )


def _configuration_rows(detail: BenchmarkRunDetail) -> list[tuple[str, str]]:
    configuration = detail.configuration or {}
    rows: list[tuple[str, str]] = []
    source_type = configuration.get("source_type")
    source_ref = configuration.get("source_ref")
    if source_type or source_ref:
        rows.append(
            (
                "Source",
                ": ".join(str(value) for value in (source_type, source_ref) if value),
            )
        )
    version = configuration.get("resolved_version")
    if version:
        rows.append(("Version", str(version)))
    digest = configuration.get("resolved_digest")
    if digest:
        rows.append(("Digest", str(digest)))
    task_filters = configuration.get("task_filters")
    if task_filters:
        rows.append(("Tasks", jsonish(task_filters)))
    config = configuration.get("config")
    if config:
        rows.append(("Settings", jsonish(config)))
    scopes = format_secret_scopes(configuration.get("resolved_secret_scopes"))
    if scopes:
        rows.append(("Secrets", scopes))
    env = format_env_config(configuration.get("env_config"))
    if env:
        rows.append(("Environment Variables", env))
    return rows


def _agent_rows(detail: BenchmarkRunDetail) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for position, agent in enumerate(detail.agents or [], start=1):
        index = agent.get("agent_index")
        label = f"Agent {index + 1 if isinstance(index, int) else position}"
        model = (
            agent.get("model_display_name")
            or agent.get("model")
            or agent.get("model_ref")
            or "–"
        )
        if not isinstance(model, str):
            model = jsonish(model)
        harness = agent.get("harness") or "default"
        status = agent.get("status")
        value = f"{harness} · {model}"
        if status:
            value += f" · {status}"
        rows.append((label, value))
    return rows


def _result_rows(detail: BenchmarkRunDetail) -> list[tuple[str, str]]:
    totals = detail.totals or {}
    rows: list[tuple[str, str]] = []
    outcome_parts = [
        f"{int(totals[key]):,} {key}"
        for key in ("passed", "failed", "errored", "cancelled")
        if isinstance(totals.get(key), int | float)
    ]
    if outcome_parts:
        rows.append(("Outcomes", ", ".join(outcome_parts)))
    for key, label in (
        ("total_input_tokens", "Input Tokens"),
        ("total_output_tokens", "Output Tokens"),
        ("total_cost_usd", "Reported Cost"),
    ):
        value = totals.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        rows.append(
            (
                label,
                f"${value:,.4f}" if key == "total_cost_usd" else f"{int(value):,}",
            )
        )
    return rows


def run_info(name_or_id: str) -> DetailResult:
    """Show benchmark run details, progress, configuration, and results."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching benchmark run..."):
        detail = client.get_benchmark_run(
            name_or_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    rows: list[tuple[str, str]] = [
        ("Name", console.escape(detail.name)),
        ("ID", detail.id),
    ]
    rows.extend(
        [
            ("Status", detail.status.replace("_", " ").title()),
            ("Benchmark", console.escape(detail.benchmark_name or "–")),
        ]
    )
    progress = detail.progress
    if isinstance(progress, dict):
        progress = make_progress(
            progress.get("ingested"),
            progress.get("expected"),
            "results",
        )
    else:
        progress = _benchmark_progress(detail)
    progress_label = format_progress(progress)
    if progress_label:
        rows.append(("Progress", progress_label))
    if detail.best_pass_at_1 is not None:
        rows.append(("Best Pass@1", _format_pass_at_1(detail.best_pass_at_1)))
    rows.append(("Agents", f"{detail.agent_count:,}"))
    if detail.created_at:
        rows.append(("Submitted", format_local_datetime(detail.created_at)))
    if detail.creator_name:
        rows.append(("Submitted By", console.escape(detail.creator_name)))
    if detail.started_at:
        rows.append(("Started", format_local_datetime(detail.started_at)))
    if detail.completed_at:
        rows.append(("Completed", format_local_datetime(detail.completed_at)))

    sections: list[DetailSection] = []
    for section in (
        kv_section("Configuration", _configuration_rows(detail)),
        kv_section("Agents", _agent_rows(detail)),
        kv_section("Results", _result_rows(detail)),
    ):
        if section is not None:
            sections.append(section)

    display_hints: list[str] = []
    if detail.platform_url:
        display_hints.append(f"View: {detail.platform_url}")
    if detail.status in BENCHMARK_RUN_STATUSES_ERROR:
        display_hints.append(f"See logs with: osmosis benchmark logs {detail.name}")
    if detail.status not in BENCHMARK_RUN_STATUSES_TERMINAL:
        display_hints.append(f"Stop with: osmosis benchmark stop {detail.name}")
    display_hints.append(
        f"Download outputs with: osmosis benchmark download {detail.name}"
    )

    return DetailResult(
        title="Benchmark Run",
        data={
            "benchmark_run": serialize_benchmark_run(detail),
            "configuration": detail.configuration,
            "agents": detail.agents,
            "progress": progress,
            "totals": detail.totals,
            "agent_metrics": detail.agent_metrics,
            **git_result_context(context),
        },
        fields=detail_fields(rows),
        sections=sections,
        display_hints=display_hints,
    )


def logs(name_or_id: str, *, limit: int, cursor: str | None = None) -> ListResult:
    """Show the most recent logs for a benchmark run, oldest-first."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching logs..."):
        page = client.get_benchmark_run_logs(
            name_or_id,
            limit=limit,
            cursor=cursor,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    return build_logs_result(
        title=f"Benchmark Run Logs: {name_or_id}",
        page=page,
        context=context,
        next_step_hint=f"Use osmosis benchmark info {name_or_id} for run details.",
    )


def download(
    name_or_id: str,
    *,
    output: str | None,
    types: str = "summary,results",
    overwrite: bool = False,
    yes: bool = False,
) -> OperationResult:
    """Download selected benchmark run outputs through the manifest contract."""
    from osmosis_ai.cli.metrics_export import resolve_benchmark_output_dir
    from osmosis_ai.platform.cli.run_download import (
        BENCHMARK_DOWNLOAD_TYPES,
        benchmark_path_category,
        parse_download_types,
        run_download,
    )

    selected_types = parse_download_types(types, allowed=BENCHMARK_DOWNLOAD_TYPES)
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output_ctx = get_output_context()
    with output_ctx.status("Fetching benchmark run..."):
        detail = client.get_benchmark_run(
            name_or_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    if detail.status in BENCHMARK_RUN_STATUSES_PENDING:
        raise CLIError(
            "Outputs are not yet available for pending or queued benchmark runs.",
            code="CONFLICT",
        )
    try:
        return run_download(
            run_id=detail.id,
            run_name=detail.name,
            run_status=detail.status,
            selected_types=selected_types,
            output=output,
            overwrite=overwrite,
            yes=yes,
            workspace_directory=context.workspace_directory,
            result_context=git_result_context(context),
            manifest_loader=lambda requested_types: (
                client.get_benchmark_run_download_manifest(
                    detail.id,
                    types=requested_types,
                    credentials=context.credentials,
                    git_identity=context.git_identity,
                )
            ),
            url_loader=lambda items: client.get_benchmark_run_download_urls(
                detail.id,
                items=items,
                credentials=context.credentials,
                git_identity=context.git_identity,
            ),
            output_resolver=resolve_benchmark_output_dir,
            path_category=benchmark_path_category,
            operation="benchmark.download",
            resource_key="benchmark_run",
        )
    except PlatformAPIError as exc:
        if exc.status_code == 404:
            raise CLIError(
                "Benchmark run output route was not found. The run may have been "
                "deleted or the platform may not support benchmark downloads yet."
            ) from exc
        raise


def stop(name_or_id: str, *, yes: bool) -> OperationResult:
    """Stop a benchmark run."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()
    with output.status("Fetching benchmark run..."):
        detail = client.get_benchmark_run(
            name_or_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    require_confirmation(
        f'Stop benchmark run "{detail.name}"?',
        yes=yes,
        default=False,
        summary=[("Name", detail.name), ("ID", detail.id)],
    )
    with output.status("Stopping benchmark run..."):
        client.stop_benchmark_run(
            detail.id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    return OperationResult(
        operation="benchmark.stop",
        status="success",
        resource={"id": detail.id, "name": detail.name, "status": "stopped"},
        message=f'Benchmark run "{detail.name}" stopped.',
    )


def _agent_model_label(agent: dict[str, Any]) -> str:
    model = agent["model"]
    if model["type"] == "hosted":
        return f"{model['base_model']}:{model['checkpoint_name']}"
    return str(model["model"])


def _task_selection_label(config: BenchmarkSubmitConfig) -> str:
    tasks = config.tasks_config
    if tasks.get("task_set"):
        return str(tasks["task_set"])
    names = tasks.get("task_names")
    categories = tasks.get("categories")
    parts: list[str] = []
    if isinstance(names, list) and names:
        parts.append(f"{len(names)} task(s)")
    if isinstance(categories, list) and categories:
        parts.append(f"{len(categories)} category(s)")
    return ", ".join(parts) if parts else "all tasks"


def _warn_if_hle_without_parity(config: BenchmarkSubmitConfig) -> None:
    # Catch likely casing mistakes too: the route will still return its precise
    # case-sensitive benchmark-name error after the user sees this guidance.
    benchmark_name = config.experiment.benchmark.strip().casefold()
    if benchmark_name == "hle" and config.tasks.task_set != "parity":
        console.print_warning(
            _HLE_PARITY_WARNING,
            code="HLE_PARITY_RECOMMENDED",
        )


def _submit_benchmark(
    client: OsmosisClient,
    config: BenchmarkSubmitConfig,
    credentials: Any,
    git_identity: str,
) -> SubmitBenchmarkRunResult:
    return client.submit_benchmark_run(
        experiment_config=config.experiment_config,
        tasks_config=config.tasks_config or None,
        agents=config.agents_config,
        execution_config=config.execution_config or None,
        env_config=config.env or None,
        credentials=credentials,
        git_identity=git_identity,
    )


def submit(config_path: Path, *, yes: bool) -> OperationResult:
    """Submit a benchmark run."""
    context = require_git_workspace_directory_context()
    workspace_directory = Path(context.workspace_directory)
    validate_workspace_directory_contract(workspace_directory)

    path = Path(config_path)
    resolved_path = path if path.is_absolute() else workspace_directory / path
    ensure_workspace_directory_config_path(
        resolved_path,
        workspace_directory,
        config_dir="configs/benchmark",
        command_label="`osmosis benchmark submit`",
    )
    config = load_benchmark_submit_config(resolved_path)

    execution = config.execution_config
    summary_rows = [
        ("Benchmark", config.experiment.benchmark),
        ("Tasks", _task_selection_label(config)),
        ("Agents", str(len(config.agents))),
        ("Attempts per task", str(execution.get("attempts_per_task", 1))),
        (
            "Max concurrent attempts",
            str(execution.get("max_concurrent_attempts", 4)),
        ),
    ]
    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title="Benchmark Run",
    )
    agent_summary_rows = [
        (
            f"{index} · {agent.harness or 'default'}",
            _agent_model_label(config.agents_config[index - 1]),
        )
        for index, agent in enumerate(config.agents, start=1)
    ]
    console.table(
        [
            (console.escape(agent), console.escape(model))
            for agent, model in agent_summary_rows
        ],
        title=f"Agents ({len(agent_summary_rows)})",
        headers=("Agent", "Model"),
    )

    full_summary: list[tuple[str, str]] = list(summary_rows)
    full_summary.extend(
        (f"agent.{agent}", model) for agent, model in agent_summary_rows
    )

    env_rows = build_env_table_rows(config.env)
    for index, agent in enumerate(config.agents, start=1):
        env_rows.extend(
            (
                f"agent {index} · {name}"
                + (" (overrides global)" if name in config.env else ""),
                value,
            )
            for name, value in build_env_table_rows(agent.env)
        )
    if env_rows:
        console.table(
            [(name, console.escape(value)) for name, value in env_rows],
            title=f"Env Vars ({len(env_rows)})",
            headers=("Name", "Value"),
        )
        full_summary.extend((f"env.{name}", value) for name, value in env_rows)

    _warn_if_hle_without_parity(config)

    if config.required_secrets:
        scopes = _fetch_secret_scopes(
            OsmosisClient(),
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
        if scopes is None:
            secret_rows = [(name, "–") for name in sorted(config.required_secrets)]
        else:
            workspace_names, personal_names = scopes
            missing = sorted(
                name
                for name in config.required_secrets
                if name not in workspace_names and name not in personal_names
            )
            if missing:
                raise CLIError(_missing_secret_message(missing))
            secret_rows = build_secret_table_rows(
                config.required_secrets,
                user_secret_names=personal_names,
                workspace_secret_names=workspace_names,
            )
        console.table(
            secret_rows,
            title=f"Secrets ({len(secret_rows)})",
            headers=("Name", "Scope"),
        )
        full_summary.extend((f"secret.{name}", scope) for name, scope in secret_rows)

    require_confirmation(
        "Submit this benchmark run?",
        yes=yes,
        summary=full_summary,
    )

    output = get_output_context()
    with output.status("Submitting benchmark run..."):
        try:
            result = _submit_benchmark(
                OsmosisClient(),
                config,
                context.credentials,
                context.git_identity,
            )
        except PlatformAPIError as exc:
            enriched = _enrich_missing_secret_error(exc)
            if enriched is not None:
                raise enriched from exc
            raise

    display_next_steps = [
        f"Status: {result.status}",
        f"Benchmark: {config.experiment.benchmark}",
        f"Check status with: osmosis benchmark info {result.name}",
    ]
    structured_next_steps: list[dict[str, Any]] = [
        {"action": "benchmark_info", "name": result.name},
        {"action": "benchmark_list"},
    ]
    if result.platform_url:
        display_next_steps.append(f"View: {result.platform_url}")
        structured_next_steps.append({"action": "open_url", "url": result.platform_url})

    return OperationResult(
        operation="benchmark.submit",
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "benchmark_name": config.experiment.benchmark,
            "workflow_id": result.workflow_id,
            "task_count": result.task_count,
            "created_at": result.created_at,
            **({"platform_url": result.platform_url} if result.platform_url else {}),
            **git_result_context(context),
            "config": {
                "experiment": config.experiment_config,
                "tasks": config.tasks_config,
                "agents": config.agents_config,
                "execution": config.execution_config,
            },
        },
        message=f"Benchmark run submitted: {result.name}",
        display_next_steps=display_next_steps,
        next_steps_structured=structured_next_steps,
    )


__all__ = [
    "catalog_info",
    "download",
    "list_benchmark_runs",
    "list_benchmarks",
    "logs",
    "run_info",
    "stop",
    "submit",
]

"""Handlers for benchmark catalog discovery and run submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailResult,
    ListColumn,
    ListResult,
    OperationResult,
    detail_fields,
    get_output_context,
)
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    BenchmarkCatalogDetail,
    BenchmarkCatalogEntry,
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
    ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
    ListColumn(key="task_count", label="Tasks", no_wrap=True, ratio=1),
    ListColumn(key="category_count", label="Categories", no_wrap=True, ratio=1),
    ListColumn(key="task_sets", label="Named Task Sets", ratio=2, overflow="fold"),
    ListColumn(key="source", label="Source", no_wrap=True, ratio=1),
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
        "description": benchmark.description,
        "source_type": benchmark.source_type,
        "source_ref": benchmark.source_ref,
        "task_count": benchmark.task_count,
        "category_count": benchmark.category_count,
        "task_sets": [_task_set_resource(task_set) for task_set in benchmark.task_sets],
    }


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
                "task_count": f"{benchmark.task_count:,}",
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
            "Use osmosis benchmark info <name> for task sets, categories, and tasks."
        ],
    )


def info(name_or_id: str) -> DetailResult:
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
        ("Description", console.escape(benchmark.description or "–")),
        ("Source", f"{benchmark.source_type}: {benchmark.source_ref}"),
        ("Runner", benchmark.runner_family),
        ("Tasks", f"{benchmark.task_count:,}"),
        ("Categories", category_display or "–"),
        ("Named Task Sets", _task_set_display(benchmark.task_sets)),
        ("Harness", harness),
        ("LLM Judge", judge),
        ("Pass Threshold", f"{benchmark.pass_threshold:g}"),
    ]

    benchmark_data = {
        **_benchmark_resource(benchmark),
        "runner_family": benchmark.runner_family,
        "supports_harness": benchmark.supports_harness,
        "requires_harness": benchmark.requires_harness,
        "requires_judge_model": benchmark.requires_judge_model,
        "judge_model_default": benchmark.judge_model_default,
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
        "Use osmosis --json benchmark info <name> to inspect the full task list.",
    ]
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
    ]
    structured_next_steps: list[dict[str, Any]] = []
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


__all__ = ["info", "list_benchmarks", "submit"]

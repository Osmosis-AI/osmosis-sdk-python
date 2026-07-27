"""Handler for ``osmosis benchmark submit``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import SubmitBenchmarkRunResult
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
from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_workspace_directory_contract,
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
            "task_count": result.task_count,
            "created_at": result.created_at,
            **({"url": result.platform_url} if result.platform_url else {}),
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


__all__ = ["submit"]

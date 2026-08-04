from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import osmosis_ai.cli.main as cli
import osmosis_ai.platform.cli.benchmark as benchmark_module
from osmosis_ai.cli.output import DetailResult, ListResult, OperationResult
from osmosis_ai.platform.api.models import (
    BenchmarkRun,
    BenchmarkRunDetail,
    LogEntry,
    LogsPage,
    PaginatedBenchmarkRuns,
)

GIT_IDENTITY = "acme/workspace"
FAKE_CREDENTIALS = object()


def _context(tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_directory=tmp_path or Path("/repo"),
        git_identity=GIT_IDENTITY,
        repo_url="https://github.com/acme/workspace.git",
        credentials=FAKE_CREDENTIALS,
    )


def _run(*, status: str = "running") -> BenchmarkRun:
    return BenchmarkRun(
        id="run-1",
        name="hle-smoke",
        status=status,
        benchmark_id="benchmark-1",
        benchmark_name="HLE",
        agent_count=2,
        best_pass_at_1=0.42,
        ingested_results=50,
        expected_results=100,
        creator_name="Ada",
        created_at="2026-07-30T00:00:00Z",
    )


def _detail(
    *,
    status: str = "running",
    is_internal_user: bool = True,
) -> BenchmarkRunDetail:
    run = _run(status=status)
    return BenchmarkRunDetail(
        **run.__dict__,
        configuration={
            "source_type": "osmosis_managed",
            "source_ref": "hle",
            "resolved_version": "1",
            "task_filters": {"task_set": "parity"},
            "config": {"attempts_per_task": 2},
            "resolved_secret_scopes": {"OPENAI_API_KEY": "workspace"},
        },
        agents=[
            {
                "agent_index": 0,
                "harness": "codex",
                "model_display_name": "GPT-5",
                "status": "running",
            },
            {
                "agent_index": 1,
                "harness": None,
                "model_display_name": "Qwen",
                "status": "pending",
            },
        ],
        progress={"ingested": 50, "expected": 100},
        totals={
            "passed": 20,
            "failed": 25,
            "errored": 5,
            "cancelled": 0,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_cost_usd": 1.25,
        },
        agent_metrics=[{"benchmark_run_agent_id": "agent-1"}],
        is_internal_user=is_internal_user,
    )


def test_list_benchmark_runs_returns_public_and_display_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def list_benchmark_runs(self, **kwargs: Any) -> PaginatedBenchmarkRuns:
            calls.append(kwargs)
            return PaginatedBenchmarkRuns(
                benchmark_runs=[_run()],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.list_benchmark_runs(limit=50, all_=False)

    assert isinstance(result, ListResult)
    assert result.items[0]["benchmark_name"] == "HLE"
    assert result.items[0]["progress"] == {
        "completed": 50,
        "total": 100,
        "unit": "results",
    }
    assert result.display_items[0]["progress"] == "50 / 100 results"
    assert result.display_items[0]["best_pass_at_1"] == "42.0%"
    assert result.display_items[0]["agent_count"] == "2"
    assert [column.key for column in result.columns] == [
        "name",
        "status",
        "progress",
        "benchmark",
        "agent_count",
        "best_pass_at_1",
        "created_at",
        "creator_name",
    ]
    assert calls == [
        {
            "limit": 50,
            "offset": 0,
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
    ]


def test_benchmark_list_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeClient:
        def list_benchmark_runs(self, **kwargs: Any) -> PaginatedBenchmarkRuns:
            return PaginatedBenchmarkRuns(
                benchmark_runs=[_run()],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "benchmark", "runs", "list"])
    envelope = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert envelope["schema_version"] == 1
    assert envelope["items"][0]["benchmark_name"] == "HLE"
    assert envelope["items"][0]["progress"]["completed"] == 50


def test_run_info_returns_config_agents_results_and_next_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_benchmark_run(
            self, name_or_id: str, **kwargs: Any
        ) -> BenchmarkRunDetail:
            assert name_or_id == "hle-smoke"
            assert kwargs["credentials"] is FAKE_CREDENTIALS
            return _detail()

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.run_info("hle-smoke")

    assert isinstance(result, DetailResult)
    assert result.data["benchmark_run"]["benchmark_name"] == "HLE"
    assert result.data["benchmark_run"]["benchmark_id"] == "benchmark-1"
    assert result.data["benchmark_run"]["best_pass_at_1"] == 0.42
    assert result.data["progress"] == {
        "completed": 50,
        "total": 100,
        "unit": "results",
    }
    assert result.data["totals"]["passed"] == 20
    assert [section.plain_lines[0] for section in result.sections] == [
        "Configuration:",
        "Agents:",
        "Results:",
    ]
    assert "osmosis benchmark runs stop hle-smoke" in result.display_hints[-2]
    assert "osmosis benchmark runs download hle-smoke" in result.display_hints[-1]


def test_run_info_reports_duration_and_per_agent_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = _detail(status="finished")
    detail.started_at = "2026-07-30T00:05:00Z"
    detail.completed_at = "2026-07-30T01:35:00Z"
    detail.agents = [
        {
            "id": "agent-1",
            "agent_index": 0,
            "harness": "codex",
            "model_display_name": "GPT-5",
            "status": "finished",
            "aggregates": {
                "reported_cost_usd": 498.0,
                "mean_duration_seconds": 54,
                "tokens_per_task": 1_100_000,
            },
        }
    ]
    detail.agent_metrics = [
        {
            "benchmark_run_agent_id": "agent-1",
            "rank": 1,
            "n_tasks": 249,
            "pass_at_1": {
                "value": 0.75,
                "ci_low": 0.719,
                "ci_high": 0.781,
                "n": 249,
            },
            "pass_at_k": [
                {"k": 2, "value": 0.812, "ci_low": 0.77, "ci_high": 0.85, "n": 249}
            ],
        }
    ]

    class FakeClient:
        def get_benchmark_run(self, *_: Any, **__: Any) -> BenchmarkRunDetail:
            return detail

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.run_info("hle-smoke")

    fields = {field.label: field.value for field in result.fields}
    assert fields["Duration"] == "1h 30m"

    agents = next(
        section for section in result.sections if section.plain_lines[0] == "Agents:"
    )
    agent_line = " ".join(agents.plain_lines)
    assert "#1" in agent_line
    assert "pass@1 75.0% (71.9–78.1)" in agent_line
    assert "pass@2 81.2%" in agent_line
    assert "$2.00/task" in agent_line
    assert "54s/task" in agent_line
    assert "1.1M tokens/task" in agent_line

    results = next(
        section for section in result.sections if section.plain_lines[0] == "Results:"
    )
    assert any("LLM Cost" in line for line in results.plain_lines)


def test_run_info_always_displays_canonical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_benchmark_run(
            self, name_or_id: str, **kwargs: Any
        ) -> BenchmarkRunDetail:
            return _detail(is_internal_user=False)

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.run_info("hle-smoke")

    fields = {field.label: field.value for field in result.fields}
    assert fields["ID"] == "run-1"


def test_logs_uses_shared_cursor_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def get_benchmark_run_logs(self, name_or_id: str, **kwargs: Any) -> LogsPage:
            assert name_or_id == "run-1"
            assert kwargs["cursor"] == "cursor-1"
            return LogsPage(
                logs=[
                    LogEntry(
                        timestamp="2026-07-30T00:00:00Z",
                        level="info",
                        step="runner",
                        message="Started",
                    )
                ],
                next_cursor="cursor-2",
            )

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.logs("run-1", limit=20, cursor="cursor-1")

    assert isinstance(result, ListResult)
    assert result.items[0]["message"] == "Started"
    assert result.extra["next_cursor"] == "cursor-2"


def test_stop_confirms_and_calls_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    confirmations: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def get_benchmark_run(
            self, name_or_id: str, **kwargs: Any
        ) -> BenchmarkRunDetail:
            calls.append({"operation": "get", "name_or_id": name_or_id, **kwargs})
            return _detail()

        def stop_benchmark_run(self, name_or_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append({"operation": "stop", "name_or_id": name_or_id, **kwargs})
            return {"status": "stopped"}

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(
        benchmark_module,
        "require_confirmation",
        lambda prompt, **kwargs: confirmations.append({"prompt": prompt, **kwargs}),
    )

    result = benchmark_module.stop("run-1", yes=True)

    assert isinstance(result, OperationResult)
    assert result.operation == "benchmark.stop"
    assert result.resource == {
        "id": "run-1",
        "name": "hle-smoke",
        "status": "stopped",
    }
    assert confirmations[0]["yes"] is True
    assert confirmations[0]["prompt"] == 'Stop benchmark run "hle-smoke"?'
    assert confirmations[0]["summary"] == [
        ("Name", "hle-smoke"),
        ("ID", "run-1"),
    ]
    assert calls == [
        {
            "operation": "get",
            "name_or_id": "run-1",
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        },
        {
            "operation": "stop",
            "name_or_id": "run-1",
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        },
    ]


def test_stop_resolves_name_before_stopping_canonical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def get_benchmark_run(
            self, name_or_id: str, **kwargs: Any
        ) -> BenchmarkRunDetail:
            calls.append(("get", name_or_id))
            return _detail()

        def stop_benchmark_run(self, name_or_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(("stop", name_or_id))
            return {"status": "stopped"}

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", _context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(
        benchmark_module,
        "require_confirmation",
        lambda *args, **kwargs: None,
    )

    result = benchmark_module.stop("hle-smoke", yes=True)

    assert calls == [("get", "hle-smoke"), ("stop", "run-1")]
    assert result.resource == {
        "id": "run-1",
        "name": "hle-smoke",
        "status": "stopped",
    }

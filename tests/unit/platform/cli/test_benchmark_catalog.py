from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import osmosis_ai.platform.cli.benchmark as benchmark_module
from osmosis_ai.cli.output import DetailResult, ListResult
from osmosis_ai.platform.api.models import (
    BenchmarkCatalogDetail,
    BenchmarkCatalogEntry,
    BenchmarkCategory,
    BenchmarkRun,
    BenchmarkTaskSet,
    PaginatedBenchmarkRuns,
    PaginatedBenchmarks,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE


def _empty_runs_page(**kwargs: Any) -> PaginatedBenchmarkRuns:
    return PaginatedBenchmarkRuns(
        benchmark_runs=[],
        total_count=0,
        has_more=False,
        next_offset=None,
    )


GIT_IDENTITY = "acme/workspace"
REPO_URL = "https://github.com/acme/workspace.git"
FAKE_CREDENTIALS = object()


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_directory=Path("/repo"),
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=FAKE_CREDENTIALS,
    )


def test_list_benchmarks_returns_catalog_and_git_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    parity = BenchmarkTaskSet(
        name="parity",
        task_count=249,
        recommended=True,
        description="Published comparison sample.",
    )

    class FakeClient:
        def list_benchmarks(self, **kwargs: Any) -> PaginatedBenchmarks:
            calls.append(kwargs)
            return PaginatedBenchmarks(
                benchmarks=[
                    BenchmarkCatalogEntry(
                        id="benchmark-1",
                        name="HLE",
                        description="Humanity's Last Exam",
                        source_type="osmosis_managed",
                        source_ref="hle",
                        task_count=2_500,
                        category_count=30,
                        task_sets=[parity],
                    )
                ],
                total_count=1,
                has_more=False,
                next_offset=None,
            )

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.list_benchmarks(limit=50, all_=False)

    assert isinstance(result, ListResult)
    assert result.total_count == 1
    assert result.items[0]["name"] == "HLE"
    assert result.items[0]["key"] == "hle"
    assert result.items[0]["task_sets"] == [
        {
            "name": "parity",
            "task_count": 249,
            "recommended": True,
            "description": "Published comparison sample.",
        }
    ]
    assert result.display_items[0]["key"] == "hle"
    assert result.display_items[0]["status"] == "Ready"
    assert result.display_items[0]["run_count"] == "0"
    assert result.display_items[0]["last_run_at"] == "–"
    assert [column.key for column in result.columns] == [
        "name",
        "key",
        "status",
        "run_count",
        "last_run_at",
        "task_count",
    ]
    assert result.columns[1].no_wrap is True
    assert result.columns[1].min_width == 20
    assert result.extra["git"]["identity"] == GIT_IDENTITY
    assert calls == [
        {
            "limit": 50,
            "offset": 0,
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
    ]


def test_info_exposes_selection_metadata_and_full_task_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    parity = BenchmarkTaskSet(
        name="parity",
        task_count=249,
        recommended=True,
        description="Published comparison sample.",
    )

    class FakeClient:
        def get_benchmark(
            self,
            name_or_id: str,
            **kwargs: Any,
        ) -> BenchmarkCatalogDetail:
            calls.append({"name_or_id": name_or_id, **kwargs})
            return BenchmarkCatalogDetail(
                id="benchmark-1",
                name="HLE",
                description="Humanity's Last Exam",
                source_type="osmosis_managed",
                source_ref="hle",
                task_count=2,
                category_count=2,
                task_sets=[parity],
                runner_family="harbor",
                supports_harness=True,
                requires_harness=True,
                requires_judge_model=True,
                judge_model_default="openai/gpt-5",
                pass_threshold=1,
                categories=[
                    BenchmarkCategory(name="math", task_count=1),
                    BenchmarkCategory(name="science", task_count=1),
                ],
                tasks=[
                    {
                        "name": "hle__math",
                        "category": "math",
                        "difficulty": None,
                    },
                    {
                        "name": "hle__science",
                        "category": "science",
                        "difficulty": None,
                    },
                ],
                unavailable_tasks=None,
                required_secret_names=["HF_TOKEN"],
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info("HLE", limit=DEFAULT_PAGE_SIZE, all_=False)

    assert isinstance(result, DetailResult)
    assert result.data["benchmark"]["key"] == "hle"
    assert result.data["benchmark"]["tasks"] == [
        {"name": "hle__math", "category": "math", "difficulty": None},
        {"name": "hle__science", "category": "science", "difficulty": None},
    ]
    assert result.data["benchmark"]["categories"] == [
        {"name": "math", "task_count": 1},
        {"name": "science", "task_count": 1},
    ]
    assert result.data["benchmark"]["required_secret_names"] == ["HF_TOKEN"]
    fields = {field.label: field.value for field in result.fields}
    assert fields["Key"] == "hle"
    assert fields["Required Secret Records"] == "HF_TOKEN"
    assert 'task_set = "parity"' in result.display_hints[0]
    assert "No leaderboard entries yet" in result.display_hints[1]
    assert any("Omit [tasks]" in hint for hint in result.display_hints)
    assert calls == [
        {
            "name_or_id": "HLE",
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
    ]


def test_catalog_detail_defaults_required_secret_names() -> None:
    detail = BenchmarkCatalogDetail(
        id="benchmark-1",
        name="Example",
        description=None,
        source_type="harbor_registry",
        source_ref="example@1",
        task_count=1,
        category_count=0,
        task_sets=[],
        runner_family="harbor",
        supports_harness=True,
        requires_harness=True,
        requires_judge_model=False,
        judge_model_default=None,
        pass_threshold=1,
        categories=[],
        tasks=[{"name": "task-1", "category": None, "difficulty": None}],
        unavailable_tasks=None,
    )

    assert detail.required_secret_names == []


def _syncing_entry(**overrides: Any) -> BenchmarkCatalogEntry:
    return BenchmarkCatalogEntry(
        id="benchmark-2",
        name="acme/custom",
        description=None,
        source_type="harbor_registry",
        source_ref="acme/custom@3",
        task_count=4_000,
        category_count=0,
        task_sets=[],
        platform_url="https://platform.example/Acme/benchmarks/benchmark-2",
        **overrides,
    )


def _list_with(
    monkeypatch: pytest.MonkeyPatch, entry: BenchmarkCatalogEntry
) -> ListResult:
    class FakeClient:
        def list_benchmarks(self, **_: Any) -> PaginatedBenchmarks:
            return PaginatedBenchmarks(
                benchmarks=[entry],
                total_count=1,
                has_more=False,
                next_offset=None,
            )

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)
    return benchmark_module.list_benchmarks(limit=50, all_=False)


def test_list_benchmarks_shows_registry_sync_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _list_with(
        monkeypatch,
        _syncing_entry(sync_status="syncing", synced_task_count=1_000),
    )

    assert result.display_items[0]["task_count"] == "1,000 / 4,000 syncing"
    assert result.items[0]["sync_status"] == "syncing"
    assert any("still syncing" in hint for hint in result.display_hints)


def test_list_benchmarks_surfaces_failed_sync_and_platform_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _list_with(
        monkeypatch,
        _syncing_entry(
            sync_status="failed",
            sync_error="Task list unavailable.",
        ),
    )

    assert result.display_items[0]["task_count"] == "unavailable"
    assert result.items[0]["sync_error"] == "Task list unavailable."
    assert any(
        "Task list unavailable." in hint
        and "https://platform.example/Acme/benchmarks" in hint
        for hint in result.display_hints
    )


def test_benchmark_info_surfaces_the_default_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_benchmark(self, *_: Any, **__: Any) -> BenchmarkCatalogDetail:
            return BenchmarkCatalogDetail(
                id="benchmark-1",
                name="Terminal-Bench 2.1",
                description=None,
                source_type="osmosis_managed",
                source_ref="terminal-bench-2-1",
                task_count=89,
                category_count=1,
                task_sets=[],
                runner_family="harbor",
                supports_harness=True,
                requires_harness=True,
                requires_judge_model=False,
                judge_model_default=None,
                pass_threshold=1,
                categories=[],
                tasks=[],
                unavailable_tasks=None,
                default_harness="terminus-2",
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info(
        "terminal-bench-2-1", limit=DEFAULT_PAGE_SIZE, all_=False
    )

    assert result.data["benchmark"]["default_harness"] == "terminus-2"
    fields = {field.label: field.value for field in result.fields}
    assert fields["Harness"] == "Required (default: terminus-2)"
    assert 'harness = "terminus-2"' in result.display_hints[0]


def test_benchmark_info_names_the_official_scaffold_as_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benchmark that allows but does not require a harness defaults to its own scaffold."""

    class FakeClient:
        def get_benchmark(self, *_: Any, **__: Any) -> BenchmarkCatalogDetail:
            return BenchmarkCatalogDetail(
                id="benchmark-3",
                name="BrowseComp",
                description=None,
                source_type="osmosis_managed",
                source_ref="browsecomp",
                task_count=10,
                category_count=1,
                task_sets=[],
                runner_family="harbor",
                supports_harness=True,
                requires_harness=False,
                requires_judge_model=False,
                judge_model_default=None,
                pass_threshold=1,
                categories=[],
                tasks=[],
                unavailable_tasks=None,
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info(
        "browsecomp", limit=DEFAULT_PAGE_SIZE, all_=False
    )

    fields = {field.label: field.value for field in result.fields}
    assert fields["Harness"] == "Optional (default: official scaffold)"
    assert "[[agents]] entry" in result.display_hints[0]


def test_benchmark_info_reports_a_scaffold_only_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benchmark with no harness support runs its own scaffold, not "nothing"."""

    class FakeClient:
        def get_benchmark(self, *_: Any, **__: Any) -> BenchmarkCatalogDetail:
            return BenchmarkCatalogDetail(
                id="benchmark-4",
                name="Toolathlon-Verified",
                description=None,
                source_type="osmosis_managed",
                source_ref="toolathlon-verified",
                task_count=5,
                category_count=1,
                task_sets=[],
                runner_family="toolathlon",
                supports_harness=False,
                requires_harness=False,
                requires_judge_model=False,
                judge_model_default=None,
                pass_threshold=1,
                categories=[],
                tasks=[],
                unavailable_tasks=None,
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info(
        "toolathlon-verified", limit=DEFAULT_PAGE_SIZE, all_=False
    )

    fields = {field.label: field.value for field in result.fields}
    assert fields["Harness"] == "Official scaffold only"


def test_benchmark_info_renders_leaderboard_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaderboard = [
        {
            "rank": 1,
            "tied": False,
            "task_set": "parity",
            "harness": "codex",
            "model": "GPT-5.5",
            "pass_at_1": {
                "value": 0.75,
                "ci_low": 0.719,
                "ci_high": 0.781,
                "n": 249,
                "method": "wilson",
            },
            "pass_at_k": [
                {"k": 1, "value": 0.75, "ci_low": 0.719, "ci_high": 0.781, "n": 249},
                {"k": 2, "value": 0.812, "ci_low": 0.77, "ci_high": 0.85, "n": 249},
            ],
            "tokens_per_task": 1_100_000,
            "mean_duration_seconds": 54,
            "reported_cost_usd": 4.2,
            "run": {
                "id": "run-1",
                "name": "warm-gull",
                "platform_url": "https://platform.example/Acme/benchmarks/runs/run-1",
            },
        },
        # A sparse entrant: every optional metric missing or the wrong type.
        {"rank": None, "tied": True, "task_set": "full", "model": "122-test-lora"},
    ]

    class FakeClient:
        def get_benchmark(self, *_: Any, **__: Any) -> BenchmarkCatalogDetail:
            return BenchmarkCatalogDetail(
                id="benchmark-1",
                name="HLE",
                description=None,
                source_type="osmosis_managed",
                source_ref="hle",
                task_count=2_500,
                category_count=30,
                task_sets=[],
                runner_family="harbor",
                supports_harness=True,
                requires_harness=True,
                requires_judge_model=True,
                judge_model_default="openai/gpt-5",
                pass_threshold=1,
                categories=[],
                tasks=[],
                unavailable_tasks=None,
                leaderboard=leaderboard,
            )

        def list_benchmark_runs(self, **_: Any) -> PaginatedBenchmarkRuns:
            return PaginatedBenchmarkRuns(
                benchmark_runs=[
                    BenchmarkRun.from_dict(
                        {
                            "id": "run-1",
                            "name": "warm-gull",
                            "status": "finished",
                            "benchmark": {"id": "benchmark-1", "name": "HLE"},
                            "agent_count": 1,
                            "best_pass_at_1": 0.75,
                            "ingested_results": 498,
                            "expected_results": 498,
                            "created_at": "2026-08-01T00:00:00Z",
                        }
                    )
                ],
                total_count=12,
                has_more=True,
                next_offset=10,
            )

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info("hle", limit=DEFAULT_PAGE_SIZE, all_=False)

    assert result.data["leaderboard"] == leaderboard
    assert result.data["runs_total_count"] == 12
    assert result.data["runs"][0]["name"] == "warm-gull"
    fields = {field.label: field.value for field in result.fields}
    assert fields["Runs"] == "12"
    assert len(result.sections) == 2
    assert result.sections[0].plain_lines[0] == "Leaderboard:"
    assert any("GPT-5.5 (codex)" in line for line in result.sections[0].plain_lines)
    assert result.sections[1].plain_lines[0] == "Runs (1 of 12):"
    assert not any(
        "No leaderboard entries yet" in hint for hint in result.display_hints
    )


def test_leaderboard_rows_format_and_tolerate_sparse_entries() -> None:
    full, sparse = benchmark_module._leaderboard_rows(
        [
            {
                "rank": 1,
                "tied": False,
                "task_set": "parity",
                "harness": "codex",
                "model": "GPT-5.5",
                "pass_at_1": {"value": 0.75, "ci_low": 0.719, "ci_high": 0.781},
                "pass_at_k": [
                    {"k": 2, "value": 0.812, "ci_low": 0.77, "ci_high": 0.85}
                ],
                "reported_cost_usd": 4.2,
                "mean_duration_seconds": 54,
                "run": {"name": "warm-gull"},
            },
            {"rank": "not-a-rank", "tied": True, "task_set": "full"},
        ]
    )

    label, value = full
    assert label == "#1"
    assert "GPT-5.5 (codex) [parity]" in value
    assert "pass@1 75.0% ± 3.1%" in value
    assert "pass@2 81.2%" in value
    assert "$4.20" in value
    assert "54s/task" in value
    assert "run warm-gull" in value

    label, value = sparse
    assert label == "–"
    assert value.startswith("– [tied]")
    assert "pass@1 –" in value


def test_catalog_status_prefers_activity_over_sync_state() -> None:
    def entry(**overrides: Any) -> BenchmarkCatalogEntry:
        return BenchmarkCatalogEntry(
            id="benchmark-1",
            name="HLE",
            description=None,
            source_type="osmosis_managed",
            source_ref="hle",
            task_count=1,
            category_count=1,
            task_sets=[],
            **overrides,
        )

    assert benchmark_module._catalog_status(entry(running_count=2)) == "Running (2)"
    assert (
        benchmark_module._catalog_status(entry(running_count=1, sync_status="failed"))
        == "Running (1)"
    )
    assert benchmark_module._catalog_status(entry(sync_status="pending")) == "Syncing"
    assert benchmark_module._catalog_status(entry(sync_status="syncing")) == "Syncing"
    assert (
        benchmark_module._catalog_status(entry(sync_status="failed")) == "Sync failed"
    )
    assert benchmark_module._catalog_status(entry()) == "Ready"


def test_benchmark_run_rows_render_status_progress_and_date() -> None:
    run = BenchmarkRun.from_dict(
        {
            "id": "run-1",
            "name": "warm-gull",
            "status": "finished",
            "benchmark": {"id": "benchmark-1", "name": "HLE"},
            "best_pass_at_1": 0.75,
            "ingested_results": 498,
            "expected_results": 498,
            "created_at": "2026-08-01T00:00:00Z",
        }
    )

    [(label, value)] = benchmark_module._benchmark_run_rows([run])

    assert label == "warm-gull"
    assert "best pass@1 75.0%" in value
    assert "498" in value

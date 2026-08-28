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
        workspace_name=None,
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
                        run_count=14,
                        last_run_at="2026-08-01T00:00:00Z",
                        last_run_status="finished",
                        last_run_name="brave-otter",
                        creator_name="Brian",
                    )
                ],
                total_count=1,
                has_more=False,
                next_offset=None,
            )

    monkeypatch.setattr(
        benchmark_module,
        "require_platform_workspace_context",
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
    assert result.display_items[0]["task_count"] == "2,500"
    last_run = result.display_items[0]["last_run"]
    assert "Finished" in last_run
    assert "ago" in last_run
    assert last_run.endswith("brave-otter")
    assert result.items[0]["run_count"] == 14
    assert result.items[0]["last_run_status"] == "finished"
    assert result.items[0]["last_run_name"] == "brave-otter"
    assert result.items[0]["creator_name"] == "Brian"
    assert [column.key for column in result.columns] == [
        "name",
        "key",
        "last_run",
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
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_platform_workspace_context",
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
    fields = {field.label: field.value for field in result.fields}
    assert fields["Key"] == "hle"
    assert 'task_set = "parity"' in result.display_hints[0]
    assert result.sections[0].plain_lines == [
        "",
        "Leaderboard:",
        "No eligible benchmark runs. Rankings will appear here once a run "
        "finishes on the full dataset or the parity sample with scores.",
        "",
    ]
    assert not any(
        "No eligible benchmark runs" in hint for hint in result.display_hints
    )
    assert any("Omit [tasks]" in hint for hint in result.display_hints)
    assert calls == [
        {
            "name_or_id": "HLE",
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
    ]


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
        "require_platform_workspace_context",
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

    assert result.display_items[0]["task_count"] == "–"
    assert "1,000 / 4,000 tasks" in result.display_items[0]["last_run"]
    assert "Syncing" in result.display_items[0]["last_run"]
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
    assert "Task list unavailable." in result.display_items[0]["last_run"]
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
        "require_platform_workspace_context",
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
    assert result.sections[0].plain_lines == [
        "",
        "Leaderboard:",
        "No eligible benchmark runs. Rankings will appear here once a "
        "run finishes on the full dataset with scores.",
        "",
    ]
    assert not any(
        "No eligible benchmark runs" in hint for hint in result.display_hints
    )
    assert not any("parity sample" in line for line in result.sections[0].plain_lines)


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
        "require_platform_workspace_context",
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
        "require_platform_workspace_context",
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
        "require_platform_workspace_context",
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
        "No eligible benchmark runs" in hint for hint in result.display_hints
    )


def test_leaderboard_section_empty_state_omits_parity_when_not_ranked() -> None:
    section = benchmark_module._leaderboard_section([], parity_ranks=False)

    assert section.plain_lines == [
        "",
        "Leaderboard:",
        "No eligible benchmark runs. Rankings will appear here once a "
        "run finishes on the full dataset with scores.",
        "",
    ]


def test_leaderboard_section_empty_state_mentions_parity_when_ranked() -> None:
    section = benchmark_module._leaderboard_section([], parity_ranks=True)

    assert section.plain_lines == [
        "",
        "Leaderboard:",
        "No eligible benchmark runs. Rankings will appear here once a "
        "run finishes on the full dataset or the parity sample with scores.",
        "",
    ]


def test_leaderboard_section_format_and_tolerate_sparse_entries() -> None:
    section = benchmark_module._leaderboard_section(
        [
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
                },
                "pass_at_k": [
                    {
                        "k": 2,
                        "value": 0.812,
                        "ci_low": 0.77,
                        "ci_high": 0.85,
                        "n": 249,
                    }
                ],
                "cost_per_task": 4.2,
                "mean_duration_seconds": 54,
                "tokens_per_task": 1_100_000,
                "run": {"name": "warm-gull"},
            },
            {"rank": "not-a-rank", "tied": True, "task_set": "full"},
        ]
    )

    assert section.plain_lines[0] == "Leaderboard:"
    full = section.plain_lines[1]
    assert full.startswith("#1 · GPT-5.5 (codex) [parity]")
    assert "pass@1 75.0% (71.9–78.1)" in full
    assert "pass@2 81.2%" in full
    assert "$4.20/task" in full
    assert "54s/task" in full
    assert "1.1M tokens/task" in full
    assert "run warm-gull" not in full
    assert not full.startswith("#1*")  # leader is not marked tied

    sparse = section.plain_lines[2]
    assert sparse.startswith("–* · –")
    assert "pass@1 –" in sparse
    assert (plain_lines[-1] if (plain_lines := section.plain_lines) else "").startswith(
        "* tied for first"
    )
    # Rich table should expose the dynamic Pass@k header + caption.
    assert any(getattr(col, "header", None) == "Pass@2" for col in section.rich.columns)
    assert section.rich.caption is not None
    assert "tied for first" in str(section.rich.caption)


def test_last_run_cell_shows_sync_state_before_run_history() -> None:
    def entry(**overrides: Any) -> BenchmarkCatalogEntry:
        fields: dict[str, Any] = {
            "id": "benchmark-1",
            "name": "HLE",
            "description": None,
            "source_type": "osmosis_managed",
            "source_ref": "hle",
            "task_count": 1,
            "category_count": 1,
            "task_sets": [],
        }
        return BenchmarkCatalogEntry(**{**fields, **overrides})

    syncing = benchmark_module._last_run_cell(
        entry(sync_status="syncing", task_count=4_000, synced_task_count=1_000)
    )
    assert "Syncing" in syncing
    assert "1,000 / 4,000 tasks" in syncing

    queued = benchmark_module._last_run_cell(entry(sync_status="pending"))
    assert "Queued" in queued
    assert "Waiting to start" in queued

    failed = benchmark_module._last_run_cell(
        entry(sync_status="failed", sync_error="Registry unreachable.")
    )
    assert "Failed" in failed
    assert "Registry unreachable." in failed

    assert benchmark_module._last_run_cell(entry()) == "No benchmark runs yet"

    ran = benchmark_module._last_run_cell(
        entry(
            last_run_at="2026-08-01T00:00:00Z",
            last_run_status="running",
            last_run_name="calm-yak",
        )
    )
    assert "Running" in ran
    assert ran.endswith("calm-yak")


def test_benchmark_runs_section_render_status_progress_and_date() -> None:
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

    section = benchmark_module._benchmark_runs_section(
        [run],
        shown=1,
        total=12,
    )

    assert section is not None
    assert section.plain_lines[0] == "Runs (1 of 12):"
    line = section.plain_lines[1]
    assert line.startswith("warm-gull · ")
    assert "[finished]" in line
    assert "best pass@1 75.0%" in line
    assert "498" in line
    assert [col.header for col in section.rich.columns] == [
        "Name",
        "Status",
        "Progress",
        "Best Pass@1",
        "Submitted",
    ]


@pytest.mark.parametrize(
    ("requires_judge_model", "requires_judge_api_key", "default", "expected"),
    [
        (False, False, None, "–"),
        (True, False, "openai/gpt-5", "Required (default: openai/gpt-5)"),
        (False, True, None, "API key only (pinned grader)"),
    ],
)
def test_benchmark_info_judge_row_covers_every_grader_shape(
    monkeypatch: pytest.MonkeyPatch,
    requires_judge_model: bool,
    requires_judge_api_key: bool,
    default: str | None,
    expected: str,
) -> None:
    """A benchmark can pin its own grader, needing the key but no model."""

    class FakeClient:
        def get_benchmark(self, *_: Any, **__: Any) -> BenchmarkCatalogDetail:
            return BenchmarkCatalogDetail(
                id="benchmark-judge",
                name="BrowseComp",
                description=None,
                source_type="osmosis_managed",
                source_ref="browsecomp",
                task_count=10,
                category_count=0,
                task_sets=[],
                runner_family="harbor",
                supports_harness=True,
                requires_harness=False,
                requires_judge_model=requires_judge_model,
                requires_judge_api_key=requires_judge_api_key,
                judge_model_default=default,
                pass_threshold=1,
                categories=[],
                tasks=[],
                unavailable_tasks=None,
            )

        list_benchmark_runs = staticmethod(_empty_runs_page)

    monkeypatch.setattr(
        benchmark_module,
        "require_platform_workspace_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.benchmark_info(
        "browsecomp", limit=DEFAULT_PAGE_SIZE, all_=False
    )

    fields = {field.label: field.value for field in result.fields}
    assert fields["LLM Judge"] == expected
    # Agents read the JSON, not the rendered row.
    assert result.data["benchmark"]["requires_judge_api_key"] is requires_judge_api_key

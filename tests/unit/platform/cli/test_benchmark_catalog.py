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
    BenchmarkTaskSet,
    PaginatedBenchmarks,
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
    assert result.items[0]["task_sets"] == [
        {
            "name": "parity",
            "task_count": 249,
            "recommended": True,
            "description": "Published comparison sample.",
        }
    ]
    assert result.display_items[0]["task_sets"] == "parity (249, recommended)"
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
                requires_harness=False,
                requires_judge_model=True,
                judge_model_default="openai/gpt-5",
                pass_threshold=1,
                categories=[
                    BenchmarkCategory(name="math", task_count=1),
                    BenchmarkCategory(name="science", task_count=1),
                ],
                tasks=[
                    {"name": "hle__math", "category": "math"},
                    {"name": "hle__science", "category": "science"},
                ],
                unavailable_tasks=None,
            )

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        _context,
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)

    result = benchmark_module.info("HLE")

    assert isinstance(result, DetailResult)
    assert result.data["benchmark"]["tasks"] == [
        {"name": "hle__math", "category": "math"},
        {"name": "hle__science", "category": "science"},
    ]
    assert result.data["benchmark"]["categories"] == [
        {"name": "math", "task_count": 1},
        {"name": "science", "task_count": 1},
    ]
    assert 'task_set = "parity"' in result.display_hints[0]
    assert "Omit [tasks]" in result.display_hints[1]
    assert calls == [
        {
            "name_or_id": "HLE",
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
    ]

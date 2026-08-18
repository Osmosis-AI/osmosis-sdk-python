"""Data models for Platform CLI API responses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# ── Dataset status constants ─────────────────────────────────────
# Single source of truth for status classification.

STATUSES_SUCCESS: frozenset[str] = frozenset({"uploaded"})
# "pending" waits (amber); "uploading"/"processing" are active work (blue).
STATUSES_PENDING: frozenset[str] = frozenset({"pending"})
STATUSES_ACTIVE: frozenset[str] = frozenset({"uploading", "processing"})
STATUSES_IN_PROGRESS: frozenset[str] = STATUSES_PENDING | STATUSES_ACTIVE
STATUSES_ERROR: frozenset[str] = frozenset({"error"})
STATUSES_INACTIVE: frozenset[str] = frozenset({"cancelled"})


@dataclass
class UploadInfo:
    """Upload instructions returned by the create-dataset endpoint."""

    method: Literal["simple", "multipart"]
    # simple upload fields
    presigned_url: str | None = None
    expires_in: int | None = None
    upload_headers: dict[str, str] | None = None
    # multipart upload fields
    part_size: int | None = None
    total_parts: int | None = None
    presigned_urls: list[dict[str, Any]] | None = None  # [{part_number, presigned_url}]

    VALID_METHODS = {"simple", "multipart"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadInfo:
        method = data.get("method", "simple")
        if method not in cls.VALID_METHODS:
            raise ValueError(
                f"Unknown upload method {method!r}. "
                f"Expected one of: {', '.join(sorted(cls.VALID_METHODS))}"
            )
        return cls(
            method=method,
            presigned_url=data.get("presigned_url"),
            expires_in=data.get("expires_in"),
            upload_headers=data.get("upload_headers"),
            part_size=data.get("part_size"),
            total_parts=data.get("total_parts"),
            presigned_urls=data.get("presigned_urls"),
        )


@dataclass
class DatasetFile:
    """A dataset record."""

    id: str
    file_name: str
    file_size: int
    status: str
    data_preview: Any = None
    file_format: str | None = None
    original_file_format: str | None = None
    row_count: int | None = None
    original_file_size: int | None = None
    creator_name: str | None = None
    created_at: str = ""
    updated_at: str = ""
    platform_url: str | None = None
    is_internal_user: bool = False
    # Upload info — only present in create_dataset response
    upload: UploadInfo | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFile:
        upload_data = data.get("upload")
        upload = UploadInfo.from_dict(upload_data) if upload_data else None
        return cls(
            id=data["id"],
            file_name=data.get("file_name", ""),
            file_size=data.get("file_size", 0),
            status=data.get("status", ""),
            data_preview=data.get("data_preview"),
            file_format=data.get("file_format"),
            original_file_format=data.get("original_file_format"),
            row_count=data.get("row_count"),
            original_file_size=data.get("original_file_size"),
            creator_name=data.get("creator_name"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            platform_url=data.get("platform_url"),
            is_internal_user=data.get("is_internal_user", False),
            upload=upload,
        )


@dataclass
class DatasetDownloadInfo:
    """Download instructions returned by the dataset download endpoint."""

    presigned_url: str
    expires_in: int | None = None
    file_name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetDownloadInfo:
        expires_in = data.get("expires_in")
        if expires_in is None:
            expires_in = data.get("expiresIn")

        return cls(
            presigned_url=data.get("presigned_url") or data["presignedUrl"],
            expires_in=expires_in,
            file_name=(
                data.get("file_name")
                or data.get("fileName")
                or data.get("download_file_name")
                or data.get("downloadFileName")
            ),
        )


@dataclass
class PaginatedDatasets:
    """Paginated list of datasets."""

    datasets: list[DatasetFile]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedDatasets:
        return cls(
            datasets=[DatasetFile.from_dict(d) for d in data.get("datasets", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


# ── Training run status constants ────────────────────────────────

RUN_STATUSES_SUCCESS: frozenset[str] = frozenset({"finished"})
# "pending"/"queued" wait (amber); "running" is in-progress work (blue).
RUN_STATUSES_PENDING: frozenset[str] = frozenset({"pending", "queued"})
RUN_STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"running"})
RUN_STATUSES_ERROR: frozenset[str] = frozenset({"failed", "crashed"})
# "unknown" is a terminal, greyed-out state alongside stopped/killed.
RUN_STATUSES_STOPPED: frozenset[str] = frozenset({"stopped", "killed", "unknown"})
RUN_STATUSES_TERMINAL: frozenset[str] = (
    RUN_STATUSES_SUCCESS | RUN_STATUSES_ERROR | RUN_STATUSES_STOPPED
)

# ── Evaluation run status constants ──────────────────────────────

EVAL_RUN_STATUSES_SUCCESS: frozenset[str] = frozenset({"finished"})
# "pending" waits (amber); "running" is in-progress work (blue).
EVAL_RUN_STATUSES_PENDING: frozenset[str] = frozenset({"pending"})
EVAL_RUN_STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"running"})
EVAL_RUN_STATUSES_ERROR: frozenset[str] = frozenset({"failed"})
EVAL_RUN_STATUSES_STOPPED: frozenset[str] = frozenset({"stopped"})
EVAL_RUN_STATUSES_TERMINAL: frozenset[str] = (
    EVAL_RUN_STATUSES_SUCCESS | EVAL_RUN_STATUSES_ERROR | EVAL_RUN_STATUSES_STOPPED
)

# ── Benchmark run status constants ───────────────────────────────

BENCHMARK_RUN_STATUSES_SUCCESS: frozenset[str] = frozenset({"finished"})
BENCHMARK_RUN_STATUSES_PENDING: frozenset[str] = frozenset({"pending", "queued"})
BENCHMARK_RUN_STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"running"})
BENCHMARK_RUN_STATUSES_ERROR: frozenset[str] = frozenset({"failed"})
BENCHMARK_RUN_STATUSES_STOPPED: frozenset[str] = frozenset({"stopped"})
BENCHMARK_RUN_STATUSES_TERMINAL: frozenset[str] = (
    BENCHMARK_RUN_STATUSES_SUCCESS
    | BENCHMARK_RUN_STATUSES_ERROR
    | BENCHMARK_RUN_STATUSES_STOPPED
)


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return value


@dataclass
class TrainingRun:
    """A training run in a workspace."""

    id: str
    name: str | None
    status: str
    model_id: str | None = None
    model_name: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    creator_name: str | None = None
    creator_email: str | None = None
    platform_url: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    rollout_id: str | None = None
    rollout_name: str | None = None
    current_step: int | None = None
    total_steps: int | None = None
    reward: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRun:
        model = data.get("model") or {}
        dataset = data.get("dataset") or {}
        rollout = data.get("rollout") or {}
        current_step = _number_or_none(data.get("current_step"))
        total_steps = _number_or_none(data.get("total_steps"))
        reward = _number_or_none(data.get("reward"))
        return cls(
            id=data["id"],
            name=data.get("name"),
            status=data.get("status", ""),
            model_id=model.get("id"),
            model_name=model.get("model_name"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            creator_name=data.get("creator_name"),
            creator_email=data.get("creator_email"),
            platform_url=data.get("platform_url"),
            dataset_id=dataset.get("id"),
            dataset_name=dataset.get("file_name"),
            rollout_id=rollout.get("id"),
            rollout_name=rollout.get("name"),
            current_step=int(current_step) if current_step is not None else None,
            total_steps=int(total_steps) if total_steps is not None else None,
            reward=float(reward) if reward is not None else None,
        )


@dataclass
class TrainingRunDetail(TrainingRun):
    """Detailed training run info with additional fields."""

    examples_processed_count: int | None = None
    notes: str | None = None
    config: dict[str, Any] | None = None
    entrypoint: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    env_config: dict[str, Any] | None = None
    resolved_secret_scopes: dict[str, Any] | None = None
    is_internal_user: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunDetail:
        # Detail API returns unified entity refs for model, dataset, and rollout.
        run = data["training_run"]
        model = data.get("model") or {}
        dataset = data.get("dataset") or {}
        rollout = data.get("rollout") or {}
        current_step = _number_or_none(run.get("current_step"))
        total_steps = _number_or_none(run.get("total_steps"))
        reward = _number_or_none(run.get("reward"))
        config = data.get("config")
        env_config = data.get("env_config")
        resolved_secret_scopes = data.get("resolved_secret_scopes")
        return cls(
            id=run["id"],
            name=run.get("name"),
            status=run.get("status", ""),
            model_id=model.get("id"),
            model_name=model.get("name"),
            created_at=run.get("created_at", ""),
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            creator_name=run.get("creator_name"),
            creator_email=run.get("creator_email"),
            platform_url=run.get("platform_url"),
            dataset_id=dataset.get("id"),
            dataset_name=dataset.get("name"),
            rollout_id=rollout.get("id"),
            rollout_name=rollout.get("name"),
            current_step=int(current_step) if current_step is not None else None,
            total_steps=int(total_steps) if total_steps is not None else None,
            reward=float(reward) if reward is not None else None,
            examples_processed_count=run.get("examples_processed_count"),
            notes=run.get("notes"),
            config=config if isinstance(config, dict) else None,
            entrypoint=data.get("entrypoint")
            if isinstance(data.get("entrypoint"), str)
            else None,
            branch=data.get("branch") if isinstance(data.get("branch"), str) else None,
            commit_sha=data.get("commit_sha")
            if isinstance(data.get("commit_sha"), str)
            else None,
            env_config=env_config if isinstance(env_config, dict) else None,
            resolved_secret_scopes=(
                resolved_secret_scopes
                if isinstance(resolved_secret_scopes, dict)
                else None
            ),
            is_internal_user=data.get("is_internal_user", False),
        )


@dataclass
class PaginatedTrainingRuns:
    """Paginated list of training runs."""

    training_runs: list[TrainingRun]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedTrainingRuns:
        return cls(
            training_runs=[
                TrainingRun.from_dict(r) for r in data.get("training_runs", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class SubmitRunResult:
    """Result of submitting a training run or evaluation run.

    Both `POST /api/cli/training-runs` and `POST /api/cli/eval-runs` return the
    same shape; this is the single response model for either submit path.
    """

    id: str
    name: str
    status: str
    created_at: str
    platform_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubmitRunResult:
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
            platform_url=data.get("platform_url"),
        )


@dataclass
class BenchmarkTaskSet:
    """A named task set exposed by a benchmark."""

    name: str
    task_count: int
    recommended: bool
    description: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkTaskSet:
        return cls(
            name=data["name"],
            task_count=data["task_count"],
            recommended=data.get("recommended", False),
            description=data.get("description"),
        )


@dataclass
class BenchmarkCategory:
    """Task count for one benchmark category."""

    name: str
    task_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCategory:
        return cls(name=data["name"], task_count=data["task_count"])


BenchmarkTaskDifficulty = Literal["easy", "medium", "hard"]


class BenchmarkCatalogTask(TypedDict):
    """One task exposed by the benchmark catalog."""

    name: str
    category: str | None
    difficulty: BenchmarkTaskDifficulty | None


def _parse_benchmark_catalog_task(data: dict[str, Any]) -> BenchmarkCatalogTask:
    difficulty = data.get("difficulty")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = None
    return {
        "name": data["name"],
        "category": data.get("category"),
        "difficulty": difficulty,
    }


def _parse_unavailable_benchmark_tasks(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    raw_tasks = data.get("tasks", [])
    tasks = (
        [
            _parse_benchmark_catalog_task(task)
            for task in raw_tasks
            if isinstance(task, dict)
        ]
        if isinstance(raw_tasks, list)
        else []
    )
    return {**data, "tasks": tasks}


@dataclass
class BenchmarkCatalogEntry:
    """Benchmark available in the current workspace catalog."""

    id: str
    name: str
    description: str | None
    source_type: str
    source_ref: str
    task_count: int
    category_count: int
    task_sets: list[BenchmarkTaskSet]
    source_url: str | None = None
    sync_status: str = "ready"
    synced_task_count: int = 0
    sync_error: str | None = None
    platform_url: str | None = None
    run_count: int = 0
    running_count: int = 0
    last_run_at: str | None = None
    last_run_status: str | None = None
    last_run_name: str | None = None
    creator_name: str | None = None

    @property
    def is_ready(self) -> bool:
        """Whether the task list has finished paging in from the registry."""
        return self.sync_status == "ready"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCatalogEntry:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            source_type=data["source_type"],
            source_ref=data["source_ref"],
            source_url=data.get("source_url"),
            task_count=data["task_count"],
            category_count=data["category_count"],
            task_sets=[
                BenchmarkTaskSet.from_dict(item) for item in data.get("task_sets", [])
            ],
            sync_status=data.get("sync_status", "ready"),
            synced_task_count=int(data.get("synced_task_count") or 0),
            sync_error=data.get("sync_error"),
            platform_url=data.get("platform_url"),
            run_count=int(data.get("run_count") or 0),
            running_count=int(data.get("running_count") or 0),
            last_run_at=data.get("last_run_at"),
            last_run_status=data.get("last_run_status"),
            last_run_name=data.get("last_run_name"),
            creator_name=data.get("creator_name"),
        )


@dataclass
class PaginatedBenchmarks:
    """Paginated workspace benchmark catalog."""

    benchmarks: list[BenchmarkCatalogEntry]
    total_count: int
    has_more: bool
    next_offset: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedBenchmarks:
        return cls(
            benchmarks=[
                BenchmarkCatalogEntry.from_dict(item)
                for item in data.get("benchmarks", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class BenchmarkCatalogDetail:
    """Detailed benchmark metadata and task-selection options."""

    id: str
    name: str
    description: str | None
    source_type: str
    source_ref: str
    task_count: int
    category_count: int
    task_sets: list[BenchmarkTaskSet]
    runner_family: str
    supports_harness: bool
    requires_harness: bool
    requires_judge_model: bool
    judge_model_default: str | None
    pass_threshold: float
    categories: list[BenchmarkCategory]
    tasks: list[BenchmarkCatalogTask]
    unavailable_tasks: dict[str, Any] | None
    requires_judge_api_key: bool = False
    default_harness: str | None = None
    source_url: str | None = None
    sync_status: str = "ready"
    synced_task_count: int = 0
    sync_error: str | None = None
    platform_url: str | None = None
    # Server-computed standings; metric shapes travel verbatim like
    # BenchmarkRunDetail.agent_metrics, so the estimator stays server-side.
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Whether the task list has finished paging in from the registry."""
        return self.sync_status == "ready"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCatalogDetail:
        benchmark = data["benchmark"]
        return cls(
            id=benchmark["id"],
            name=benchmark["name"],
            description=benchmark.get("description"),
            source_type=benchmark["source_type"],
            source_ref=benchmark["source_ref"],
            source_url=benchmark.get("source_url"),
            task_count=benchmark["task_count"],
            category_count=benchmark["category_count"],
            task_sets=[
                BenchmarkTaskSet.from_dict(item)
                for item in benchmark.get("task_sets", [])
            ],
            runner_family=benchmark["runner_family"],
            supports_harness=benchmark["supports_harness"],
            requires_harness=benchmark["requires_harness"],
            requires_judge_model=benchmark["requires_judge_model"],
            requires_judge_api_key=benchmark.get("requires_judge_api_key", False),
            judge_model_default=benchmark.get("judge_model_default"),
            pass_threshold=float(benchmark["pass_threshold"]),
            categories=[
                BenchmarkCategory.from_dict(item)
                for item in benchmark.get("categories", [])
            ],
            tasks=[
                _parse_benchmark_catalog_task(item)
                for item in benchmark.get("tasks", [])
            ],
            unavailable_tasks=_parse_unavailable_benchmark_tasks(
                benchmark.get("unavailable_tasks")
            ),
            default_harness=benchmark.get("default_harness"),
            sync_status=benchmark.get("sync_status", "ready"),
            synced_task_count=int(benchmark.get("synced_task_count") or 0),
            sync_error=benchmark.get("sync_error"),
            platform_url=benchmark.get("platform_url"),
            leaderboard=[
                item for item in data.get("leaderboard", []) if isinstance(item, dict)
            ],
        )


@dataclass
class SubmitBenchmarkRunResult:
    """Result of submitting a benchmark run."""

    id: str
    name: str
    status: str
    created_at: str
    workflow_id: str
    task_count: int
    platform_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubmitBenchmarkRunResult:
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
            workflow_id=data["workflow_id"],
            task_count=data["task_count"],
            platform_url=data.get("platform_url"),
        )


@dataclass
class BenchmarkRun:
    """A benchmark run in the current workspace."""

    id: str
    name: str
    status: str
    benchmark_name: str
    created_at: str
    benchmark_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    creator_name: str | None = None
    creator_email: str | None = None
    platform_url: str | None = None
    agent_count: int = 0
    best_pass_at_1: float | None = None
    ingested_results: int = 0
    expected_results: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRun:
        benchmark = data.get("benchmark")
        if not isinstance(benchmark, dict):
            benchmark = {}
        best_pass_at_1 = _number_or_none(data.get("best_pass_at_1"))
        return cls(
            id=data["id"],
            name=data.get("name") or data.get("benchmark_run_name", ""),
            status=data.get("status", ""),
            benchmark_name=data.get("benchmark_name") or benchmark.get("name", ""),
            benchmark_id=data.get("benchmark_id") or benchmark.get("id"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            creator_name=data.get("creator_name"),
            creator_email=data.get("creator_email"),
            platform_url=data.get("platform_url"),
            agent_count=int(data.get("agent_count") or 0),
            best_pass_at_1=(
                float(best_pass_at_1) if best_pass_at_1 is not None else None
            ),
            ingested_results=int(data.get("ingested_results") or 0),
            expected_results=int(data.get("expected_results") or 0),
        )


@dataclass
class BenchmarkRunDetail(BenchmarkRun):
    """Detailed benchmark run with configuration, agents, and result totals."""

    configuration: dict[str, Any] | None = None
    agents: list[dict[str, Any]] | None = None
    progress: dict[str, Any] | None = None
    totals: dict[str, Any] | None = None
    agent_metrics: list[dict[str, Any]] | None = None
    is_internal_user: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRunDetail:
        run = data["benchmark_run"]
        configuration = data.get("configuration")
        if not isinstance(configuration, dict):
            configuration = None
        progress = data.get("progress")
        if not isinstance(progress, dict):
            progress = None
        totals = data.get("totals")
        if not isinstance(totals, dict):
            totals = None
        raw_agents = data.get("agents")
        agents = (
            [item for item in raw_agents if isinstance(item, dict)]
            if isinstance(raw_agents, list)
            else []
        )
        raw_agent_metrics = data.get("agent_metrics")
        agent_metrics = (
            [item for item in raw_agent_metrics if isinstance(item, dict)]
            if isinstance(raw_agent_metrics, list)
            else []
        )
        benchmark_name = run.get("benchmark_name")
        if not isinstance(benchmark_name, str):
            benchmark_name = (
                configuration.get("benchmark_name", "") if configuration else ""
            )
        benchmark_id = run.get("benchmark_id")
        if not isinstance(benchmark_id, str):
            configured_id = configuration.get("benchmark_id") if configuration else None
            benchmark_id = configured_id if isinstance(configured_id, str) else None
        merged_run = {
            **run,
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_name,
        }
        base = BenchmarkRun.from_dict(merged_run)
        ingested_results = base.ingested_results
        expected_results = base.expected_results
        if progress is not None:
            ingested_results = int(progress.get("ingested") or 0)
            expected_results = int(progress.get("expected") or 0)
        pass_at_1_values = []
        for metrics in agent_metrics:
            interval = metrics.get("pass_at_1")
            if not isinstance(interval, dict):
                continue
            value = _number_or_none(interval.get("value"))
            if value is not None:
                pass_at_1_values.append(float(value))
        return cls(
            id=base.id,
            name=base.name,
            status=base.status,
            benchmark_name=base.benchmark_name,
            benchmark_id=base.benchmark_id,
            created_at=base.created_at,
            started_at=base.started_at,
            completed_at=base.completed_at,
            creator_name=base.creator_name,
            creator_email=base.creator_email,
            platform_url=base.platform_url,
            agent_count=base.agent_count or len(agents),
            best_pass_at_1=(
                base.best_pass_at_1
                if base.best_pass_at_1 is not None
                else max(pass_at_1_values, default=None)
            ),
            ingested_results=ingested_results,
            expected_results=expected_results,
            configuration=configuration,
            agents=agents,
            progress=progress,
            totals=totals,
            agent_metrics=agent_metrics,
            is_internal_user=data.get("is_internal_user", False),
        )


@dataclass
class PaginatedBenchmarkRuns:
    """Paginated benchmark runs for a workspace."""

    benchmark_runs: list[BenchmarkRun]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedBenchmarkRuns:
        return cls(
            benchmark_runs=[
                BenchmarkRun.from_dict(item) for item in data.get("benchmark_runs", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


# ── Training run metrics ─────────────────────────────────────────


@dataclass
class MetricSummary:
    """Initial, latest, and delta for a single metric."""

    key: str
    title: str
    initial: float | None
    latest: float | None
    delta: float | None
    min: float | None
    max: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricSummary:
        return cls(
            key=data["key"],
            title=data["title"],
            initial=data.get("initial"),
            latest=data.get("latest"),
            delta=data.get("delta"),
            min=data.get("min"),
            max=data.get("max"),
        )


@dataclass
class MetricDataPoint:
    """A single data point in a metric time series."""

    step: int
    value: float
    timestamp: int  # epoch ms

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDataPoint:
        return cls(
            step=data["step"],
            value=data["value"],
            timestamp=data["timestamp"],
        )


@dataclass
class MetricHistory:
    """History of a single metric across training steps."""

    metric_key: str
    title: str
    data_points: list[MetricDataPoint]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricHistory:
        return cls(
            metric_key=data["metric_key"],
            title=data["title"],
            data_points=[
                MetricDataPoint.from_dict(dp) for dp in data.get("data_points", [])
            ],
        )


@dataclass
class TrainingRunMetricsOverview:
    """Summary metrics for a training run."""

    duration_ms: int | None
    metric_summaries: list[MetricSummary]
    examples_processed_count: int | None
    latest_step: int | None = None
    total_steps: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunMetricsOverview:
        return cls(
            duration_ms=data.get("duration_ms"),
            metric_summaries=[
                MetricSummary.from_dict(s) for s in data.get("metric_summaries", [])
            ],
            examples_processed_count=data.get("examples_processed_count"),
            latest_step=data.get("latest_step"),
            total_steps=data.get("total_steps"),
        )


@dataclass
class TrainingRunMetrics:
    """Complete metrics response for a training run."""

    status: str
    overview: TrainingRunMetricsOverview
    metrics: list[MetricHistory]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunMetrics:
        return cls(
            status=data["status"],
            overview=TrainingRunMetricsOverview.from_dict(data["overview"]),
            metrics=[MetricHistory.from_dict(m) for m in data.get("metrics", [])],
        )


# ── Logs (training runs, eval runs, datasets) ────────────────────
# All three /logs endpoints share one wire shape.


@dataclass
class LogEntry:
    """A single log line from a training run, evaluation run, or dataset."""

    timestamp: str
    level: str
    step: str
    message: str
    details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        details = data.get("details")
        return cls(
            timestamp=data.get("timestamp", ""),
            level=data.get("level", ""),
            step=data.get("step", ""),
            message=data.get("message", ""),
            details=details if isinstance(details, dict) else None,
        )


@dataclass
class LogsPage:
    """One cursor page of logs.

    The server returns entries oldest-first within the page; ``next_cursor``
    continues in the requested paging direction (``None`` at the end).
    """

    logs: list[LogEntry]
    next_cursor: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogsPage:
        return cls(
            logs=[LogEntry.from_dict(log) for log in data.get("logs", [])],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class EvalRewardStats:
    """Distribution stats for per-sample rewards in an eval run."""

    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRewardStats:
        return cls(
            mean=data.get("mean"),
            median=data.get("median"),
            std=data.get("std"),
            min=data.get("min"),
            max=data.get("max"),
        )


@dataclass
class EvalPassAtKPoint:
    """A single pass@k point (probability of a pass within k attempts)."""

    k: int
    value: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalPassAtKPoint:
        return cls(k=data["k"], value=data["value"])


@dataclass
class EvalRunMetricsOverview:
    """Summary metrics for an evaluation run."""

    duration_ms: int | None
    total_samples: int | None
    completed_samples: int | None
    graded: int | None
    passed: int | None
    failed: int | None
    skipped: int | None
    pass_rate: float | None
    pass_threshold: float | None
    tokens_used: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunMetricsOverview:
        return cls(
            duration_ms=data.get("duration_ms"),
            total_samples=data.get("total_samples"),
            completed_samples=data.get("completed_samples"),
            graded=data.get("graded"),
            passed=data.get("passed"),
            failed=data.get("failed"),
            skipped=data.get("skipped"),
            pass_rate=data.get("pass_rate"),
            pass_threshold=data.get("pass_threshold"),
            tokens_used=data.get("tokens_used"),
        )


@dataclass
class EvalRunMetrics:
    """Complete metrics response for an evaluation run."""

    status: str
    overview: EvalRunMetricsOverview
    reward_stats: EvalRewardStats | None
    pass_at_k: list[EvalPassAtKPoint]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunMetrics:
        reward_stats = data.get("reward_stats")
        return cls(
            status=data["status"],
            overview=EvalRunMetricsOverview.from_dict(data["overview"]),
            reward_stats=(
                EvalRewardStats.from_dict(reward_stats)
                if reward_stats is not None
                else None
            ),
            pass_at_k=[
                EvalPassAtKPoint.from_dict(p) for p in data.get("pass_at_k", [])
            ],
        )


@dataclass
class BaseModelInfo:
    """A base (foundation) model record."""

    id: str
    model_name: str
    base_model: str | None = None
    creator_name: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModelInfo:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            creator_name=data.get("creator_name"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class PaginatedBaseModels:
    """Paginated list of base models."""

    models: list[BaseModelInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedBaseModels:
        return cls(
            models=[BaseModelInfo.from_dict(m) for m in data.get("models", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class LoraModelInfo:
    """A LoRA model produced by a training run."""

    id: str
    model_name: str
    base_model: str | None = None
    training_run_name: str | None = None
    checkpoint_step: int | None = None
    reward: float | None = None
    deployment_status: str | None = None
    deployed_at: str | None = None
    deployed_by: str | None = None
    created_at: str = ""
    # The platform omits deployment fields entirely when inference is
    # unavailable for the account.
    has_deployment_info: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraModelInfo:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            training_run_name=data.get("training_run_name"),
            checkpoint_step=data.get("checkpoint_step"),
            reward=data.get("reward"),
            deployment_status=data.get("deployment_status"),
            deployed_at=data.get("deployed_at"),
            deployed_by=data.get("deployed_by"),
            created_at=data.get("created_at", ""),
            has_deployment_info="deployment_status" in data,
        )


@dataclass
class LoraModelDetail(LoraModelInfo):
    """Detailed LoRA model info with Hugging Face export and platform link."""

    hf_upload_status: str = ""
    hf_url: str | None = None
    uploaded_by: str | None = None
    # Canonical `model` value for the inference API
    # ("<base_model_path>:<lora-model-name>").
    inference_model: str | None = None
    platform_url: str | None = None
    is_internal_user: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraModelDetail:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            training_run_name=data.get("training_run_name"),
            checkpoint_step=data.get("checkpoint_step"),
            reward=data.get("reward"),
            deployment_status=data.get("deployment_status"),
            deployed_at=data.get("deployed_at"),
            deployed_by=data.get("deployed_by"),
            created_at=data.get("created_at", ""),
            hf_upload_status=data.get("hf_upload_status", ""),
            hf_url=data.get("hf_url"),
            uploaded_by=data.get("uploaded_by"),
            has_deployment_info="deployment_status" in data,
            inference_model=data.get("inference_model"),
            platform_url=data.get("platform_url"),
            is_internal_user=data.get("is_internal_user", False),
        )


@dataclass
class PaginatedLoraModels:
    """Paginated list of LoRA models."""

    models: list[LoraModelInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None
    active_deployments: int = 0
    max_active_deployments: int = 0
    # The platform omits the deployment quota fields when inference is
    # unavailable for the account.
    has_deployment_info: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedLoraModels:
        return cls(
            models=[LoraModelInfo.from_dict(m) for m in data.get("models", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
            active_deployments=data.get("active_deployments", 0),
            max_active_deployments=data.get("max_active_deployments", 0),
            has_deployment_info="active_deployments" in data,
        )


@dataclass
class LoraModelSummary:
    """Minimal LoRA model identity returned from deploy/undeploy endpoints."""

    id: str
    model_name: str
    status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraModelSummary:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            status=data.get("status", ""),
        )


# ── Environment Secrets ──────────────────────────────────────────
# Workspace-scoped secrets. The platform never returns the secret *value*:
# list responses carry names + metadata only, and create responses carry
# only metadata. These models therefore have no field for the value — there
# is intentionally nowhere for a value to land if one were ever returned.

# The platform wire value for a personal secret's scope is "user"; the
# user-facing vocabulary calls it "personal". Both the wire value and the
# display value are part of the stable JSON/API contract — keep them exact.
WIRE_SCOPE_PERSONAL = "user"
DISPLAY_SCOPE_PERSONAL = "personal"


def wire_to_display_scope(scope: str | None) -> str | None:
    """Map a wire scope value to its user-facing display value.

    Only the personal scope differs ("user" → "personal"); every other value
    (including ``"workspace"`` and ``None``) passes through unchanged.
    """
    return DISPLAY_SCOPE_PERSONAL if scope == WIRE_SCOPE_PERSONAL else scope


@dataclass
class EnvironmentSecretInfo:
    """A workspace environment secret record (metadata only — never the value)."""

    id: str
    name: str
    created_at: str = ""
    updated_at: str = ""
    creator_name: str | None = None
    updater_name: str | None = None
    # "workspace" or "user". None when the platform did not report it
    # (older responses / endpoints that don't distinguish scope).
    scope: str | None = None
    # Page/operation-level link to the secrets console page. Populated by the
    # platform on create (and exposed at the list level via
    # ``PaginatedEnvironmentSecrets.platform_url``); ``None`` for list items.
    platform_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentSecretInfo:
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            creator_name=data.get("creator_name"),
            updater_name=data.get("updater_name"),
            scope=data.get("scope"),
            platform_url=data.get("platform_url"),
        )


@dataclass
class PaginatedEnvironmentSecrets:
    """Paginated list of environment secrets (names + metadata only)."""

    environment_secrets: list[EnvironmentSecretInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None
    platform_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedEnvironmentSecrets:
        return cls(
            environment_secrets=[
                EnvironmentSecretInfo.from_dict(s)
                for s in data.get("environment_secrets", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
            platform_url=data.get("platform_url"),
        )


# ── Deployment status ────────────────────────────────────────────
# Lifecycle of a LoRA model's deployment: "active" / "inactive".

DEPLOYMENT_STATUSES_SUCCESS: frozenset[str] = frozenset({"active"})


# ── Rollouts ─────────────────────────────────────────────────────


@dataclass
class RolloutInfo:
    """A rollout record."""

    id: str
    name: str
    description: str | None = None
    is_active: bool = True
    last_synced_commit_sha: str | None = None
    repo_full_name: str | None = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutInfo:
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description"),
            is_active=data.get("is_active", True),
            last_synced_commit_sha=data.get("last_synced_commit_sha"),
            repo_full_name=data.get("repo_full_name"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class PaginatedRollouts:
    """Paginated list of rollouts."""

    rollouts: list[RolloutInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedRollouts:
        return cls(
            rollouts=[RolloutInfo.from_dict(r) for r in data.get("rollouts", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class DevRolloutServerInfo:
    """A dev rollout server record."""

    id: str
    name: str
    url: str
    status: str
    expires_at: str | None = None
    started_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevRolloutServerInfo:
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            url=data.get("url", ""),
            status=data.get("status", ""),
            expires_at=data.get("expires_at"),
            started_at=data.get("started_at"),
        )


@dataclass
class PaginatedDevRolloutServers:
    """Paginated list of dev rollout servers."""

    dev_rollout_servers: list[DevRolloutServerInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedDevRolloutServers:
        return cls(
            dev_rollout_servers=[
                DevRolloutServerInfo.from_dict(s)
                for s in data.get("dev_rollout_servers", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


# ── LoRA checkpoints (for `osmosis train info`) ──────────────────


@dataclass
class LoraCheckpointInfo:
    """A LoRA checkpoint produced by a training run."""

    id: str
    checkpoint_step: int
    status: str
    checkpoint_name: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraCheckpointInfo:
        return cls(
            id=data["id"],
            checkpoint_step=data.get("checkpoint_step", 0),
            status=data.get("status", ""),
            checkpoint_name=data.get("checkpoint_name", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class TrainingRunCheckpoints:
    """All deployable LoRA checkpoints for a training run."""

    training_run_name: str
    checkpoints: list[LoraCheckpointInfo]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunCheckpoints:
        return cls(
            training_run_name=data.get("training_run_name", ""),
            checkpoints=[
                LoraCheckpointInfo.from_dict(c) for c in data.get("checkpoints", [])
            ],
        )


# ── Evaluation Runs ──────────────────────────────────────────────


@dataclass
class EvaluationRun:
    """An evaluation run record."""

    id: str
    name: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    model: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    rollout: dict[str, Any] | None = None
    creator_name: str | None = None
    creator_email: str | None = None
    platform_url: str | None = None
    results: dict[str, Any] | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        config = data.get("config")
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            model=data.get("model"),
            dataset=data.get("dataset"),
            rollout=data.get("rollout"),
            creator_name=data.get("creator_name"),
            creator_email=data.get("creator_email"),
            platform_url=data.get("platform_url"),
            results=data.get("results"),
            config=config if isinstance(config, dict) else None,
        )


@dataclass
class EvaluationRunDetail(EvaluationRun):
    """Detailed evaluation run info.

    Mirrors :class:`TrainingRunDetail`: a typed subclass of the list row so
    callers read ``detail.status`` / ``detail.name`` with static safety instead
    of stringly-typed ``eval_run.get(...)`` lookups.
    """

    config: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    entrypoint: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    env_config: dict[str, Any] | None = None
    resolved_secret_scopes: dict[str, Any] | None = None
    dataset_df_stats: dict[str, Any] | None = None
    is_internal_user: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRunDetail:
        run = data["eval_run"]
        config = data.get("config")
        model_path = config.get("model_path") if isinstance(config, dict) else None
        env_config = data.get("env_config")
        resolved_secret_scopes = data.get("resolved_secret_scopes")
        dataset_df_stats = data.get("dataset_df_stats")
        return cls(
            id=run["id"],
            name=run.get("name", ""),
            status=run.get("status", ""),
            created_at=run.get("created_at", ""),
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            creator_name=run.get("creator_name"),
            creator_email=run.get("creator_email"),
            platform_url=run.get("platform_url"),
            model={"name": model_path} if isinstance(model_path, str) else None,
            dataset=data.get("dataset"),
            rollout=data.get("rollout"),
            config=config,
            results=data.get("results"),
            entrypoint=data.get("entrypoint")
            if isinstance(data.get("entrypoint"), str)
            else None,
            branch=data.get("branch") if isinstance(data.get("branch"), str) else None,
            commit_sha=data.get("commit_sha")
            if isinstance(data.get("commit_sha"), str)
            else None,
            env_config=env_config if isinstance(env_config, dict) else None,
            resolved_secret_scopes=(
                resolved_secret_scopes
                if isinstance(resolved_secret_scopes, dict)
                else None
            ),
            dataset_df_stats=(
                dataset_df_stats if isinstance(dataset_df_stats, dict) else None
            ),
            is_internal_user=data.get("is_internal_user", False),
        )


@dataclass
class PaginatedEvaluationRuns:
    """Paginated list of evaluation runs."""

    eval_runs: list[EvaluationRun]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedEvaluationRuns:
        return cls(
            eval_runs=[EvaluationRun.from_dict(r) for r in data.get("eval_runs", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass(frozen=True)
class RunDownloadFile:
    """One run-scoped file returned by a samples manifest endpoint.

    ``path`` is the fixed local path relative to the run output root. The
    optional ``token`` is an opaque server handle (a rollout id or an export
    snapshot token) echoed back with ``path`` when requesting a presigned
    URL; the server remains solely responsible for deriving S3 keys.
    """

    path: str
    size: int
    token: str | None = None

    @property
    def identity(self) -> tuple[str | None, str]:
        return self.token, self.path

    def to_request_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {"path": self.path}
        if self.token is not None:
            item["token"] = self.token
        return item

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDownloadFile:
        path = data.get("path")
        size = data.get("size")
        token = data.get("token")
        if not isinstance(path, str) or not path:
            raise ValueError("Download manifest file path must be a non-empty string")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Download manifest size is invalid for {path!r}")
        if token is not None and not isinstance(token, str):
            raise ValueError(f"Download manifest token is invalid for {path!r}")
        return cls(path=path, size=size, token=token)


@dataclass(frozen=True)
class RunDownloadManifest:
    """Complete file manifest for an evaluation run download."""

    files: list[RunDownloadFile]
    totals: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDownloadManifest:
        raw_files = data.get("files", [])
        if not isinstance(raw_files, list):
            raise ValueError("Download manifest files must be a list")
        totals = data.get("totals", {})
        return cls(
            files=[RunDownloadFile.from_dict(item) for item in raw_files],
            totals=totals if isinstance(totals, dict) else {},
        )


@dataclass(frozen=True)
class RunDownloadURL:
    """Presigned URL for one manifest item."""

    path: str
    url: str
    token: str | None = None

    @property
    def identity(self) -> tuple[str | None, str]:
        return self.token, self.path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDownloadURL:
        path = data.get("path")
        url = data.get("url") or data.get("presignedUrl") or data.get("presigned_url")
        token = data.get("token")
        if not isinstance(path, str) or not path:
            raise ValueError("Download URL path must be a non-empty string")
        if not isinstance(url, str) or not url:
            raise ValueError(f"Download URL is missing for {path!r}")
        if token is not None and not isinstance(token, str):
            raise ValueError(f"Download URL token is invalid for {path!r}")
        return cls(path=path, url=url, token=token)


@dataclass(frozen=True)
class RunDownloadURLBatch:
    """Presigned URLs returned for one bounded manifest batch."""

    items: list[RunDownloadURL]
    expires_in: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDownloadURLBatch:
        raw_items = data.get("items")
        if raw_items is None:
            raw_items = data.get("files", data.get("urls", []))
        if not isinstance(raw_items, list):
            raise ValueError("Download URL response items must be a list")
        expires_in = data.get("expires_in")
        return cls(
            items=[RunDownloadURL.from_dict(item) for item in raw_items],
            expires_in=expires_in if isinstance(expires_in, int) else None,
        )


# ── Workspaces ───────────────────────────────────────────────────


@dataclass
class WorkspaceSummary:
    """One workspace the caller belongs to."""

    id: str
    name: str
    connected_repo_full_name: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceSummary:
        repo = data.get("connected_repo")
        full_name = repo.get("repo_full_name") if isinstance(repo, dict) else None
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            connected_repo_full_name=full_name if isinstance(full_name, str) else None,
        )


# ── Quickstart ───────────────────────────────────────────────────


@dataclass
class QuickstartStatus:
    """Setup state of one workspace, as read by the quickstart wizard.

    The wire response nests the repository under ``repo``; this flattens it so
    the wizard's poll loop reads ``status.repo_connected`` directly.
    """

    repo_connected: bool
    repo_full_name: str | None
    billing_ready: bool
    completed: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuickstartStatus:
        repo = data.get("repo")
        repo_data = repo if isinstance(repo, dict) else {}
        full_name = repo_data.get("full_name")
        return cls(
            repo_connected=repo_data.get("connected", False),
            repo_full_name=full_name if isinstance(full_name, str) else None,
            billing_ready=data.get("billing_ready", False),
            completed=data.get("completed", False),
        )

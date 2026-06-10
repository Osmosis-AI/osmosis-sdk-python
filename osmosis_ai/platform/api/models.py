"""Data models for Platform CLI API responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# ── Dataset status constants ─────────────────────────────────────
# Single source of truth for status classification.

STATUSES_SUCCESS: frozenset[str] = frozenset({"uploaded"})
# "pending" waits (amber); "uploading"/"processing" are active work (blue).
STATUSES_PENDING: frozenset[str] = frozenset({"pending"})
STATUSES_ACTIVE: frozenset[str] = frozenset({"uploading", "processing"})
STATUSES_IN_PROGRESS: frozenset[str] = STATUSES_PENDING | STATUSES_ACTIVE
STATUSES_ERROR: frozenset[str] = frozenset({"error"})
STATUSES_INACTIVE: frozenset[str] = frozenset({"cancelled"})
STATUSES_TERMINAL: frozenset[str] = (
    STATUSES_SUCCESS | STATUSES_ERROR | STATUSES_INACTIVE
)


@dataclass
class UploadInfo:
    """Upload instructions returned by the create-dataset endpoint."""

    method: Literal["simple", "multipart"]
    s3_key: str
    # simple upload fields
    presigned_url: str | None = None
    expires_in: int | None = None
    upload_headers: dict[str, str] | None = None
    # multipart upload fields
    upload_id: str | None = None
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
            s3_key=data["s3_key"],
            presigned_url=data.get("presigned_url"),
            expires_in=data.get("expires_in"),
            upload_headers=data.get("upload_headers"),
            upload_id=data.get("upload_id"),
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
    df_stats: Any = None
    file_format: str | None = None
    original_file_format: str | None = None
    row_count: int | None = None
    original_file_size: int | None = None
    creator_name: str | None = None
    organization_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    platform_url: str | None = None
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
            df_stats=data.get("df_stats"),
            file_format=data.get("file_format"),
            original_file_format=data.get("original_file_format"),
            row_count=data.get("row_count"),
            original_file_size=data.get("original_file_size"),
            creator_name=data.get("creator_name"),
            organization_id=data.get("organization_id"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            platform_url=data.get("platform_url"),
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
    commit_sha: str | None = None
    env_config: dict[str, Any] | None = None
    resolved_secret_scopes: dict[str, Any] | None = None

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
            commit_sha=data.get("commit_sha")
            if isinstance(data.get("commit_sha"), str)
            else None,
            env_config=env_config if isinstance(env_config, dict) else None,
            resolved_secret_scopes=(
                resolved_secret_scopes
                if isinstance(resolved_secret_scopes, dict)
                else None
            ),
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

    training_run_id: str
    status: str
    overview: TrainingRunMetricsOverview
    metrics: list[MetricHistory]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunMetrics:
        return cls(
            training_run_id=data["training_run_id"],
            status=data["status"],
            overview=TrainingRunMetricsOverview.from_dict(data["overview"]),
            metrics=[MetricHistory.from_dict(m) for m in data.get("metrics", [])],
        )


# ── Training run logs ────────────────────────────────────────────


@dataclass
class TrainingRunLogEntry:
    """A single training run lifecycle log line."""

    timestamp: str
    level: str
    step: str
    message: str
    details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunLogEntry:
        details = data.get("details")
        return cls(
            timestamp=data.get("timestamp", ""),
            level=data.get("level", ""),
            step=data.get("step", ""),
            message=data.get("message", ""),
            details=details if isinstance(details, dict) else None,
        )


@dataclass
class TrainingRunLogs:
    """One cursor page of training run logs.

    The server returns entries oldest-first within the page for both paging
    directions; ``next_cursor`` pages further back in time (``None`` when no
    more pages exist).
    """

    logs: list[TrainingRunLogEntry]
    next_cursor: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunLogs:
        return cls(
            logs=[TrainingRunLogEntry.from_dict(log) for log in data.get("logs", [])],
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

    eval_run_id: str
    status: str
    overview: EvalRunMetricsOverview
    reward_stats: EvalRewardStats | None
    pass_at_k: list[EvalPassAtKPoint]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunMetrics:
        reward_stats = data.get("reward_stats")
        return cls(
            eval_run_id=data["eval_run_id"],
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
    created_at: str = ""

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
            created_at=data.get("created_at", ""),
        )


@dataclass
class PaginatedLoraModels:
    """Paginated list of LoRA models."""

    models: list[LoraModelInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedLoraModels:
        return cls(
            models=[LoraModelInfo.from_dict(m) for m in data.get("models", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
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
DEPLOYMENT_STATUSES_INACTIVE: frozenset[str] = frozenset({"inactive"})


# ── Rollouts ─────────────────────────────────────────────────────


@dataclass
class RolloutInfo:
    """A rollout record."""

    id: str
    name: str
    description: str | None = None
    is_active: bool = True
    last_synced_at: str | None = None
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
            last_synced_at=data.get("last_synced_at"),
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

    training_run_id: str
    training_run_name: str
    checkpoints: list[LoraCheckpointInfo]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunCheckpoints:
        return cls(
            training_run_id=data["training_run_id"],
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
    row_index: int | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        row_index = _number_or_none(data.get("row_index"))
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
            row_index=int(row_index) if row_index is not None else None,
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
    commit_sha: str | None = None
    env_config: dict[str, Any] | None = None
    resolved_secret_scopes: dict[str, Any] | None = None
    dataset_df_stats: dict[str, Any] | None = None
    recent_logs: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRunDetail:
        run = data["eval_run"]
        config = data.get("config")
        model_path = config.get("model_path") if isinstance(config, dict) else None
        row_index = _number_or_none(run.get("row_index"))
        env_config = data.get("env_config")
        resolved_secret_scopes = data.get("resolved_secret_scopes")
        dataset_df_stats = data.get("dataset_df_stats")
        recent_logs = data.get("recent_logs")
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
            row_index=int(row_index) if row_index is not None else None,
            config=config,
            results=data.get("results"),
            entrypoint=data.get("entrypoint")
            if isinstance(data.get("entrypoint"), str)
            else None,
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
            recent_logs=recent_logs if isinstance(recent_logs, list) else None,
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

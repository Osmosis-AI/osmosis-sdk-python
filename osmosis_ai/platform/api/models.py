"""Data models for Platform CLI API responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# ── Dataset status constants ─────────────────────────────────────
# Single source of truth for status classification.

STATUSES_SUCCESS: frozenset[str] = frozenset({"uploaded"})
STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"pending", "uploading", "processing"})
STATUSES_ERROR: frozenset[str] = frozenset({"error"})
STATUSES_INACTIVE: frozenset[str] = frozenset({"cancelled", "deleted"})
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
    processing_step: str | None = None
    processing_percent: float | None = None
    error: str | None = None
    data_preview: Any = None
    df_stats: Any = None
    organization_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
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
            processing_step=data.get("processing_step"),
            processing_percent=data.get("processing_percent"),
            error=data.get("error"),
            data_preview=data.get("data_preview"),
            df_stats=data.get("df_stats"),
            organization_id=data.get("organization_id"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            upload=upload,
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the file is in a terminal processing state."""
        return self.status in STATUSES_TERMINAL


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
RUN_STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"pending", "running"})
RUN_STATUSES_ERROR: frozenset[str] = frozenset({"failed", "crashed"})
RUN_STATUSES_STOPPED: frozenset[str] = frozenset({"stopped", "killed"})
RUN_STATUSES_TERMINAL: frozenset[str] = (
    RUN_STATUSES_SUCCESS | RUN_STATUSES_ERROR | RUN_STATUSES_STOPPED
)


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
    eval_accuracy: float | None = None
    reward_increase_delta: float | None = None
    processing_step: str | None = None
    processing_percent: float | None = None
    error_message: str | None = None
    creator_name: str | None = None
    creator_email: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRun:
        model = data.get("model") or {}
        return cls(
            id=data["id"],
            name=data.get("name"),
            status=data.get("status", ""),
            model_id=data.get("model_id"),
            model_name=model.get("model_name"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            eval_accuracy=data.get("eval_accuracy"),
            reward_increase_delta=data.get("reward_increase_delta"),
            processing_step=data.get("processing_step"),
            processing_percent=data.get("processing_percent"),
            error_message=data.get("error_message"),
            creator_name=data.get("creator_name"),
            creator_email=data.get("creator_email"),
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the run is in a terminal state."""
        return self.status in RUN_STATUSES_TERMINAL


@dataclass
class TrainingRunDetail(TrainingRun):
    """Detailed training run info with additional fields."""

    examples_processed_count: int | None = None
    notes: str | None = None
    hf_status: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunDetail:
        # Detail API returns { training_run: {..., enhanced_status}, model: {...} }
        run = data["training_run"]
        model = data.get("model") or {}
        return cls(
            id=run["id"],
            name=run.get("name"),
            status=run.get("enhanced_status") or run.get("status", ""),
            model_id=run.get("model_id"),
            model_name=model.get("model_name"),
            created_at=run.get("created_at", ""),
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            eval_accuracy=run.get("eval_accuracy"),
            reward_increase_delta=run.get("reward_increase_delta"),
            processing_step=run.get("processing_step"),
            processing_percent=run.get("processing_percent"),
            error_message=run.get("error_message"),
            creator_name=run.get("creator_name"),
            creator_email=run.get("creator_email"),
            examples_processed_count=run.get("examples_processed_count"),
            notes=run.get("notes"),
            hf_status=run.get("hf_status"),
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
class DeleteTrainingRunResult:
    """Result of deleting a training run."""

    deleted: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeleteTrainingRunResult:
        return cls(deleted=data["deleted"])


@dataclass
class SubmitTrainingRunResult:
    """Result of submitting a new training run."""

    id: str
    name: str
    status: str
    created_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubmitTrainingRunResult:
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
        )


@dataclass
class AffectedTrainingRun:
    """A training run affected by a resource deletion."""

    id: str
    training_run_name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffectedTrainingRun:
        return cls(
            id=data["id"],
            training_run_name=data.get("training_run_name"),
        )


@dataclass
class DatasetAffectedResources:
    """Affected resources for a dataset deletion."""

    affected_training_runs: list[AffectedTrainingRun]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetAffectedResources:
        return cls(
            affected_training_runs=[
                AffectedTrainingRun.from_dict(r)
                for r in data.get("affected_training_runs", [])
            ],
        )

    @property
    def has_blocking_runs(self) -> bool:
        return len(self.affected_training_runs) > 0


@dataclass
class ModelAffectedResources:
    """Affected resources for a model deletion."""

    training_runs_using_model: list[AffectedTrainingRun]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelAffectedResources:
        return cls(
            training_runs_using_model=[
                AffectedTrainingRun.from_dict(r)
                for r in data.get("training_runs_using_model", [])
            ],
        )

    @property
    def has_blocking_runs(self) -> bool:
        """Whether there are training runs that block deletion."""
        return len(self.training_runs_using_model) > 0


# ── Training run metrics ─────────────────────────────────────────


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

    mlflow_run_id: str
    mlflow_status: str
    duration_ms: int | None
    duration_formatted: str | None
    reward: float | None
    reward_delta: float | None
    examples_processed_count: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunMetricsOverview:
        return cls(
            mlflow_run_id=data["mlflow_run_id"],
            mlflow_status=data["mlflow_status"],
            duration_ms=data.get("duration_ms"),
            duration_formatted=data.get("duration_formatted"),
            reward=data.get("reward"),
            reward_delta=data.get("reward_increase_delta"),
            examples_processed_count=data.get("examples_processed_count"),
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


@dataclass
class ProcessCount:
    """Running process counts for a workspace-scoped resource category."""

    count: int
    valid: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessCount:
        return cls(count=data.get("count", 0), valid=data.get("valid", True))


@dataclass
class WorkspaceDeletionStatus:
    """Workspace deletion readiness status."""

    can_delete: bool
    is_owner: bool
    is_last_workspace: bool
    has_running_processes: bool
    feature_pipelines: ProcessCount
    training_runs: ProcessCount
    models: ProcessCount

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceDeletionStatus:
        return cls(
            can_delete=data.get("can_delete", False),
            is_owner=data.get("is_owner", False),
            is_last_workspace=data.get("is_last_workspace", False),
            has_running_processes=data.get("has_running_processes", False),
            feature_pipelines=ProcessCount.from_dict(data.get("feature_pipelines", {})),
            training_runs=ProcessCount.from_dict(data.get("training_runs", {})),
            models=ProcessCount.from_dict(data.get("models", {})),
        )


@dataclass
class BaseModelInfo:
    """A base (foundation) model record."""

    id: str
    model_name: str
    base_model: str | None = None
    status: str = ""
    description: str | None = None
    creator_name: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModelInfo:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            status=data.get("status", ""),
            description=data.get("description"),
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


# ── Deployments ──────────────────────────────────────────────────
# Status lifecycle: "active" / "inactive" / "failed".
# Naming lives on the LoRA checkpoint (`checkpoint_name`), not the deployment.

DEPLOYMENT_STATUSES_SUCCESS: frozenset[str] = frozenset({"active"})
DEPLOYMENT_STATUSES_INACTIVE: frozenset[str] = frozenset({"inactive"})
DEPLOYMENT_STATUSES_ERROR: frozenset[str] = frozenset({"failed"})


@dataclass
class DeploymentInfo:
    """A LoRA deployment record (one deployment per checkpoint)."""

    id: str
    checkpoint_name: str
    status: str
    checkpoint_step: int
    base_model: str
    training_run_id: str | None = None
    training_run_name: str | None = None
    creator_name: str | None = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentInfo:
        return cls(
            id=data["id"],
            checkpoint_name=data.get("checkpoint_name", ""),
            status=data.get("status", ""),
            checkpoint_step=data.get("checkpoint_step", 0),
            base_model=data.get("base_model", ""),
            training_run_id=data.get("training_run_id"),
            training_run_name=data.get("training_run_name"),
            creator_name=data.get("creator_name"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class PaginatedDeployments:
    """Paginated list of deployments."""

    deployments: list[DeploymentInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedDeployments:
        return cls(
            deployments=[
                DeploymentInfo.from_dict(d) for d in data.get("deployments", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class DeploymentSummary:
    """Minimal deployment identity returned from deploy/undeploy endpoints."""

    id: str
    checkpoint_name: str
    status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentSummary:
        return cls(
            id=data["id"],
            checkpoint_name=data.get("checkpoint_name", ""),
            status=data.get("status", ""),
        )


@dataclass
class RenameDeploymentResult:
    """Result of renaming a checkpoint via PATCH /api/cli/deployments/[checkpointId]."""

    id: str
    old_checkpoint_name: str
    checkpoint_name: str
    status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenameDeploymentResult:
        return cls(
            id=data["id"],
            old_checkpoint_name=data.get("old_checkpoint_name", ""),
            checkpoint_name=data.get("checkpoint_name", ""),
            status=data.get("status", ""),
        )


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


# ── LoRA checkpoints (for `osmosis train info` + `osmosis deploy`) ─────


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

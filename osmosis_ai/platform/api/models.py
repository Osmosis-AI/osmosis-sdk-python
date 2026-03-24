"""Data models for Platform CLI API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Dataset status constants ─────────────────────────────────────
# Single source of truth for status classification.

STATUSES_SUCCESS: frozenset[str] = frozenset({"ready"})
STATUSES_IN_PROGRESS: frozenset[str] = frozenset({"processing", "uploaded"})
STATUSES_ERROR: frozenset[str] = frozenset({"failed", "error"})
STATUSES_INACTIVE: frozenset[str] = frozenset({"cancelled", "deleted"})
STATUSES_TERMINAL: frozenset[str] = (
    STATUSES_SUCCESS | STATUSES_ERROR | STATUSES_INACTIVE
)


@dataclass
class Project:
    """A project in a workspace."""

    id: str
    project_name: str
    role: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        return cls(
            id=data["id"],
            project_name=data["project_name"],
            role=data.get("role", "member"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "project_name": self.project_name, "role": self.role}


@dataclass
class PaginatedProjects:
    """Paginated list of projects."""

    projects: list[Project]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedProjects:
        return cls(
            projects=[Project.from_dict(p) for p in data.get("projects", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )


@dataclass
class DatasetSummary:
    """Summary of a dataset file (used in project detail)."""

    id: str
    file_name: str
    file_size: int
    status: str
    created_at: str


@dataclass
class ProjectDetail:
    """Detailed project info including recent datasets and summary counts."""

    id: str
    project_name: str
    role: str
    created_at: str
    updated_at: str
    dataset_count: int = 0
    recent_datasets: list[DatasetSummary] = field(default_factory=list)
    training_run_count: int = 0
    base_model_count: int = 0
    output_model_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDetail:
        datasets_data = data.get("datasets", {})
        recent = [
            DatasetSummary(
                id=d["id"],
                file_name=d["file_name"],
                file_size=d["file_size"],
                status=d["status"],
                created_at=d["created_at"],
            )
            for d in datasets_data.get("recent", [])
        ]
        runs_data = data.get("training_runs", {})
        models_data = data.get("models", {})
        return cls(
            id=data["id"],
            project_name=data["project_name"],
            role=data.get("role", "member"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            dataset_count=datasets_data.get("total_count", 0),
            recent_datasets=recent,
            training_run_count=runs_data.get("total_count", 0),
            base_model_count=models_data.get("base_count", 0),
            output_model_count=models_data.get("output_count", 0),
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
    """A training data file record."""

    id: str
    file_name: str
    file_size: int
    status: str
    processing_step: str | None = None
    processing_percent: float | None = None
    error: str | None = None
    data_preview: Any = None
    df_stats: Any = None
    project_id: str | None = None
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
            project_id=data.get("project_id"),
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
    """A training run in a project."""

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

    output_model_id: str | None = None
    project_id: str | None = None
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
            output_model_id=run.get("output_model_id"),
            project_id=run.get("project_id"),
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
class PreservedModel:
    """A model that was preserved after its training run was deleted."""

    id: str
    name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreservedModel:
        return cls(id=data["id"], name=data["name"])


@dataclass
class DeleteTrainingRunResult:
    """Result of deleting a training run."""

    deleted: bool
    preserved_output_model: PreservedModel | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeleteTrainingRunResult:
        preserved = data.get("preservedOutputModel")
        return cls(
            deleted=data.get("deleted", True),
            preserved_output_model=PreservedModel.from_dict(preserved)
            if preserved
            else None,
        )


@dataclass
class AffectedTrainingRun:
    """A training run affected by a model deletion."""

    id: str
    name: str | None
    project_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffectedTrainingRun:
        return cls(
            id=data["id"],
            name=data.get("name"),
            project_name=data["project_name"],
        )


@dataclass
class ModelAffectedResources:
    """Affected resources for a model deletion."""

    training_runs_using_model: list[AffectedTrainingRun]
    creator_training_run: AffectedTrainingRun | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelAffectedResources:
        creator = data.get("creator_training_run")
        return cls(
            training_runs_using_model=[
                AffectedTrainingRun.from_dict(r)
                for r in data.get("training_runs_using_model", [])
            ],
            creator_training_run=AffectedTrainingRun.from_dict(creator)
            if creator
            else None,
        )

    @property
    def has_blocking_runs(self) -> bool:
        """Whether there are training runs that block deletion."""
        return len(self.training_runs_using_model) > 0


@dataclass
class ProjectProcessCount:
    """Running process counts for a project."""

    count: int
    valid: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectProcessCount:
        return cls(count=data.get("count", 0), valid=data.get("valid", True))


@dataclass
class ProjectDeletionStatus:
    """Deletion readiness status for a single project."""

    project_id: str
    project_name: str
    has_running_processes: bool
    feature_pipelines: ProjectProcessCount
    training_runs: ProjectProcessCount
    models: ProjectProcessCount

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDeletionStatus:
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            has_running_processes=data.get("has_running_processes", False),
            feature_pipelines=ProjectProcessCount.from_dict(
                data.get("feature_pipelines", {})
            ),
            training_runs=ProjectProcessCount.from_dict(data.get("training_runs", {})),
            models=ProjectProcessCount.from_dict(data.get("models", {})),
        )


@dataclass
class WorkspaceDeletionStatus:
    """Workspace deletion readiness status."""

    can_delete: bool
    is_owner: bool
    is_last_workspace: bool
    projects: list[ProjectDeletionStatus]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceDeletionStatus:
        return cls(
            can_delete=data.get("can_delete", False),
            is_owner=data.get("is_owner", False),
            is_last_workspace=data.get("is_last_workspace", False),
            projects=[
                ProjectDeletionStatus.from_dict(p) for p in data.get("projects", [])
            ],
        )

    @property
    def projects_with_running_processes(self) -> list[ProjectDeletionStatus]:
        """Projects that have running processes blocking deletion."""
        return [p for p in self.projects if p.has_running_processes]


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
class OutputModelInfo:
    """An output model created by a training run."""

    id: str
    model_name: str
    base_model: str | None = None
    status: str = ""
    description: str | None = None
    training_run_id: str | None = None
    training_run_name: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputModelInfo:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            status=data.get("status", ""),
            description=data.get("description"),
            training_run_id=data.get("training_run_id"),
            training_run_name=data.get("training_run_name"),
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
class PaginatedOutputModels:
    """Paginated list of output models."""

    models: list[OutputModelInfo]
    total_count: int
    has_more: bool
    next_offset: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedOutputModels:
        return cls(
            models=[OutputModelInfo.from_dict(m) for m in data.get("models", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
            next_offset=data.get("next_offset"),
        )

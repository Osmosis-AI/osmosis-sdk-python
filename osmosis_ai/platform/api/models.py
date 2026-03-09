"""Data models for Platform CLI API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
class DatasetSummary:
    """Summary of a dataset file (used in project detail)."""

    id: str
    file_name: str
    file_size: int
    status: str
    created_at: str


@dataclass
class ProjectDetail:
    """Detailed project info including recent datasets."""

    id: str
    project_name: str
    role: str
    created_at: str
    updated_at: str
    dataset_count: int = 0
    recent_datasets: list[DatasetSummary] = field(default_factory=list)

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
        return cls(
            id=data["id"],
            project_name=data["project_name"],
            role=data.get("role", "member"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            dataset_count=datasets_data.get("total_count", 0),
            recent_datasets=recent,
        )


@dataclass
class UploadInfo:
    """Upload instructions returned by the create-dataset endpoint."""

    method: str  # "simple" | "multipart"
    s3_key: str
    # simple upload fields
    presigned_url: str | None = None
    expires_in: int | None = None
    upload_headers: dict[str, str] | None = None
    # multipart upload fields
    upload_id: str | None = None
    part_size: int | None = None
    total_parts: int | None = None
    presigned_urls: list[dict[str, Any]] | None = None  # [{partNumber, presignedUrl}]

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
            s3_key=data.get("s3_key", ""),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedDatasets:
        return cls(
            datasets=[DatasetFile.from_dict(d) for d in data.get("datasets", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
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
    notes: str | None = None
    hf_status: str | None = None
    examples_processed_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunDetail:
        # Detail API returns { training_run: {...}, model: {...}, enhanced_status: "..." }
        run = data["training_run"]
        model = data.get("model") or {}
        return cls(
            id=run["id"],
            name=run.get("name"),
            status=data.get("enhanced_status") or run.get("status", ""),
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
            notes=run.get("notes"),
            hf_status=run.get("hf_status"),
            examples_processed_count=run.get("examples_processed_count"),
        )


@dataclass
class PaginatedTrainingRuns:
    """Paginated list of training runs."""

    training_runs: list[TrainingRun]
    total_count: int
    has_more: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedTrainingRuns:
        return cls(
            training_runs=[
                TrainingRun.from_dict(r) for r in data.get("training_runs", [])
            ],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
        )


@dataclass
class ModelInfo:
    """A model record."""

    id: str
    model_name: str
    base_model: str | None = None
    status: str = ""
    description: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(
            id=data["id"],
            model_name=data.get("model_name", ""),
            base_model=data.get("base_model"),
            status=data.get("status", ""),
            description=data.get("description"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class PaginatedModels:
    """Paginated list of models."""

    models: list[ModelInfo]
    total_count: int
    has_more: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaginatedModels:
        return cls(
            models=[ModelInfo.from_dict(m) for m in data.get("models", [])],
            total_count=data.get("total_count", 0),
            has_more=data.get("has_more", False),
        )

"""Data models for Platform CLI API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFile:
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
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the file is in a terminal processing state."""
        return self.status in ("uploaded", "error", "cancelled", "deleted")


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


@dataclass
class PresignedUpload:
    """Result from upload-url endpoint."""

    presigned_url: str
    s3_key: str
    expires_in: int
    upload_headers: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresignedUpload:
        return cls(
            presigned_url=data["presigned_url"],
            s3_key=data["s3_key"],
            expires_in=data.get("expires_in", 3600),
            upload_headers=data.get("upload_headers", {}),
        )

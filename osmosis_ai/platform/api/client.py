"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from osmosis_ai.platform.auth.platform_client import platform_request

from .models import (
    DatasetFile,
    PaginatedDatasets,
    Project,
    ProjectDetail,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import WorkspaceCredentials


class OsmosisClient:
    """Client for /api/cli/* endpoints.

    Uses the active workspace credentials from local storage unless explicit
    workspace credentials are provided.
    """

    # ── Workspace ────────────────────────────────────────────────────

    def refresh_workspace_info(
        self, *, credentials: WorkspaceCredentials | None = None
    ) -> dict:
        """GET /api/cli/verify — refresh cached workspace info (user, org, projects)."""
        return platform_request("/api/cli/verify", credentials=credentials)

    # ── Projects ─────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> Project:
        data = platform_request(
            "/api/cli/projects",
            method="POST",
            data={"name": name},
            credentials=credentials,
        )
        return Project.from_dict(data)

    def get_project(
        self,
        project_id: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> ProjectDetail:
        data = platform_request(
            f"/api/cli/projects/{project_id}",
            credentials=credentials,
        )
        return ProjectDetail.from_dict(data)

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
        project_id: str,
        file_name: str,
        file_size: int,
        extension: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> DatasetFile:
        data = platform_request(
            "/api/cli/datasets",
            method="POST",
            data={
                "project_id": project_id,
                "file_name": file_name,
                "file_size": file_size,
                "extension": extension,
            },
            credentials=credentials,
        )
        return DatasetFile.from_dict(data)

    def complete_upload(
        self,
        file_id: str,
        s3_key: str,
        extension: str | None = None,
        upload_id: str | None = None,
        parts: list[dict] | None = None,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> DatasetFile:
        payload: dict = {"s3_key": s3_key}
        if extension is not None:
            payload["extension"] = extension
        if upload_id is not None or parts is not None:
            if not upload_id or not parts:
                raise ValueError(
                    "upload_id and parts must both be provided for multipart completion"
                )
            payload["upload_id"] = upload_id
            payload["parts"] = parts
        # Completing a multipart upload can take a while (S3 must assemble
        # all parts), so use a longer timeout than the default 30s.
        timeout = 120.0 if upload_id else 30.0
        data = platform_request(
            f"/api/cli/datasets/{file_id}/complete",
            method="POST",
            data=payload,
            timeout=timeout,
            credentials=credentials,
        )
        return DatasetFile.from_dict(data)

    def abort_upload(
        self,
        file_id: str,
        upload_id: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> None:
        platform_request(
            f"/api/cli/datasets/{file_id}/abort",
            method="POST",
            data={"upload_id": upload_id},
            credentials=credentials,
        )

    def list_datasets(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> PaginatedDatasets:
        qs = urlencode({"project_id": project_id, "limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/datasets?{qs}", credentials=credentials)
        return PaginatedDatasets.from_dict(data)

    def get_dataset(
        self,
        file_id: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> DatasetFile:
        data = platform_request(
            f"/api/cli/datasets/{file_id}",
            credentials=credentials,
        )
        return DatasetFile.from_dict(data)

    def delete_dataset(
        self,
        file_id: str,
        *,
        credentials: WorkspaceCredentials | None = None,
    ) -> bool:
        platform_request(
            f"/api/cli/datasets/{file_id}",
            method="DELETE",
            credentials=credentials,
        )
        return True

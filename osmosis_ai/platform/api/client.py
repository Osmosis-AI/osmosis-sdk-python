"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request

from .models import (
    DatasetAffectedResources,
    DatasetFile,
    DeleteTrainingRunResult,
    ModelAffectedResources,
    PaginatedBaseModels,
    PaginatedDatasets,
    PaginatedOutputModels,
    PaginatedProjects,
    PaginatedTrainingRuns,
    Project,
    ProjectDetail,
    TrainingRunAffectedResources,
    TrainingRunDetail,
    WorkspaceDeletionStatus,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def _safe_path(segment: str) -> str:
    """URL-encode a path segment to prevent path traversal."""
    return quote(segment, safe="")


class OsmosisClient:
    """Client for /api/cli/* endpoints.

    Uses the active workspace credentials from local storage unless explicit
    workspace credentials are provided.
    """

    # ── Workspace ────────────────────────────────────────────────────

    def refresh_workspace_info(
        self,
        *,
        credentials: Credentials | None = None,
        workspace_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch subscription status for a workspace via /api/cli/workspaces.

        Returns a dict with ``has_subscription`` for the matched workspace,
        or an empty dict if the workspace is not found.
        """
        data = platform_request(
            "/api/cli/workspaces",
            credentials=credentials,
            require_workspace=False,
        )
        for ws in data.get("workspaces", []):
            if workspace_name and ws.get("name") == workspace_name:
                return {"has_subscription": ws.get("has_subscription")}
        return {}

    def list_workspaces(
        self,
        *,
        credentials: Credentials | None = None,
    ) -> dict[str, Any]:
        """List all workspaces the user belongs to."""
        return platform_request(
            "/api/cli/workspaces",
            credentials=credentials,
            require_workspace=False,
        )

    def create_workspace(
        self,
        name: str,
        timezone: str = "UTC",
        *,
        credentials: Credentials | None = None,
    ) -> dict[str, Any]:
        """Create a new workspace (organization)."""
        return platform_request(
            "/api/cli/workspaces",
            method="POST",
            data={"name": name, "timezone": timezone},
            credentials=credentials,
            require_workspace=False,
        )

    def delete_workspace(
        self,
        workspace_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> bool:
        """Delete a workspace (organization)."""
        platform_request(
            f"/api/cli/workspaces/{_safe_path(workspace_id)}",
            method="DELETE",
            credentials=credentials,
            require_workspace=False,
        )
        return True

    def get_workspace_deletion_status(
        self,
        workspace_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> WorkspaceDeletionStatus:
        """Get workspace deletion readiness status."""
        data = platform_request(
            f"/api/cli/workspaces/{_safe_path(workspace_id)}/deletion-status",
            credentials=credentials,
            require_workspace=False,
        )
        return WorkspaceDeletionStatus.from_dict(data)

    # ── Projects ─────────────────────────────────────────────────────

    def list_projects(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str | None = None,
    ) -> PaginatedProjects:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/projects?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedProjects.from_dict(data)

    def create_project(
        self,
        name: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str | None = None,
    ) -> Project:
        data = platform_request(
            "/api/cli/projects",
            method="POST",
            data={"name": name},
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return Project.from_dict(data)

    def get_project(
        self,
        project_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> ProjectDetail:
        data = platform_request(
            f"/api/cli/projects/{_safe_path(project_id)}",
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
        credentials: Credentials | None = None,
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
        parts: list[dict[str, Any]] | None = None,
        *,
        credentials: Credentials | None = None,
    ) -> DatasetFile:
        """Complete an upload.

        The server reads s3_key and upload_id from the DB record.
        For multipart uploads, provide the list of completed parts.
        For simple uploads, no parts needed.
        """
        payload: dict = {}
        if parts is not None:
            # Validate no duplicate part numbers before sending
            part_numbers = [p["PartNumber"] for p in parts]
            if len(part_numbers) != len(set(part_numbers)):
                raise ValueError(
                    f"Duplicate part numbers detected in {len(parts)} parts"
                )
            payload["parts"] = parts
        # Completing a multipart upload can take a while (S3 must assemble
        # all parts), so use a longer timeout than the default 30s.
        timeout = 120.0 if parts else 30.0
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/complete",
            method="POST",
            data=payload,
            timeout=timeout,
            credentials=credentials,
        )
        return DatasetFile.from_dict(data)

    def abort_upload(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> None:
        """Abort an in-progress upload.

        The server reads upload_id from the DB record and handles both
        multipart (abort S3 + cancel) and simple (cancel only) uploads.
        """
        platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/abort",
            method="POST",
            data={},
            credentials=credentials,
        )

    def list_datasets(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedDatasets:
        qs = urlencode({"project_id": project_id, "limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/datasets?{qs}", credentials=credentials)
        return PaginatedDatasets.from_dict(data)

    def get_dataset(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> DatasetFile:
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}",
            credentials=credentials,
        )
        return DatasetFile.from_dict(data)

    def delete_dataset(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> bool:
        platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}",
            method="DELETE",
            credentials=credentials,
        )
        return True

    def get_dataset_affected_resources(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> DatasetAffectedResources:
        """Get affected resources for a dataset deletion confirmation."""
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/affected-resources",
            credentials=credentials,
        )
        return DatasetAffectedResources.from_dict(data)

    # ── Training Runs ─────────────────────────────────────────────

    def list_training_runs(
        self,
        project_id: str,
        limit: int = 20,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedTrainingRuns:
        qs = urlencode({"project_id": project_id, "limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/training-runs?{qs}", credentials=credentials)
        return PaginatedTrainingRuns.from_dict(data)

    def get_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> TrainingRunDetail:
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}", credentials=credentials
        )
        return TrainingRunDetail.from_dict(data)

    def stop_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> dict[str, Any]:
        """Stop a pending or running training run."""
        return platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
        )

    def delete_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> DeleteTrainingRunResult:
        """Delete a training run (stops it first if active)."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}",
            method="DELETE",
            credentials=credentials,
        )
        return DeleteTrainingRunResult.from_dict(data)

    def get_training_run_affected_resources(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> TrainingRunAffectedResources:
        """Get affected resources for a training run deletion confirmation."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/affected-resources",
            credentials=credentials,
        )
        return TrainingRunAffectedResources.from_dict(data)

    # ── Models ────────────────────────────────────────────────────

    def list_base_models(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedBaseModels:
        qs = urlencode({"project_id": project_id, "limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/models/base?{qs}", credentials=credentials)
        return PaginatedBaseModels.from_dict(data)

    def list_output_models(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedOutputModels:
        qs = urlencode({"project_id": project_id, "limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/models/output?{qs}", credentials=credentials)
        return PaginatedOutputModels.from_dict(data)

    def delete_model(
        self,
        model_id: str,
        project_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> bool:
        """Delete a model with full cascade cleanup."""
        qs = urlencode({"project_id": project_id})
        platform_request(
            f"/api/cli/models/{_safe_path(model_id)}?{qs}",
            method="DELETE",
            credentials=credentials,
        )
        return True

    def get_model_affected_resources(
        self,
        model_id: str,
        project_id: str,
        model_type: Literal["base", "output"] = "base",
        *,
        credentials: Credentials | None = None,
    ) -> ModelAffectedResources:
        """Get affected resources for a model deletion.

        Args:
            model_id: The model ID.
            project_id: The project ID.
            model_type: "base" or "output".
        """
        qs = urlencode({"project_id": project_id, "type": model_type})
        data = platform_request(
            f"/api/cli/models/{_safe_path(model_id)}/affected-resources?{qs}",
            credentials=credentials,
        )
        return ModelAffectedResources.from_dict(data)

    def fetch_all_models(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> tuple[PaginatedBaseModels, PaginatedOutputModels]:
        """Fetch base and output models in parallel."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            base_fut = pool.submit(
                self.list_base_models,
                project_id,
                limit,
                offset,
                credentials=credentials,
            )
            output_fut = pool.submit(
                self.list_output_models,
                project_id,
                limit,
                offset,
                credentials=credentials,
            )
            return base_fut.result(), output_fut.result()

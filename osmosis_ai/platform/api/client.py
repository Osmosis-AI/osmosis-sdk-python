"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .models import (
    DatasetAffectedResources,
    DatasetFile,
    DeleteTrainingRunResult,
    ModelAffectedResources,
    PaginatedBaseModels,
    PaginatedDatasets,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    SubmitTrainingRunResult,
    TrainingRunDetail,
    TrainingRunMetrics,
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
        workspace_name: str,
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
            if ws.get("name") == workspace_name:
                return {"found": True, "has_subscription": ws.get("has_subscription")}
        return {"found": False}

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

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
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
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedDatasets:
        qs = urlencode({"limit": limit, "offset": offset})
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

    def submit_training_run(
        self,
        *,
        model_path: str,
        dataset: str,
        rollout_name: str,
        entrypoint: str,
        commit_sha: str | None = None,
        config: dict[str, Any] | None = None,
        credentials: Credentials | None = None,
    ) -> SubmitTrainingRunResult:
        """Submit a new training run."""
        data: dict[str, Any] = {
            "model_path": model_path,
            "dataset": dataset,
            "rollout_name": rollout_name,
            "entrypoint": entrypoint,
        }
        # Also send dataset_id for backward compatibility with older server versions
        try:
            from uuid import UUID

            UUID(dataset)
            data["dataset_id"] = dataset
        except ValueError:
            pass
        if commit_sha is not None:
            data["commit_sha"] = commit_sha
        if config is not None:
            data["config"] = config
        result = platform_request(
            "/api/cli/training-runs",
            method="POST",
            data=data,
            credentials=credentials,
        )
        return SubmitTrainingRunResult.from_dict(result)

    def list_training_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedTrainingRuns:
        qs = urlencode({"limit": limit, "offset": offset})
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

    def get_training_run_metrics(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> TrainingRunMetrics:
        """Fetch training run metrics (only available for terminal runs)."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/metrics",
            credentials=credentials,
        )
        return TrainingRunMetrics.from_dict(data)

    # ── Rollouts ──────────────────────────────────────────────────

    def list_rollouts(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedRollouts:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/rollouts?{qs}", credentials=credentials)
        return PaginatedRollouts.from_dict(data)

    # ── Models ────────────────────────────────────────────────────

    def list_base_models(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
    ) -> PaginatedBaseModels:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(f"/api/cli/models/base?{qs}", credentials=credentials)
        return PaginatedBaseModels.from_dict(data)

    def delete_model(
        self,
        model_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> bool:
        """Delete a model with full cascade cleanup."""
        platform_request(
            f"/api/cli/models/{_safe_path(model_id)}",
            method="DELETE",
            credentials=credentials,
        )
        return True

    def get_model_affected_resources(
        self,
        model_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> ModelAffectedResources:
        """Get affected resources for a model deletion."""
        data = platform_request(
            f"/api/cli/models/{_safe_path(model_id)}/affected-resources",
            credentials=credentials,
        )
        return ModelAffectedResources.from_dict(data)

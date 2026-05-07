"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .models import (
    DatasetAffectedResources,
    DatasetDownloadInfo,
    DatasetFile,
    DeleteTrainingRunResult,
    DeploymentInfo,
    DeploymentSummary,
    ModelAffectedResources,
    PaginatedBaseModels,
    PaginatedDatasets,
    PaginatedDeployments,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    RenameDeploymentResult,
    SubmitTrainingRunResult,
    TrainingRunCheckpoints,
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

    Workspace-scoped methods require an explicit ``workspace_id`` so calls can
    be tied to the linked project context.
    """

    # ── Workspace ────────────────────────────────────────────────────

    def refresh_workspace_info(
        self,
        *,
        credentials: Credentials | None = None,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
        cleanup_on_401: bool = True,
    ) -> dict[str, Any]:
        """Fetch workspace metadata via /api/cli/workspaces.

        Returns a dict with ``has_subscription`` plus optional Git integration
        fields for the matched workspace, or an empty dict if the workspace is
        not found. When both ``workspace_id`` and ``workspace_name`` are passed,
        ``workspace_id`` takes precedence.
        """
        data = platform_request(
            "/api/cli/workspaces",
            credentials=credentials,
            require_workspace=False,
            cleanup_on_401=cleanup_on_401,
        )
        for ws in data.get("workspaces", []):
            if workspace_id is not None and ws.get("id") == workspace_id:
                return {
                    "found": True,
                    "id": ws.get("id"),
                    "name": ws.get("name"),
                    "has_subscription": ws.get("has_subscription"),
                    "has_github_app_installation": ws.get(
                        "has_github_app_installation", False
                    ),
                    "connected_repo": ws.get("connected_repo"),
                }
        if workspace_id is None and workspace_name is not None:
            for ws in data.get("workspaces", []):
                if ws.get("name") == workspace_name:
                    return {
                        "found": True,
                        "id": ws.get("id"),
                        "name": ws.get("name"),
                        "has_subscription": ws.get("has_subscription"),
                        "has_github_app_installation": ws.get(
                            "has_github_app_installation", False
                        ),
                        "connected_repo": ws.get("connected_repo"),
                    }
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
        workspace_id: str,
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
            workspace_id=workspace_id,
        )
        return DatasetFile.from_dict(data)

    def complete_upload(
        self,
        file_id: str,
        parts: list[dict[str, Any]] | None = None,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
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
            workspace_id=workspace_id,
        )
        return DatasetFile.from_dict(data)

    def abort_upload(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
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
            workspace_id=workspace_id,
        )

    def list_datasets(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> PaginatedDatasets:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/datasets?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedDatasets.from_dict(data)

    def get_dataset(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DatasetFile:
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DatasetFile.from_dict(data)

    def get_dataset_download_url(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DatasetDownloadInfo:
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/download",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DatasetDownloadInfo.from_dict(data)

    def delete_dataset(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> bool:
        platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}",
            method="DELETE",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return True

    def get_dataset_affected_resources(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DatasetAffectedResources:
        """Get affected resources for a dataset deletion confirmation."""
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/affected-resources",
            credentials=credentials,
            workspace_id=workspace_id,
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
        workspace_id: str,
    ) -> SubmitTrainingRunResult:
        """Submit a new training run."""
        data: dict[str, Any] = {
            "model_path": model_path,
            "dataset": dataset,
            "rollout_name": rollout_name,
            "entrypoint": entrypoint,
        }
        if commit_sha is not None:
            data["commit_sha"] = commit_sha
        if config is not None:
            data["config"] = config
        result = platform_request(
            "/api/cli/training-runs",
            method="POST",
            data=data,
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return SubmitTrainingRunResult.from_dict(result)

    def list_training_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> PaginatedTrainingRuns:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/training-runs?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedTrainingRuns.from_dict(data)

    def get_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> TrainingRunDetail:
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return TrainingRunDetail.from_dict(data)

    def stop_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Stop a pending or running training run."""
        return platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
            workspace_id=workspace_id,
        )

    def delete_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DeleteTrainingRunResult:
        """Delete a training run (stops it first if active)."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}",
            method="DELETE",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DeleteTrainingRunResult.from_dict(data)

    def get_training_run_metrics(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> TrainingRunMetrics:
        """Fetch training run metrics (only available for terminal runs)."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/metrics",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return TrainingRunMetrics.from_dict(data)

    # ── Rollouts ──────────────────────────────────────────────────

    def list_rollouts(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> PaginatedRollouts:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/rollouts?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedRollouts.from_dict(data)

    # ── Models ────────────────────────────────────────────────────

    def list_base_models(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> PaginatedBaseModels:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/models/base?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedBaseModels.from_dict(data)

    def delete_model(
        self,
        model_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> bool:
        """Delete a model with full cascade cleanup."""
        platform_request(
            f"/api/cli/models/{_safe_path(model_id)}",
            method="DELETE",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return True

    def get_model_affected_resources(
        self,
        model_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> ModelAffectedResources:
        """Get affected resources for a model deletion."""
        data = platform_request(
            f"/api/cli/models/{_safe_path(model_id)}/affected-resources",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return ModelAffectedResources.from_dict(data)

    # ── Deployments ───────────────────────────────────────────────
    # All mutating endpoints key off `checkpoint` (UUID or checkpoint_name).
    # Lifecycle: deploy → active, undeploy → inactive, failure → failed.

    def list_deployments(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> PaginatedDeployments:
        """List LoRA deployments in the specified workspace."""
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/deployments?{qs}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return PaginatedDeployments.from_dict(data)

    def get_deployment(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DeploymentInfo:
        """Fetch a deployment by checkpoint UUID or checkpoint name."""
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DeploymentInfo.from_dict(data["deployment"])

    def deploy_checkpoint(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DeploymentSummary:
        """Deploy (or reactivate) a LoRA checkpoint.

        Idempotent: deploying a checkpoint that is already active returns
        the existing deployment.
        """
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}/deploy",
            method="POST",
            data={},
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DeploymentSummary.from_dict(data["deployment"])

    def undeploy_checkpoint(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> DeploymentSummary:
        """Undeploy a LoRA checkpoint (transitions to ``inactive``)."""
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}/undeploy",
            method="POST",
            data={},
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return DeploymentSummary.from_dict(data)

    def rename_checkpoint(
        self,
        checkpoint: str,
        new_name: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> RenameDeploymentResult:
        """Rename a LoRA checkpoint.

        Renaming an ``active`` deployment also re-registers the LoRA with
        inference under the new name.
        """
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}",
            method="PATCH",
            data={"checkpoint_name": new_name},
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return RenameDeploymentResult.from_dict(data)

    def delete_deployment(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> bool:
        """Delete a deployment record (idempotent)."""
        platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}",
            method="DELETE",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return True

    # ── Training-run checkpoints ──────────────────────────────────
    # Still used by `osmosis train status` to list deployable checkpoints.

    def list_training_run_checkpoints(
        self,
        name_or_id: str,
        *,
        credentials: Credentials | None = None,
        workspace_id: str,
    ) -> TrainingRunCheckpoints:
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(name_or_id)}/checkpoints",
            credentials=credentials,
            workspace_id=workspace_id,
        )
        return TrainingRunCheckpoints.from_dict(data)

"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .models import (
    DatasetDownloadInfo,
    DatasetFile,
    DeploymentInfo,
    DeploymentSummary,
    PaginatedBaseModels,
    PaginatedDatasets,
    PaginatedDeployments,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    SubmitTrainingRunResult,
    TrainingRunCheckpoints,
    TrainingRunDetail,
    TrainingRunMetrics,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def _safe_path(segment: str) -> str:
    """URL-encode a path segment to prevent path traversal."""
    return quote(segment, safe="")


class OsmosisClient:
    """Client for /api/cli/* endpoints.

    Repo-scoped methods require an explicit ``git_identity`` so calls can
    be tied to the trusted Git project context.
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
            require_git_repo=False,
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

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
        file_name: str,
        file_size: int,
        extension: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
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
            git_identity=git_identity,
        )
        return DatasetFile.from_dict(data)

    def complete_upload(
        self,
        file_id: str,
        parts: list[dict[str, Any]] | None = None,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
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
            git_identity=git_identity,
        )
        return DatasetFile.from_dict(data)

    def abort_upload(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
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
            git_identity=git_identity,
        )

    def list_datasets(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedDatasets:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/datasets?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedDatasets.from_dict(data)

    def get_dataset(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DatasetFile:
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return DatasetFile.from_dict(data)

    def get_dataset_download_url(
        self,
        file_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DatasetDownloadInfo:
        data = platform_request(
            f"/api/cli/datasets/{_safe_path(file_id)}/download",
            credentials=credentials,
            git_identity=git_identity,
        )
        return DatasetDownloadInfo.from_dict(data)

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
        rollout_env: dict[str, str] | None = None,
        rollout_secret_refs: dict[str, str] | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> SubmitTrainingRunResult:
        """Submit a new training run.

        ``rollout_env`` is a literal env-var-name → value map applied to the
        rollout container. ``rollout_secret_refs`` maps env-var names to the
        names of workspace ``environment_secret`` records; values are resolved
        server-side and never travel through the CLI.
        """
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
        if rollout_env:
            data["rollout_env"] = rollout_env
        if rollout_secret_refs:
            data["rollout_secret_refs"] = rollout_secret_refs
        result = platform_request(
            "/api/cli/training-runs",
            method="POST",
            data=data,
            credentials=credentials,
            git_identity=git_identity,
        )
        return SubmitTrainingRunResult.from_dict(result)

    def list_training_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedTrainingRuns:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/training-runs?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedTrainingRuns.from_dict(data)

    def get_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> TrainingRunDetail:
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return TrainingRunDetail.from_dict(data)

    def stop_training_run(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> dict[str, Any]:
        """Stop a pending or running training run."""
        return platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_training_run_metrics(
        self,
        run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> TrainingRunMetrics:
        """Fetch training run metrics (only available for terminal runs)."""
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(run_id)}/metrics",
            credentials=credentials,
            git_identity=git_identity,
        )
        return TrainingRunMetrics.from_dict(data)

    # ── Rollouts ──────────────────────────────────────────────────

    def list_rollouts(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedRollouts:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/rollouts?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedRollouts.from_dict(data)

    # ── Models ────────────────────────────────────────────────────

    def list_base_models(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedBaseModels:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/models/base?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedBaseModels.from_dict(data)

    # ── Deployments ───────────────────────────────────────────────
    # All mutating endpoints key off `checkpoint` (UUID or checkpoint_name).
    # Lifecycle: deploy → active, undeploy → inactive, failure → failed.

    def list_deployments(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedDeployments:
        """List LoRA deployments in the connected repository scope."""
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/deployments?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedDeployments.from_dict(data)

    def get_deployment(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DeploymentInfo:
        """Fetch a deployment by checkpoint UUID or checkpoint name."""
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return DeploymentInfo.from_dict(data["deployment"])

    def deploy_checkpoint(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
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
            git_identity=git_identity,
        )
        return DeploymentSummary.from_dict(data["deployment"])

    def undeploy_checkpoint(
        self,
        checkpoint: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DeploymentSummary:
        """Undeploy a LoRA checkpoint (transitions to ``inactive``)."""
        data = platform_request(
            f"/api/cli/deployments/{_safe_path(checkpoint)}/undeploy",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )
        return DeploymentSummary.from_dict(data)

    # ── Training-run checkpoints ──────────────────────────────────
    # Still used by `osmosis train status` to list deployable checkpoints.

    def list_training_run_checkpoints(
        self,
        name_or_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> TrainingRunCheckpoints:
        data = platform_request(
            f"/api/cli/training-runs/{_safe_path(name_or_id)}/checkpoints",
            credentials=credentials,
            git_identity=git_identity,
        )
        return TrainingRunCheckpoints.from_dict(data)

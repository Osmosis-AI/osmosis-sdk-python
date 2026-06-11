"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .models import (
    DatasetDownloadInfo,
    DatasetFile,
    EnvironmentSecretInfo,
    EvalRunMetrics,
    EvaluationRunDetail,
    LogsPage,
    LoraModelDetail,
    LoraModelSummary,
    PaginatedBaseModels,
    PaginatedDatasets,
    PaginatedEnvironmentSecrets,
    PaginatedEvaluationRuns,
    PaginatedLoraModels,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    SubmitRunResult,
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
    be tied to the trusted workspace directory context.
    """

    def _get_logs(
        self,
        resource_path: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LogsPage:
        """Fetch one page of logs for ``{resource_path}/logs``.

        Without ``cursor``, ``direction="older"`` returns the most recent page;
        the returned ``next_cursor`` pages further back in time.
        """
        params: dict[str, Any] = {"limit": limit, "direction": direction}
        if cursor is not None:
            params["cursor"] = cursor
        data = platform_request(
            f"{resource_path}/logs?{urlencode(params)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return LogsPage.from_dict(data)

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
        file_name: str,
        file_size: int,
        extension: str,
        *,
        overwrite_dataset_id: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DatasetFile:
        payload: dict[str, Any] = {
            "file_name": file_name,
            "file_size": file_size,
            "extension": extension,
        }
        if overwrite_dataset_id is not None:
            payload["overwrite_dataset_id"] = overwrite_dataset_id

        data = platform_request(
            "/api/cli/datasets",
            method="POST",
            data=payload,
            credentials=credentials,
            git_identity=git_identity,
        )
        return DatasetFile.from_dict(data)

    def complete_upload(
        self,
        file_id: str,
        parts: list[dict[str, Any]] | None = None,
        *,
        file_extension: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> DatasetFile:
        """Complete an upload.

        The server reads s3_key and upload_id from the DB record.
        For multipart uploads, provide the list of completed parts.
        For simple uploads, no parts needed.
        """
        payload: dict = {}
        if file_extension is not None:
            payload["file_extension"] = file_extension
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

    def get_dataset_logs(
        self,
        name_or_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LogsPage:
        """Fetch one page of dataset logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page;
        the returned ``next_cursor`` pages further back in time.
        """
        return self._get_logs(
            f"/api/cli/datasets/{_safe_path(name_or_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

    # ── Training Runs ─────────────────────────────────────────────

    def submit_training_run(
        self,
        *,
        experiment_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
        sampling_config: dict[str, Any] | None = None,
        checkpoints_config: dict[str, Any] | None = None,
        advanced_config: dict[str, Any] | None = None,
        env_config: dict[str, str] | None = None,
        secrets: list[str] | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> SubmitRunResult:
        """Submit a new training run.

        ``env_config`` is a literal env-var-name to value map applied to the
        rollout container. ``secrets`` is a list of ``environment_secret`` names;
        their values are resolved server-side and never travel through the CLI.
        """
        data: dict[str, Any] = {
            "experiment_config": experiment_config,
        }
        if training_config:
            data["training_config"] = training_config
        if sampling_config:
            data["sampling_config"] = sampling_config
        if checkpoints_config:
            data["checkpoints_config"] = checkpoints_config
        if advanced_config:
            data["advanced_config"] = advanced_config
        if env_config:
            data["env_config"] = env_config
        if secrets:
            data["secrets"] = {"required": secrets}
        result = platform_request(
            "/api/cli/training-runs",
            method="POST",
            data=data,
            credentials=credentials,
            git_identity=git_identity,
        )
        return SubmitRunResult.from_dict(result)

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
        """Stop a non-terminal training run (queued, pending, or running)."""
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

    def get_training_run_logs(
        self,
        name_or_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LogsPage:
        """Fetch one page of training run logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page;
        the returned ``next_cursor`` pages further back in time.
        """
        return self._get_logs(
            f"/api/cli/training-runs/{_safe_path(name_or_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

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

    def list_lora_models(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedLoraModels:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/models/lora?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedLoraModels.from_dict(data)

    def get_lora_model(
        self,
        lora_model_name: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LoraModelDetail:
        """Get details for a single LoRA model by name."""
        data = platform_request(
            f"/api/cli/models/{_safe_path(lora_model_name)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return LoraModelDetail.from_dict(data)

    def deploy_lora_model(
        self,
        lora_model_name: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LoraModelSummary:
        """Deploy (or reactivate) a LoRA model by name.

        Idempotent: deploying a LoRA model that is already active returns
        the existing deployment.
        """
        data = platform_request(
            f"/api/cli/models/{_safe_path(lora_model_name)}/deploy",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )
        return LoraModelSummary.from_dict(data)

    def undeploy_lora_model(
        self,
        lora_model_name: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LoraModelSummary:
        """Undeploy a LoRA model (transitions to ``inactive``); idempotent."""
        data = platform_request(
            f"/api/cli/models/{_safe_path(lora_model_name)}/undeploy",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )
        return LoraModelSummary.from_dict(data)

    # ── Environment Secrets ───────────────────────────────────────
    # Scoped secrets. The platform never echoes secret values:
    # list returns names + metadata only; set returns metadata only.

    def list_environment_secrets(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        scope: str = "all",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedEnvironmentSecrets:
        """List environment secrets (names + metadata only).

        ``scope`` is one of ``"all"`` (workspace + the caller's personal
        secrets), ``"workspace"``, or ``"user"`` (the caller's personal
        secrets only). The platform never returns secret values.
        """
        qs = urlencode({"limit": limit, "offset": offset, "scope": scope})
        data = platform_request(
            f"/api/cli/environment-secrets?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedEnvironmentSecrets.from_dict(data)

    def set_environment_secret(
        self,
        name: str,
        value: str,
        *,
        scope: str,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EnvironmentSecretInfo:
        """Create or update (upsert) an environment secret.

        ``scope`` is ``"workspace"`` or ``"user"``. The secret ``value`` is
        sent once in the request body and is never returned by the platform —
        the response carries only metadata. Callers must not log or echo
        ``value``.
        """
        data = platform_request(
            "/api/cli/environment-secrets",
            method="POST",
            data={"name": name, "value": value, "scope": scope},
            credentials=credentials,
            git_identity=git_identity,
        )
        return EnvironmentSecretInfo.from_dict(data)

    def delete_environment_secret(
        self,
        name: str,
        *,
        scope: str,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> None:
        """Delete an environment secret by name within ``scope``.

        ``scope`` is ``"workspace"`` or ``"user"``.
        """
        platform_request(
            "/api/cli/environment-secrets",
            method="DELETE",
            data={"name": name, "scope": scope},
            credentials=credentials,
            git_identity=git_identity,
        )

    # ── Training-run checkpoints ──────────────────────────────────
    # Still used by `osmosis train info` to list deployable checkpoints.

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

    # ── Evaluation Runs ──────────────────────────────────────────

    def submit_evaluation_run(
        self,
        *,
        experiment_config: dict[str, Any],
        evaluation_config: dict[str, Any] | None = None,
        advanced_config: dict[str, Any] | None = None,
        env_config: dict[str, str] | None = None,
        secrets: list[str] | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> SubmitRunResult:
        """Submit a new evaluation run."""
        data: dict[str, Any] = {
            "experiment_config": experiment_config,
        }
        if evaluation_config:
            data["evaluation_config"] = evaluation_config
        if advanced_config:
            data["advanced_config"] = advanced_config
        if env_config:
            data["env_config"] = env_config
        if secrets:
            data["secrets"] = {"required": secrets}
        result = platform_request(
            "/api/cli/eval-runs",
            method="POST",
            data=data,
            credentials=credentials,
            git_identity=git_identity,
        )
        return SubmitRunResult.from_dict(result)

    def list_eval_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedEvaluationRuns:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/eval-runs?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedEvaluationRuns.from_dict(data)

    def get_eval_run(
        self,
        eval_run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EvaluationRunDetail:
        data = platform_request(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return EvaluationRunDetail.from_dict(data)

    def get_eval_run_metrics(
        self,
        eval_run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EvalRunMetrics:
        """Fetch evaluation run metrics (unavailable for pending runs)."""
        data = platform_request(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}/metrics",
            credentials=credentials,
            git_identity=git_identity,
        )
        return EvalRunMetrics.from_dict(data)

    def get_eval_run_logs(
        self,
        name_or_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LogsPage:
        """Fetch one page of evaluation run logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page;
        the returned ``next_cursor`` pages further back in time.
        """
        return self._get_logs(
            f"/api/cli/eval-runs/{_safe_path(name_or_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

    def stop_eval_run(
        self,
        eval_run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> dict[str, Any]:
        """Stop a non-terminal evaluation run (queued, pending, or running)."""
        return platform_request(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )

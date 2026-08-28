"""High-level API client for Osmosis Platform CLI endpoints."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from osmosis_ai.platform.auth.platform_client import platform_request, platform_stream
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .models import (
    BenchmarkCatalogDetail,
    BenchmarkRunDetail,
    DatasetDownloadInfo,
    DatasetFile,
    EnvironmentSecretInfo,
    EvalRunImportResult,
    EvalRunImportUploads,
    EvalRunMetrics,
    EvaluationRunDetail,
    LogEntry,
    LogsPage,
    LoraModelDetail,
    LoraModelSummary,
    PaginatedBaseModels,
    PaginatedBenchmarkRuns,
    PaginatedBenchmarks,
    PaginatedDatasets,
    PaginatedDevRolloutServers,
    PaginatedEnvironmentSecrets,
    PaginatedEvaluationRuns,
    PaginatedLoraModels,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    QuickstartStatus,
    RunDownloadFile,
    RunDownloadManifest,
    RunDownloadURLBatch,
    SubmitBenchmarkRunResult,
    SubmitRunResult,
    TrainingRunCheckpoints,
    TrainingRunDetail,
    TrainingRunMetrics,
    WorkspaceSummary,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def _safe_path(segment: str) -> str:
    """URL-encode a path segment to prevent path traversal."""
    return quote(segment, safe="")


class OsmosisClient:
    """Client for /api/cli/* endpoints.

    Methods accept either an explicit ``git_identity`` or the root CLI workspace
    selection through the request context when ``git_identity`` is ``None``.
    """

    def _get_logs(
        self,
        resource_path: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> LogsPage:
        """Fetch one page of logs for ``{resource_path}/logs``.

        Without ``cursor``, ``direction="older"`` returns the most recent page.
        With a cursor, ``next_cursor`` continues in the requested direction.
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

    def _get_run_download_manifest(
        self,
        resource_path: str,
        *,
        types: Sequence[str],
        rows: str | None = None,
        route: str = "samples",
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadManifest:
        params: dict[str, str] = {"types": ",".join(types)}
        if rows is not None:
            params["rows"] = rows
        data = platform_request(
            f"{resource_path}/{route}/manifest?{urlencode(params)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return RunDownloadManifest.from_dict(data)

    def _get_run_download_urls(
        self,
        resource_path: str,
        *,
        items: Sequence[RunDownloadFile],
        route: str = "samples",
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadURLBatch:
        if not 1 <= len(items) <= 500:
            raise ValueError(
                "Download URL batches must contain between 1 and 500 items"
            )
        data = platform_request(
            f"{resource_path}/{route}/download-urls",
            method="POST",
            data={"items": [item.to_request_item() for item in items]},
            credentials=credentials,
            git_identity=git_identity,
        )
        return RunDownloadURLBatch.from_dict(data)

    # ── Datasets ─────────────────────────────────────────────────────

    def create_dataset(
        self,
        file_name: str,
        file_size: int,
        extension: str,
        *,
        overwrite_dataset_id: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
    ) -> LogsPage:
        """Fetch one page of dataset logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page.
        With a cursor, ``next_cursor`` continues in the requested direction.
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
        provided_secrets: dict[str, str] | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
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
        if secrets or provided_secrets:
            secrets_payload: dict[str, Any] = {"required": secrets or []}
            if provided_secrets:
                secrets_payload["provided"] = provided_secrets
            data["secrets"] = secrets_payload
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
    ) -> LogsPage:
        """Fetch one page of training run logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page.
        With a cursor, ``next_cursor`` continues in the requested direction.
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
        branch: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedRollouts:
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if branch:
            params["branch"] = branch
        qs = urlencode(params)
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        provided_secrets: dict[str, str] | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
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
        if secrets or provided_secrets:
            secrets_payload: dict[str, Any] = {"required": secrets or []}
            if provided_secrets:
                secrets_payload["provided"] = provided_secrets
            data["secrets"] = secrets_payload
        result = platform_request(
            "/api/cli/eval-runs",
            method="POST",
            data=data,
            credentials=credentials,
            git_identity=git_identity,
        )
        return SubmitRunResult.from_dict(result)

    def start_eval_run_import(
        self,
        *,
        local_run_id: str,
        manifest_digest: str,
        run: dict[str, Any],
        schema_versions: dict[str, Any],
        provenance: dict[str, Any],
        files: Sequence[dict[str, Any]],
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EvalRunImportResult:
        """Start or resume the server-authoritative import for a local eval."""
        result = platform_request(
            "/api/cli/eval-runs/imports",
            method="POST",
            data={
                "schema_version": 1,
                "local_run_id": local_run_id,
                "manifest_digest": manifest_digest,
                "run": run,
                "schema_versions": schema_versions,
                "provenance": provenance,
                "files": list(files),
            },
            credentials=credentials,
            git_identity=git_identity,
        )
        return EvalRunImportResult.from_dict(result)

    def get_eval_run_import_uploads(
        self,
        session_id: str,
        *,
        paths: Sequence[str],
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EvalRunImportUploads:
        """Return upload instructions for up to 100 files still missing."""
        if not 1 <= len(paths) <= 100:
            raise ValueError("Eval import upload batches must contain 1 to 100 paths")
        result = platform_request(
            f"/api/cli/eval-runs/imports/{_safe_path(session_id)}/uploads",
            method="POST",
            data={"paths": list(paths)},
            credentials=credentials,
            git_identity=git_identity,
        )
        return EvalRunImportUploads.from_dict(result)

    def complete_eval_run_import_upload(
        self,
        session_id: str,
        *,
        path: str,
        parts: list[dict[str, Any]] | None = None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> None:
        """Record one simple or multipart file upload as complete."""
        payload: dict[str, Any] = {"path": path}
        if parts is not None:
            normalized_parts = [
                {"part_number": part["PartNumber"], "etag": part["ETag"]}
                for part in parts
            ]
            part_numbers = [part["part_number"] for part in normalized_parts]
            if len(part_numbers) != len(set(part_numbers)):
                raise ValueError("Duplicate multipart part numbers")
            payload["parts"] = normalized_parts
        platform_request(
            f"/api/cli/eval-runs/imports/{_safe_path(session_id)}/uploads/complete",
            method="POST",
            data=payload,
            timeout=120.0 if parts is not None else 30.0,
            credentials=credentials,
            git_identity=git_identity,
        )

    def finalize_eval_run_import(
        self,
        session_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> EvalRunImportResult:
        """Finalize an import after all declared files have been uploaded."""
        result = platform_request(
            f"/api/cli/eval-runs/imports/{_safe_path(session_id)}/finalize",
            method="POST",
            data={},
            timeout=300.0,
            credentials=credentials,
            git_identity=git_identity,
        )
        return EvalRunImportResult.from_dict(result)

    def submit_benchmark_run(
        self,
        *,
        experiment_config: dict[str, Any],
        agents: list[dict[str, Any]],
        tasks_config: dict[str, Any] | None = None,
        execution_config: dict[str, Any] | None = None,
        env_config: dict[str, str] | None = None,
        secrets: dict[str, Any] | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> SubmitBenchmarkRunResult:
        """Submit a new benchmark run."""
        data: dict[str, Any] = {
            "experiment_config": experiment_config,
            "agents": agents,
        }
        if tasks_config:
            data["tasks_config"] = tasks_config
        if execution_config:
            data["execution_config"] = execution_config
        if env_config:
            data["env_config"] = env_config
        if secrets:
            data["secrets"] = secrets
        result = platform_request(
            "/api/cli/benchmark-runs",
            method="POST",
            data=data,
            credentials=credentials,
            git_identity=git_identity,
        )
        return SubmitBenchmarkRunResult.from_dict(result)

    def list_benchmarks(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> PaginatedBenchmarks:
        """List benchmarks available in the current workspace."""
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/benchmarks?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedBenchmarks.from_dict(data)

    def get_benchmark(
        self,
        name_or_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> BenchmarkCatalogDetail:
        """Get benchmark metadata and task-selection options."""
        data = platform_request(
            f"/api/cli/benchmarks/{_safe_path(name_or_id)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return BenchmarkCatalogDetail.from_dict(data)

    def list_benchmark_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        benchmark: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> PaginatedBenchmarkRuns:
        """List benchmark runs in the current workspace.

        `benchmark` (a benchmark ID) scopes the list to one benchmark's runs.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if benchmark is not None:
            params["benchmark"] = benchmark
        qs = urlencode(params)
        data = platform_request(
            f"/api/cli/benchmark-runs?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedBenchmarkRuns.from_dict(data)

    def get_benchmark_run(
        self,
        name_or_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> BenchmarkRunDetail:
        """Get benchmark run details by name or ID."""
        data = platform_request(
            f"/api/cli/benchmark-runs/{_safe_path(name_or_id)}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return BenchmarkRunDetail.from_dict(data)

    def get_benchmark_run_logs(
        self,
        name_or_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> LogsPage:
        """Fetch one cursor page of benchmark run logs."""
        return self._get_logs(
            f"/api/cli/benchmark-runs/{_safe_path(name_or_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_benchmark_run_download_manifest(
        self,
        benchmark_run_id: str,
        *,
        types: Sequence[str],
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadManifest:
        """Get the fixed-layout download manifest for a benchmark run."""
        return self._get_run_download_manifest(
            f"/api/cli/benchmark-runs/{_safe_path(benchmark_run_id)}",
            types=types,
            route="outputs",
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_benchmark_run_download_urls(
        self,
        benchmark_run_id: str,
        *,
        items: Sequence[RunDownloadFile],
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadURLBatch:
        """Exchange benchmark manifest items for bounded presigned URLs."""
        return self._get_run_download_urls(
            f"/api/cli/benchmark-runs/{_safe_path(benchmark_run_id)}",
            items=items,
            route="outputs",
            credentials=credentials,
            git_identity=git_identity,
        )

    def stop_benchmark_run(
        self,
        name_or_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> dict[str, Any]:
        """Stop a non-terminal benchmark run."""
        return platform_request(
            f"/api/cli/benchmark-runs/{_safe_path(name_or_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )

    def list_eval_runs(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
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
        git_identity: str | None,
    ) -> LogsPage:
        """Fetch one page of evaluation run logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page.
        With a cursor, ``next_cursor`` continues in the requested direction.
        """
        return self._get_logs(
            f"/api/cli/eval-runs/{_safe_path(name_or_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_eval_run_download_manifest(
        self,
        eval_run_id: str,
        *,
        types: Sequence[str],
        rows: str | None = None,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadManifest:
        return self._get_run_download_manifest(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}",
            types=types,
            rows=rows,
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_eval_run_download_urls(
        self,
        eval_run_id: str,
        *,
        items: Sequence[RunDownloadFile],
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> RunDownloadURLBatch:
        return self._get_run_download_urls(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}",
            items=items,
            credentials=credentials,
            git_identity=git_identity,
        )

    def stop_eval_run(
        self,
        eval_run_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str | None,
    ) -> dict[str, Any]:
        """Stop a non-terminal evaluation run (queued, pending, or running)."""
        return platform_request(
            f"/api/cli/eval-runs/{_safe_path(eval_run_id)}/stop",
            method="POST",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )

    # ── Dev Rollout Servers ───────────────────────────────────────

    def provision_dev_rollout_server(
        self,
        *,
        rollout_name: str,
        commit_sha: str,
        repository_path: str,
        entrypoint: str,
        ttl_hours: int | None,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> dict[str, Any]:
        return platform_request(
            "/api/cli/dev-rollout-server",
            method="POST",
            data={
                "rollout_name": rollout_name,
                "commit_sha": commit_sha,
                "repository_path": repository_path,
                "entrypoint": entrypoint,
                "ttl_hours": ttl_hours,
            },
            credentials=credentials,
            git_identity=git_identity,
        )

    def teardown_dev_rollout_server(
        self,
        server_id: str,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> dict[str, Any]:
        return platform_request(
            f"/api/cli/dev-rollout-server/{_safe_path(server_id)}",
            method="DELETE",
            data={},
            credentials=credentials,
            git_identity=git_identity,
        )

    def get_dev_rollout_server_logs(
        self,
        server_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        direction: str = "older",
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> LogsPage:
        """Fetch one page of dev rollout server logs.

        Without ``cursor``, ``direction="older"`` returns the most recent page;
        ``direction="newer"`` pages forward for live follow.
        """
        return self._get_logs(
            f"/api/cli/dev-rollout-server/{_safe_path(server_id)}",
            limit=limit,
            cursor=cursor,
            direction=direction,
            credentials=credentials,
            git_identity=git_identity,
        )

    def stream_dev_rollout_server_logs(
        self,
        server_id: str,
        *,
        tail: int = DEFAULT_PAGE_SIZE,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> Iterator[LogEntry]:
        """Stream a dev rollout server's logs live via Server-Sent Events.

        The server sends the most recent ``tail`` lines first, then pushes new
        lines as they arrive. The iterator ends when the stream closes (e.g. the
        server is torn down).
        """
        qs = urlencode({"tail": tail})
        for data in platform_stream(
            f"/api/cli/dev-rollout-server/{_safe_path(server_id)}/logs/stream?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        ):
            yield LogEntry.from_dict(data)

    def list_dev_rollout_servers(
        self,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        *,
        credentials: Credentials | None = None,
        git_identity: str,
    ) -> PaginatedDevRolloutServers:
        qs = urlencode({"limit": limit, "offset": offset})
        data = platform_request(
            f"/api/cli/dev-rollout-server?{qs}",
            credentials=credentials,
            git_identity=git_identity,
        )
        return PaginatedDevRolloutServers.from_dict(data)

    # ── Workspaces ────────────────────────────────────────────────

    def list_workspaces(
        self,
        *,
        credentials: Credentials | None = None,
    ) -> list[WorkspaceSummary]:
        data = platform_request(
            "/api/cli/workspaces",
            credentials=credentials,
            require_git_repo=False,
        )
        return [
            WorkspaceSummary.from_dict(workspace)
            for workspace in data.get("workspaces", [])
        ]

    # ── Quickstart ────────────────────────────────────────────────
    # Scoped by an explicit organization_id rather than a git identity: the
    # wizard runs before a workspace clone exists.

    def get_quickstart_status(
        self,
        organization_id: str,
        *,
        credentials: Credentials | None = None,
    ) -> QuickstartStatus:
        qs = urlencode({"organizationId": organization_id})
        data = platform_request(
            f"/api/cli/quickstart?{qs}",
            credentials=credentials,
            require_git_repo=False,
        )
        return QuickstartStatus.from_dict(data)

    def complete_quickstart(
        self,
        organization_id: str,
        intent: str,
        *,
        credentials: Credentials | None = None,
    ) -> None:
        """Record that the caller finished the quickstart wizard; idempotent."""
        platform_request(
            "/api/cli/quickstart",
            method="POST",
            data={"organizationId": organization_id, "intent": intent},
            credentials=credentials,
            require_git_repo=False,
        )

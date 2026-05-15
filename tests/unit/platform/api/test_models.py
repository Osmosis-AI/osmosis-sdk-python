"""Tests for osmosis_ai.platform.api.models."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.api import models as api_models
from osmosis_ai.platform.api.models import (
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_INACTIVE,
    STATUSES_SUCCESS,
    STATUSES_TERMINAL,
    DatasetDownloadInfo,
    DatasetFile,
    MetricDataPoint,
    MetricHistory,
    SubmitTrainingRunResult,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
    UploadInfo,
)

REMOVED_RESPONSE_MODELS = (
    "DeleteTrainingRunResult",
    "AffectedTrainingRun",
    "DatasetAffectedResources",
    "ModelAffectedResources",
    "WorkspaceDeletionStatus",
    "ProcessCount",
    "RenameDeploymentResult",
)


class TestRemovedResponseModels:
    """Deleted response models must not be exposed by the models module."""

    @pytest.mark.parametrize("model_name", REMOVED_RESPONSE_MODELS)
    def test_removed_response_model_is_not_exposed(self, model_name: str) -> None:
        assert not hasattr(api_models, model_name)


# =============================================================================
# UploadInfo Tests
# =============================================================================


class TestUploadInfo:
    """Tests for UploadInfo.from_dict."""

    def test_from_dict_simple(self) -> None:
        """Verify simple upload with presigned_url is parsed correctly."""
        data = {
            "method": "simple",
            "s3_key": "uploads/abc123.jsonl",
            "presigned_url": "https://s3.example.com/bucket/abc123?sig=xxx",
            "expires_in": 3600,
            "upload_headers": {"Content-Type": "application/octet-stream"},
        }
        info = UploadInfo.from_dict(data)
        assert info.method == "simple"
        assert info.s3_key == "uploads/abc123.jsonl"
        assert info.presigned_url == "https://s3.example.com/bucket/abc123?sig=xxx"
        assert info.expires_in == 3600
        assert info.upload_headers == {"Content-Type": "application/octet-stream"}
        assert info.upload_id is None
        assert info.part_size is None
        assert info.total_parts is None
        assert info.presigned_urls is None

    def test_from_dict_multipart(self) -> None:
        """Verify multipart upload with upload_id, part_size, total_parts, presigned_urls."""
        urls = [
            {"part_number": 1, "presigned_url": "https://s3.example.com/part1"},
            {"part_number": 2, "presigned_url": "https://s3.example.com/part2"},
            {"part_number": 3, "presigned_url": "https://s3.example.com/part3"},
        ]
        data = {
            "method": "multipart",
            "s3_key": "uploads/large-file.jsonl",
            "upload_id": "mp-upload-xyz",
            "part_size": 5242880,
            "total_parts": 3,
            "presigned_urls": urls,
        }
        info = UploadInfo.from_dict(data)
        assert info.method == "multipart"
        assert info.s3_key == "uploads/large-file.jsonl"
        assert info.upload_id == "mp-upload-xyz"
        assert info.part_size == 5242880
        assert info.total_parts == 3
        assert info.presigned_urls == urls
        assert info.presigned_url is None

    def test_from_dict_unknown_method(self) -> None:
        """Verify ValueError is raised for an unsupported upload method."""
        data = {"method": "chunked", "s3_key": "uploads/file.jsonl"}
        with pytest.raises(ValueError, match="Unknown upload method 'chunked'"):
            UploadInfo.from_dict(data)

    def test_from_dict_defaults_to_simple(self) -> None:
        """Verify missing method key defaults to 'simple'."""
        data = {"s3_key": "uploads/file.jsonl"}
        info = UploadInfo.from_dict(data)
        assert info.method == "simple"
        assert info.s3_key == "uploads/file.jsonl"


# =============================================================================
# DatasetFile.is_terminal Tests
# =============================================================================


class TestDatasetFileIsTerminal:
    """Tests for DatasetFile.is_terminal property."""

    @pytest.mark.parametrize(
        "status",
        ["uploaded", "error", "cancelled", "deleted"],
    )
    def test_terminal_statuses(self, status: str) -> None:
        """Verify terminal statuses return True."""
        ds = DatasetFile.from_dict(
            {"id": "ds-t", "file_name": "f.jsonl", "file_size": 100, "status": status}
        )
        assert ds.is_terminal is True

    @pytest.mark.parametrize(
        "status",
        ["processing", "pending", "uploading", ""],
    )
    def test_non_terminal_statuses(self, status: str) -> None:
        """Verify non-terminal statuses return False."""
        ds = DatasetFile.from_dict(
            {"id": "ds-nt", "file_name": "f.jsonl", "file_size": 100, "status": status}
        )
        assert ds.is_terminal is False


class TestDatasetDownloadInfo:
    """Tests for DatasetDownloadInfo.from_dict."""

    def test_from_dict_snake_case(self) -> None:
        info = DatasetDownloadInfo.from_dict(
            {
                "presigned_url": "https://example.com/data.jsonl",
                "expires_in": 3600,
                "file_name": "data.jsonl",
            }
        )
        assert info.presigned_url == "https://example.com/data.jsonl"
        assert info.expires_in == 3600
        assert info.file_name == "data.jsonl"

    def test_from_dict_camel_case(self) -> None:
        info = DatasetDownloadInfo.from_dict(
            {
                "presignedUrl": "https://example.com/data.csv",
                "expiresIn": 3600,
                "downloadFileName": "data.csv",
            }
        )
        assert info.presigned_url == "https://example.com/data.csv"
        assert info.expires_in == 3600
        assert info.file_name == "data.csv"


# =============================================================================
# Status Constants Tests
# =============================================================================


class TestStatusConstants:
    """Tests for module-level status frozenset constants."""

    def test_terminal_is_union(self) -> None:
        """Verify STATUSES_TERMINAL equals the union of success, error, and inactive."""
        assert (
            STATUSES_TERMINAL == STATUSES_SUCCESS | STATUSES_ERROR | STATUSES_INACTIVE
        )

    def test_no_overlap(self) -> None:
        """Verify no status appears in more than one category."""
        categories = [
            STATUSES_SUCCESS,
            STATUSES_IN_PROGRESS,
            STATUSES_ERROR,
            STATUSES_INACTIVE,
        ]
        for i, a in enumerate(categories):
            for b in categories[i + 1 :]:
                overlap = a & b
                assert overlap == frozenset(), (
                    f"Overlap found between categories: {overlap}"
                )


# =============================================================================
# Metric Data Model Tests
# =============================================================================


class TestSubmitTrainingRunResult:
    """Tests for SubmitTrainingRunResult.from_dict."""

    def test_from_dict(self) -> None:
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "my-training-run",
            "status": "pending",
            "created_at": "2026-04-10T12:00:00Z",
        }
        result = SubmitTrainingRunResult.from_dict(data)
        assert result.id == "550e8400-e29b-41d4-a716-446655440000"
        assert result.name == "my-training-run"
        assert result.status == "pending"
        assert result.created_at == "2026-04-10T12:00:00Z"


class TestTrainingRun:
    def test_from_dict_parses_reward(self) -> None:
        run = api_models.TrainingRun.from_dict(
            {
                "id": "run_1",
                "name": "reward-run",
                "status": "finished",
                "reward": 0.875,
            }
        )

        assert run.reward == 0.875

    def test_from_dict_defaults_missing_reward_to_none(self) -> None:
        run = api_models.TrainingRun.from_dict(
            {
                "id": "run_1",
                "name": "legacy-run",
                "status": "finished",
            }
        )

        assert run.reward is None

    def test_positional_constructor_preserves_reward_increase_delta(self) -> None:
        run = api_models.TrainingRun(
            "run_1", "run", "finished", None, None, "", None, None, 0.1, 0.2
        )

        assert run.reward_increase_delta == 0.2
        assert run.reward is None


class TestMetricDataPoint:
    def test_from_dict(self) -> None:
        data = {"step": 10, "value": 0.85, "timestamp": 1711800060000}
        point = MetricDataPoint.from_dict(data)
        assert point.step == 10
        assert point.value == 0.85
        assert point.timestamp == 1711800060000


class TestMetricHistory:
    def test_from_dict(self) -> None:
        data = {
            "metric_key": "rollout/raw_reward",
            "title": "Training Reward",
            "data_points": [
                {"step": 0, "value": 0.5, "timestamp": 1711800000000},
                {"step": 10, "value": 0.65, "timestamp": 1711800060000},
            ],
        }
        history = MetricHistory.from_dict(data)
        assert history.metric_key == "rollout/raw_reward"
        assert history.title == "Training Reward"
        assert len(history.data_points) == 2
        assert history.data_points[0].step == 0
        assert history.data_points[1].value == 0.65

    def test_from_dict_empty_data_points(self) -> None:
        data = {
            "metric_key": "rollout/raw_reward",
            "title": "Training Reward",
            "data_points": [],
        }
        history = MetricHistory.from_dict(data)
        assert history.data_points == []


class TestTrainingRunMetricsOverview:
    def test_from_dict_full(self) -> None:
        data = {
            "mlflow_run_id": "mlflow-abc",
            "mlflow_status": "FINISHED",
            "duration_ms": 3600000,
            "duration_formatted": "1h",
            "reward": 0.85,
            "reward_increase_delta": 0.15,
            "examples_processed_count": 5000,
        }
        overview = TrainingRunMetricsOverview.from_dict(data)
        assert overview.mlflow_run_id == "mlflow-abc"
        assert overview.duration_ms == 3600000
        assert overview.duration_formatted == "1h"
        assert overview.reward == 0.85
        assert overview.reward_delta == 0.15
        assert overview.examples_processed_count == 5000

    def test_from_dict_nulls(self) -> None:
        data = {
            "mlflow_run_id": "mlflow-xyz",
            "mlflow_status": "FAILED",
            "duration_ms": None,
            "duration_formatted": None,
            "reward": None,
            "reward_increase_delta": None,
            "examples_processed_count": None,
        }
        overview = TrainingRunMetricsOverview.from_dict(data)
        assert overview.duration_ms is None
        assert overview.reward is None
        assert overview.reward_delta is None


class TestTrainingRunMetrics:
    def test_from_dict(self) -> None:
        data = {
            "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "finished",
            "overview": {
                "mlflow_run_id": "mlflow-abc",
                "mlflow_status": "FINISHED",
                "duration_ms": 3600000,
                "duration_formatted": "1h",
                "reward": 0.85,
                "reward_increase_delta": 0.15,
                "examples_processed_count": 5000,
            },
            "metrics": [
                {
                    "metric_key": "rollout/raw_reward",
                    "title": "Training Reward",
                    "data_points": [
                        {"step": 0, "value": 0.5, "timestamp": 1711800000000},
                    ],
                },
            ],
        }
        result = TrainingRunMetrics.from_dict(data)
        assert result.training_run_id == "550e8400-e29b-41d4-a716-446655440000"
        assert result.status == "finished"
        assert result.overview.reward == 0.85
        assert len(result.metrics) == 1
        assert result.metrics[0].metric_key == "rollout/raw_reward"

    def test_from_dict_empty_metrics(self) -> None:
        data = {
            "training_run_id": "run-empty",
            "status": "finished",
            "overview": {
                "mlflow_run_id": "mlflow-empty",
                "mlflow_status": "FINISHED",
                "duration_ms": None,
                "duration_formatted": None,
                "reward": None,
                "reward_increase_delta": None,
                "examples_processed_count": None,
            },
            "metrics": [],
        }
        result = TrainingRunMetrics.from_dict(data)
        assert result.metrics == []


class TestDeploymentModels:
    def test_deployment_info_from_dict(self) -> None:
        from osmosis_ai.platform.api.models import DeploymentInfo

        d = DeploymentInfo.from_dict(
            {
                "id": "dep_1",
                "checkpoint_name": "qwen3-run1-step-100",
                "status": "active",
                "training_run_id": "run_1",
                "training_run_name": "qwen3-run1",
                "checkpoint_step": 100,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                "creator_name": "brian",
                "created_at": "2026-04-20T00:00:00Z",
            }
        )
        assert d.id == "dep_1"
        assert d.checkpoint_name == "qwen3-run1-step-100"
        assert d.status == "active"
        assert d.checkpoint_step == 100
        assert d.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    def test_deployment_info_minimal(self) -> None:
        """Server may omit optional fields — from_dict must tolerate it."""
        from osmosis_ai.platform.api.models import DeploymentInfo

        d = DeploymentInfo.from_dict(
            {
                "id": "dep_1",
                "checkpoint_name": "x",
                "status": "active",
                "base_model": "Qwen/Qwen3",
                "checkpoint_step": 0,
            }
        )
        assert d.training_run_id is None
        assert d.training_run_name is None
        assert d.creator_name is None
        assert d.created_at == ""

    def test_paginated_deployments_from_dict(self) -> None:
        from osmosis_ai.platform.api.models import PaginatedDeployments

        p = PaginatedDeployments.from_dict(
            {
                "deployments": [
                    {
                        "id": "dep_1",
                        "checkpoint_name": "a",
                        "status": "active",
                        "base_model": "Qwen/Qwen3",
                        "checkpoint_step": 1,
                    }
                ],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }
        )
        assert len(p.deployments) == 1
        assert p.total_count == 1
        assert p.has_more is False
        assert p.next_offset is None

    def test_deployment_summary_from_dict(self) -> None:
        from osmosis_ai.platform.api.models import DeploymentSummary

        s = DeploymentSummary.from_dict(
            {"id": "dep_1", "checkpoint_name": "x", "status": "active"}
        )
        assert s.id == "dep_1"
        assert s.checkpoint_name == "x"
        assert s.status == "active"

    def test_deployment_status_frozensets(self) -> None:
        from osmosis_ai.platform.api.models import (
            DEPLOYMENT_STATUSES_ERROR,
            DEPLOYMENT_STATUSES_INACTIVE,
            DEPLOYMENT_STATUSES_SUCCESS,
        )

        assert "active" in DEPLOYMENT_STATUSES_SUCCESS
        assert "inactive" in DEPLOYMENT_STATUSES_INACTIVE
        assert "failed" in DEPLOYMENT_STATUSES_ERROR

    def test_lora_checkpoint_info(self) -> None:
        from osmosis_ai.platform.api.models import LoraCheckpointInfo

        c = LoraCheckpointInfo.from_dict(
            {
                "id": "cp_1",
                "checkpoint_name": "qwen3-run1-step-100",
                "checkpoint_step": 100,
                "status": "uploaded",
                "created_at": "2026-04-20T00:00:00Z",
            }
        )
        assert c.checkpoint_step == 100
        assert c.status == "uploaded"
        assert c.checkpoint_name == "qwen3-run1-step-100"

    def test_lora_checkpoint_info_missing_name(self) -> None:
        """checkpoint_name may be absent on older platform deployments."""
        from osmosis_ai.platform.api.models import LoraCheckpointInfo

        c = LoraCheckpointInfo.from_dict(
            {
                "id": "cp_1",
                "checkpoint_step": 100,
                "status": "uploaded",
                "created_at": "2026-04-20T00:00:00Z",
            }
        )
        assert c.checkpoint_name == ""

    def test_training_run_checkpoints(self) -> None:
        from osmosis_ai.platform.api.models import TrainingRunCheckpoints

        r = TrainingRunCheckpoints.from_dict(
            {
                "training_run_id": "run_1",
                "training_run_name": "qwen3-run1",
                "checkpoints": [
                    {
                        "id": "cp_1",
                        "checkpoint_name": "qwen3-run1-step-100",
                        "checkpoint_step": 100,
                        "status": "uploaded",
                        "created_at": "2026-04-20T00:00:00Z",
                    },
                    {
                        "id": "cp_2",
                        "checkpoint_name": "qwen3-run1-step-200",
                        "checkpoint_step": 200,
                        "status": "uploaded",
                        "created_at": "2026-04-20T01:00:00Z",
                    },
                ],
            }
        )
        assert r.training_run_name == "qwen3-run1"
        assert len(r.checkpoints) == 2
        assert r.checkpoints[0].checkpoint_name == "qwen3-run1-step-100"
        assert r.checkpoints[1].checkpoint_step == 200
        assert r.checkpoints[1].checkpoint_name == "qwen3-run1-step-200"

"""Tests for osmosis_ai.platform.api.models."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.api.models import (
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_INACTIVE,
    STATUSES_SUCCESS,
    STATUSES_TERMINAL,
    AffectedTrainingRun,
    DatasetFile,
    MetricDataPoint,
    MetricHistory,
    ModelAffectedResources,
    ProjectDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
    UploadInfo,
    WorkspaceDeletionStatus,
)

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
# ProjectDetail Tests
# =============================================================================


class TestProjectDetail:
    """Tests for ProjectDetail.from_dict."""

    def test_from_dict_with_datasets(self) -> None:
        """Verify full data with 2 recent datasets is parsed correctly."""
        data = {
            "id": "proj-001",
            "project_name": "My Project",
            "role": "admin",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-06-15T12:00:00Z",
            "datasets": {
                "total_count": 42,
                "recent": [
                    {
                        "id": "ds-1",
                        "file_name": "train.jsonl",
                        "file_size": 1024,
                        "status": "ready",
                        "created_at": "2025-06-01T00:00:00Z",
                    },
                    {
                        "id": "ds-2",
                        "file_name": "eval.jsonl",
                        "file_size": 512,
                        "status": "processing",
                        "created_at": "2025-06-10T00:00:00Z",
                    },
                ],
            },
        }
        detail = ProjectDetail.from_dict(data)
        assert detail.id == "proj-001"
        assert detail.project_name == "My Project"
        assert detail.role == "admin"
        assert detail.created_at == "2025-01-01T00:00:00Z"
        assert detail.updated_at == "2025-06-15T12:00:00Z"
        assert detail.dataset_count == 42
        assert len(detail.recent_datasets) == 2
        assert detail.recent_datasets[0].id == "ds-1"
        assert detail.recent_datasets[0].file_name == "train.jsonl"
        assert detail.recent_datasets[1].id == "ds-2"
        assert detail.recent_datasets[1].status == "processing"

    def test_from_dict_empty_datasets(self) -> None:
        """Verify datasets key with empty recent list and zero count."""
        data = {
            "id": "proj-002",
            "project_name": "Empty Project",
            "role": "member",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "datasets": {"recent": [], "total_count": 0},
        }
        detail = ProjectDetail.from_dict(data)
        assert detail.dataset_count == 0
        assert detail.recent_datasets == []

    def test_from_dict_missing_datasets_key(self) -> None:
        """Verify missing 'datasets' key falls back to defaults."""
        data = {
            "id": "proj-003",
            "project_name": "No Datasets Key",
            "role": "viewer",
            "created_at": "2025-03-01T00:00:00Z",
            "updated_at": "2025-03-01T00:00:00Z",
        }
        detail = ProjectDetail.from_dict(data)
        assert detail.dataset_count == 0
        assert detail.recent_datasets == []

    def test_from_dict_default_role(self) -> None:
        """Verify missing 'role' key defaults to 'member'."""
        data = {
            "id": "proj-004",
            "project_name": "Default Role Project",
            "created_at": "2025-02-01T00:00:00Z",
            "updated_at": "2025-02-01T00:00:00Z",
        }
        detail = ProjectDetail.from_dict(data)
        assert detail.role == "member"


# =============================================================================
# DatasetFile.is_terminal Tests
# =============================================================================


class TestDatasetFileIsTerminal:
    """Tests for DatasetFile.is_terminal property."""

    @pytest.mark.parametrize(
        "status",
        ["ready", "failed", "error", "cancelled", "deleted"],
    )
    def test_terminal_statuses(self, status: str) -> None:
        """Verify terminal statuses return True."""
        ds = DatasetFile.from_dict(
            {"id": "ds-t", "file_name": "f.jsonl", "file_size": 100, "status": status}
        )
        assert ds.is_terminal is True

    @pytest.mark.parametrize(
        "status",
        ["processing", "uploaded", "pending", ""],
    )
    def test_non_terminal_statuses(self, status: str) -> None:
        """Verify non-terminal statuses return False."""
        ds = DatasetFile.from_dict(
            {"id": "ds-nt", "file_name": "f.jsonl", "file_size": 100, "status": status}
        )
        assert ds.is_terminal is False


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


class TestAffectedTrainingRun:
    """Tests for AffectedTrainingRun.from_dict."""

    def test_from_dict(self) -> None:
        data = {"id": "run-1", "name": "My Run", "project_name": "proj-a"}
        run = AffectedTrainingRun.from_dict(data)
        assert run.id == "run-1"
        assert run.name == "My Run"
        assert run.project_name == "proj-a"

    def test_from_dict_null_name(self) -> None:
        data = {"id": "run-2", "project_name": "proj-b"}
        run = AffectedTrainingRun.from_dict(data)
        assert run.name is None


class TestModelAffectedResources:
    """Tests for ModelAffectedResources.from_dict."""

    def test_empty(self) -> None:
        data = {"training_runs_using_model": [], "creator_training_run": None}
        res = ModelAffectedResources.from_dict(data)
        assert res.training_runs_using_model == []
        assert res.creator_training_run is None
        assert res.has_blocking_runs is False

    def test_with_blocking_runs(self) -> None:
        data = {
            "training_runs_using_model": [
                {"id": "r1", "name": "Run 1", "project_name": "proj"},
                {"id": "r2", "name": None, "project_name": "proj"},
            ],
            "creator_training_run": None,
        }
        res = ModelAffectedResources.from_dict(data)
        assert len(res.training_runs_using_model) == 2
        assert res.has_blocking_runs is True

    def test_with_creator_run(self) -> None:
        data = {
            "training_runs_using_model": [],
            "creator_training_run": {
                "id": "r3",
                "name": "Creator",
                "project_name": "proj",
            },
        }
        res = ModelAffectedResources.from_dict(data)
        assert res.creator_training_run is not None
        assert res.creator_training_run.id == "r3"
        assert res.has_blocking_runs is False


class TestWorkspaceDeletionStatus:
    """Tests for WorkspaceDeletionStatus.from_dict."""

    def test_can_delete(self) -> None:
        data = {
            "can_delete": True,
            "is_owner": True,
            "is_last_workspace": False,
            "projects": [],
        }
        status = WorkspaceDeletionStatus.from_dict(data)
        assert status.can_delete is True
        assert status.is_owner is True
        assert status.is_last_workspace is False
        assert status.projects == []
        assert status.projects_with_running_processes == []

    def test_with_running_processes(self) -> None:
        data = {
            "can_delete": False,
            "is_owner": True,
            "is_last_workspace": False,
            "projects": [
                {
                    "project_id": "p1",
                    "project_name": "Project A",
                    "has_running_processes": True,
                    "feature_pipelines": {"count": 2, "valid": False},
                    "training_runs": {"count": 0, "valid": True},
                    "models": {"count": 1, "valid": False},
                },
                {
                    "project_id": "p2",
                    "project_name": "Project B",
                    "has_running_processes": False,
                    "feature_pipelines": {"count": 0, "valid": True},
                    "training_runs": {"count": 0, "valid": True},
                    "models": {"count": 0, "valid": True},
                },
            ],
        }
        status = WorkspaceDeletionStatus.from_dict(data)
        assert status.can_delete is False
        assert len(status.projects) == 2
        blocking = status.projects_with_running_processes
        assert len(blocking) == 1
        assert blocking[0].project_name == "Project A"
        assert blocking[0].feature_pipelines.count == 2
        assert blocking[0].feature_pipelines.valid is False

    def test_not_owner(self) -> None:
        data = {
            "can_delete": False,
            "is_owner": False,
            "is_last_workspace": False,
            "projects": [],
        }
        status = WorkspaceDeletionStatus.from_dict(data)
        assert status.can_delete is False
        assert status.is_owner is False


# =============================================================================
# Metric Data Model Tests
# =============================================================================


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

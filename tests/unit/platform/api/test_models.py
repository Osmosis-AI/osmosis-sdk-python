"""Tests for osmosis_ai.platform.api.models."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.api.models import (
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_INACTIVE,
    STATUSES_SUCCESS,
    STATUSES_TERMINAL,
    DatasetFile,
    ProjectDetail,
    UploadInfo,
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

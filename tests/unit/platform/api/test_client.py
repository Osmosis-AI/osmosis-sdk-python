"""Tests for osmosis_ai.platform.api.client.OsmosisClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.platform.api.client import OsmosisClient, _safe_path


class TestCompleteUploadValidation:
    """Tests for parameter validation in OsmosisClient.complete_upload."""

    def test_upload_id_without_parts_raises(self) -> None:
        """Verify ValueError when upload_id is provided but parts is None."""
        client = OsmosisClient()
        with pytest.raises(
            ValueError, match="upload_id and parts must both be provided"
        ):
            client.complete_upload(
                file_id="file-1",
                s3_key="uploads/file.jsonl",
                upload_id="abc",
                parts=None,
            )

    def test_parts_without_upload_id_raises(self) -> None:
        """Verify ValueError when parts is provided but upload_id is None."""
        client = OsmosisClient()
        with pytest.raises(
            ValueError, match="upload_id and parts must both be provided"
        ):
            client.complete_upload(
                file_id="file-1",
                s3_key="uploads/file.jsonl",
                upload_id=None,
                parts=[{"PartNumber": 1, "ETag": "etag-1"}],
            )

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_both_none_ok(self, mock_request: MagicMock) -> None:
        """Verify no error when both upload_id and parts are None (simple upload)."""
        mock_request.return_value = {
            "id": "file-1",
            "file_name": "file.jsonl",
            "file_size": 1024,
            "status": "uploaded",
        }
        client = OsmosisClient()
        result = client.complete_upload(
            file_id="file-1",
            s3_key="uploads/file.jsonl",
            upload_id=None,
            parts=None,
        )
        assert result.id == "file-1"
        assert result.status == "uploaded"
        mock_request.assert_called_once()

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_both_provided_ok(self, mock_request: MagicMock) -> None:
        """Verify no error when both upload_id and parts are provided (multipart)."""
        mock_request.return_value = {
            "id": "file-2",
            "file_name": "large.jsonl",
            "file_size": 52428800,
            "status": "uploaded",
        }
        parts = [
            {"PartNumber": 1, "ETag": "etag-aaa"},
            {"PartNumber": 2, "ETag": "etag-bbb"},
        ]
        client = OsmosisClient()
        result = client.complete_upload(
            file_id="file-2",
            s3_key="uploads/large.jsonl",
            upload_id="abc",
            parts=parts,
        )
        assert result.id == "file-2"
        assert result.status == "uploaded"
        # Verify multipart payload was passed correctly
        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert payload["upload_id"] == "abc"
        assert payload["parts"] == parts
        # Verify multipart timeout (120s)
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 120.0


class TestSafePath:
    """Tests for the _safe_path helper used to prevent path traversal."""

    def test_normal_id_unchanged(self) -> None:
        assert _safe_path("abc-123") == "abc-123"

    def test_uuid_unchanged(self) -> None:
        assert (
            _safe_path("550e8400-e29b-41d4-a716-446655440000")
            == "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_slash_encoded(self) -> None:
        assert _safe_path("../etc/passwd") == "..%2Fetc%2Fpasswd"

    def test_dot_dot_slash_encoded(self) -> None:
        result = _safe_path("../../secret")
        assert "/" not in result
        assert result == "..%2F..%2Fsecret"

    def test_space_encoded(self) -> None:
        assert _safe_path("my file") == "my%20file"


class TestURLPathSafety:
    """Verify that OsmosisClient methods apply _safe_path to URL path segments."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_project_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "id": "p1",
            "project_name": "test",
        }
        client = OsmosisClient()
        client.get_project("../admin")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/projects/..%2Fadmin"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_complete_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "id": "f1",
            "file_name": "f.jsonl",
            "file_size": 100,
            "status": "uploaded",
        }
        client = OsmosisClient()
        client.complete_upload(file_id="a/b", s3_key="key")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/complete"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="a/b", upload_id="uid")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/abort"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_dataset_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "id": "f1",
            "file_name": "f.jsonl",
            "file_size": 100,
            "status": "uploaded",
        }
        client = OsmosisClient()
        client.get_dataset("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_delete_dataset_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.delete_dataset("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_training_run_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run": {"id": "r1", "status": "running"},
        }
        client = OsmosisClient()
        client.get_training_run("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb"

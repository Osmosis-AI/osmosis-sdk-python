"""Tests for osmosis_ai.platform.api.client.OsmosisClient.complete_upload validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.platform.api.client import OsmosisClient


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

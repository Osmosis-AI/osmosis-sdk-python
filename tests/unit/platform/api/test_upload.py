"""Tests for osmosis_ai.platform.api.upload — S3 upload with httpx."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from osmosis_ai.platform.api.models import UploadInfo
from osmosis_ai.platform.api.upload import (
    SliceFileObj,
    _http_put_with_backoff,
    _is_loopback_url,
    _require_https,
    upload_file_multipart,
    upload_file_simple,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_upload_info(**overrides) -> UploadInfo:
    """Create an UploadInfo with sensible defaults, applying any overrides."""
    defaults = dict(
        method="simple",
        s3_key="datasets/test-file.jsonl",
        presigned_url=None,
        expires_in=3600,
        upload_headers=None,
        upload_id=None,
        part_size=None,
        total_parts=None,
        presigned_urls=None,
    )
    defaults.update(overrides)
    return UploadInfo(**defaults)


def _make_response(
    status_code: int = 200, headers: dict | None = None, text: str = ""
) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    return resp


# =============================================================================
# TestRequireHttps
# =============================================================================


class TestRequireHttps:
    """Tests for _require_https and _is_loopback_url."""

    def test_https_url_passes(self) -> None:
        _require_https("https://s3.amazonaws.com/bucket/key")

    def test_http_non_loopback_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must use HTTPS"):
            _require_https("http://s3.amazonaws.com/bucket/key")

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:4566/bucket/key",
            "http://127.0.0.1:4566/bucket/key",
            "http://[::1]:4566/bucket/key",
        ],
    )
    def test_http_loopback_allowed(self, url: str) -> None:
        """HTTP is allowed for loopback addresses (local development)."""
        _require_https(url)  # should not raise

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://localhost:4566/x", True),
            ("http://127.0.0.1:4566/x", True),
            ("http://[::1]:4566/x", True),
            ("http://s3.amazonaws.com/x", False),
            ("http://192.168.1.1:4566/x", False),
        ],
    )
    def test_is_loopback_url(self, url: str, expected: bool) -> None:
        assert _is_loopback_url(url) is expected


# =============================================================================
# TestSliceFileObj
# =============================================================================


class TestSliceFileObj:
    """Tests for the SliceFileObj wrapper that reads a file slice."""

    def _make_file(self, size: int = 100) -> BytesIO:
        """Create a BytesIO filled with sequential byte values."""
        return BytesIO((bytes(range(256)) * ((size // 256) + 1))[:size])

    def test_read_full_slice(self) -> None:
        """read() with no argument returns exactly read_limit bytes from the offset."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            data = sfo.read()
        assert len(data) == 20
        assert data == bytes(range(10, 30))

    def test_read_with_limit(self) -> None:
        """Successive read(n) calls return correct chunks."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            first = sfo.read(5)
            second = sfo.read(5)
        assert first == bytes(range(10, 15))
        assert second == bytes(range(15, 20))

    def test_read_past_end(self) -> None:
        """After reading all bytes in the slice, read() returns empty bytes."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            sfo.read()  # consume all
            tail = sfo.read()
        assert tail == b""

    def test_tell_tracks_position(self) -> None:
        """tell() reflects the number of bytes read so far."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=0, read_limit=50)
        with sfo:
            sfo.read(10)
            assert sfo.tell() == 10

    def test_seek_absolute(self) -> None:
        """seek(offset, 0) sets position from the start of the slice."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            sfo.seek(5, 0)
            assert sfo.tell() == 5

    def test_seek_relative(self) -> None:
        """seek(offset, 1) adjusts position relative to current."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            sfo.read(10)
            sfo.seek(-5, 1)
            assert sfo.tell() == 5

    def test_seek_from_end(self) -> None:
        """seek(offset, 2) sets position relative to the end of the slice."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            sfo.seek(-5, 2)
            assert sfo.tell() == 15  # read_limit(20) - 5

    def test_seek_clamp(self) -> None:
        """seek beyond read_limit clamps to read_limit; before 0 clamps to 0."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=10, read_limit=20)
        with sfo:
            sfo.seek(999, 0)
            assert sfo.tell() == 20
            sfo.seek(-999, 0)
            assert sfo.tell() == 0

    def test_readable_writable(self) -> None:
        """readable() returns True; writable() returns False."""
        fobj = BytesIO(bytes(range(100)))
        sfo = SliceFileObj(fobj, seek_from=0, read_limit=10)
        assert sfo.readable() is True
        assert sfo.writable() is False


# =============================================================================
# TestHttpPutWithBackoff
# =============================================================================


class TestHttpPutWithBackoff:
    """Tests for _http_put_with_backoff retry logic."""

    @staticmethod
    def _make_mock_client() -> MagicMock:
        """Create a mock httpx.Client."""
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        return client

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_success_first_attempt(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Successful 200 on the first attempt returns the response."""
        resp = _make_response(200)
        ctx = self._make_mock_client()
        ctx.put.return_value = resp
        mock_client_cls.return_value = ctx

        result = _http_put_with_backoff("https://s3.example.com/obj", data=b"hello")

        assert result is resp
        mock_sleep.assert_not_called()

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_retry_then_success(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """First attempt returns 503, second returns 200; sleep called once."""
        resp_503 = _make_response(503, text="Service Unavailable")
        resp_200 = _make_response(200)

        ctx = self._make_mock_client()
        ctx.put.side_effect = [resp_503, resp_200]
        mock_client_cls.return_value = ctx

        result = _http_put_with_backoff("https://s3.example.com/obj", data=b"data")

        assert result.status_code == 200
        mock_sleep.assert_called_once()

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_non_retryable_error(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """A 403 response raises RuntimeError immediately without retries."""
        resp_403 = _make_response(403, text="Forbidden")

        ctx = self._make_mock_client()
        ctx.put.return_value = resp_403
        mock_client_cls.return_value = ctx

        with pytest.raises(RuntimeError, match="Upload failed: HTTP 403"):
            _http_put_with_backoff("https://s3.example.com/obj", data=b"data")

        mock_sleep.assert_not_called()

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_retries_exhausted(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Repeated 500 responses exhaust retries and raise RuntimeError."""
        resp_500 = _make_response(500, text="Internal Server Error")

        ctx = self._make_mock_client()
        ctx.put.return_value = resp_500
        mock_client_cls.return_value = ctx

        with pytest.raises(RuntimeError, match="after 3 attempts"):
            _http_put_with_backoff(
                "https://s3.example.com/obj", data=b"data", max_retries=3
            )

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_rewinds_file_like_data(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """On retry, seek() is called to rewind file-like data objects."""
        resp_503 = _make_response(503, text="Retry")
        resp_200 = _make_response(200)

        ctx = self._make_mock_client()
        ctx.put.side_effect = [resp_503, resp_200]
        mock_client_cls.return_value = ctx

        data = MagicMock()
        data.tell.return_value = 0
        data.seek = MagicMock()

        _http_put_with_backoff("https://s3.example.com/obj", data=data)

        # seek should have been called to rewind before the second attempt
        data.seek.assert_called_with(0)

    @patch("osmosis_ai.platform.api.upload.time.sleep")
    @patch("osmosis_ai.platform.api.upload.httpx.Client")
    def test_network_error_retries(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """httpx.NetworkError triggers a retry."""
        resp_200 = _make_response(200)

        ctx = self._make_mock_client()
        ctx.put.side_effect = [httpx.NetworkError("connection reset"), resp_200]
        mock_client_cls.return_value = ctx

        result = _http_put_with_backoff("https://s3.example.com/obj", data=b"data")

        assert result.status_code == 200
        mock_sleep.assert_called_once()


# =============================================================================
# TestUploadFileMultipart
# =============================================================================


class TestUploadFileMultipart:
    """Tests for upload_file_multipart."""

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_missing_presigned_urls_raises(
        self, mock_put: MagicMock, tmp_path: Path
    ) -> None:
        """RuntimeError when presigned_urls is None."""
        info = _make_upload_info(
            method="multipart",
            presigned_urls=None,
            total_parts=2,
            part_size=1024,
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 2048)

        with pytest.raises(RuntimeError, match="missing multipart fields"):
            upload_file_multipart(file_path, info)

        mock_put.assert_not_called()

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_missing_part_size_raises(
        self, mock_put: MagicMock, tmp_path: Path
    ) -> None:
        """RuntimeError when part_size is None but presigned_urls is present."""
        info = _make_upload_info(
            method="multipart",
            presigned_urls=[
                {"part_number": 1, "presigned_url": "https://s3/part1"},
            ],
            total_parts=1,
            part_size=None,
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 1024)

        with pytest.raises(RuntimeError, match="missing part_size"):
            upload_file_multipart(file_path, info)

        mock_put.assert_not_called()

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_malformed_url_entry_raises(
        self, mock_put: MagicMock, tmp_path: Path
    ) -> None:
        """RuntimeError when a presigned URL entry is missing presigned_url."""
        info = _make_upload_info(
            method="multipart",
            presigned_urls=[{"part_number": 1}],  # missing presigned_url
            total_parts=1,
            part_size=1024,
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 1024)

        with pytest.raises(RuntimeError, match="Malformed presigned URL entry"):
            upload_file_multipart(file_path, info)

        mock_put.assert_not_called()

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_successful_upload(self, mock_put: MagicMock, tmp_path: Path) -> None:
        """Two-part upload returns sorted completed parts with ETags."""
        info = _make_upload_info(
            method="multipart",
            presigned_urls=[
                {"part_number": 2, "presigned_url": "https://s3/part2"},
                {"part_number": 1, "presigned_url": "https://s3/part1"},
            ],
            total_parts=2,
            part_size=10,
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"a" * 20)

        resp_part1 = _make_response(200, headers={"etag": '"etag-part1"'})
        resp_part2 = _make_response(200, headers={"etag": '"etag-part2"'})
        mock_put.side_effect = [resp_part1, resp_part2]

        result = upload_file_multipart(file_path, info)

        assert len(result) == 2
        assert result[0] == {"PartNumber": 1, "ETag": "etag-part1"}
        assert result[1] == {"PartNumber": 2, "ETag": "etag-part2"}
        assert mock_put.call_count == 2

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_missing_etag_raises(self, mock_put: MagicMock, tmp_path: Path) -> None:
        """RuntimeError when S3 response does not include an ETag header."""
        info = _make_upload_info(
            method="multipart",
            presigned_urls=[
                {"part_number": 1, "presigned_url": "https://s3/part1"},
            ],
            total_parts=1,
            part_size=1024,
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 512)

        resp_no_etag = _make_response(200, headers={})
        mock_put.return_value = resp_no_etag

        with pytest.raises(RuntimeError, match="did not return an ETag"):
            upload_file_multipart(file_path, info)


# =============================================================================
# TestUploadFileSimple
# =============================================================================


class TestUploadFileSimple:
    """Tests for upload_file_simple."""

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_missing_presigned_url_raises(
        self, mock_put: MagicMock, tmp_path: Path
    ) -> None:
        """RuntimeError when presigned_url is None."""
        info = _make_upload_info(method="simple", presigned_url=None)
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 100)

        with pytest.raises(RuntimeError, match="missing presigned_url"):
            upload_file_simple(file_path, info)

        mock_put.assert_not_called()

    @patch("osmosis_ai.platform.api.upload._http_put_with_backoff")
    def test_successful_upload(self, mock_put: MagicMock, tmp_path: Path) -> None:
        """Verify _http_put_with_backoff is called with correct headers."""
        info = _make_upload_info(
            method="simple",
            presigned_url="https://s3.example.com/upload",
            upload_headers={"x-amz-meta-key": "value"},
        )
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"x" * 256)

        upload_file_simple(file_path, info)

        mock_put.assert_called_once()
        args, kwargs = mock_put.call_args
        # url is the first positional argument
        assert args[0] == "https://s3.example.com/upload"
        # headers are passed as a keyword argument
        headers = kwargs["headers"]
        assert headers["Content-Length"] == "256"
        assert headers["x-amz-meta-key"] == "value"

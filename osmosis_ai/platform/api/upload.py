"""S3 upload with httpx — simple PUT and multipart support.

Design aligned with huggingface_hub: sequential uploads, single file handle
for multipart, unified retry via _http_put_with_backoff.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from io import RawIOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

# ── URL validation ────────────────────────────────────────────────────


def _require_https(url: str, context: str = "Upload URL") -> None:
    """Reject non-HTTPS presigned URLs to prevent plaintext data transmission."""
    if not url.lower().startswith("https://"):
        raise RuntimeError(
            f"{context} must use HTTPS (got {url[:60]!r}). "
            "Refusing to upload over an insecure connection."
        )


if TYPE_CHECKING:
    from typing import IO

    from .models import UploadInfo

# ── Constants ─────────────────────────────────────────────────────────

PART_UPLOAD_TIMEOUT = 300.0  # 5 minutes per part
SIMPLE_UPLOAD_TIMEOUT = 600.0  # 10 minutes for simple upload

MAX_RETRIES = 5
BACKOFF_BASE = 1.0  # seconds
BACKOFF_CAP = 8.0  # seconds

_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

ProgressCallback = Callable[[int, int], None]  # (bytes_completed, total_bytes)


# ── SliceFileObj — virtual file slice (ported from huggingface_hub) ───


class SliceFileObj(RawIOBase):
    """Wrap a file object to read only a slice [seek_from, seek_from + read_limit).

    Ported from huggingface_hub/utils/_lfs.py.  Allows streaming a single
    part of a multipart upload from one open file descriptor without reading
    the entire part into memory.
    """

    def __init__(self, fobj: IO[bytes], seek_from: int, read_limit: int) -> None:
        self._fobj = fobj
        self._seek_from = seek_from
        self._read_limit = read_limit
        self._bytes_read = 0

    # -- context manager --------------------------------------------------

    def __enter__(self) -> SliceFileObj:
        self._fobj.seek(self._seek_from)
        self._bytes_read = 0
        return self

    def __exit__(self, *_: object) -> None:
        pass

    # -- file-like interface ----------------------------------------------

    def read(self, n: int = -1) -> bytes:  # type: ignore[override]
        remaining = self._read_limit - self._bytes_read
        if remaining <= 0:
            return b""
        if n < 0 or n > remaining:
            n = remaining
        data = self._fobj.read(n)
        if data:
            self._bytes_read += len(data)
        return data if data else b""

    def tell(self) -> int:
        return self._bytes_read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            # Absolute — relative to slice start
            self._bytes_read = max(0, min(offset, self._read_limit))
            self._fobj.seek(self._seek_from + self._bytes_read)
        elif whence == 1:
            # Relative to current position
            self._bytes_read = max(0, min(self._bytes_read + offset, self._read_limit))
            self._fobj.seek(self._seek_from + self._bytes_read)
        elif whence == 2:
            # Relative to end of slice
            self._bytes_read = max(0, min(self._read_limit + offset, self._read_limit))
            self._fobj.seek(self._seek_from + self._bytes_read)
        return self._bytes_read

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


# ── Retry helper (inspired by huggingface_hub http_backoff) ──────────


def _http_put_with_backoff(
    url: str,
    *,
    data: Any,
    headers: dict[str, str] | None = None,
    timeout: float = PART_UPLOAD_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    client: httpx.Client | None = None,
) -> httpx.Response:
    """PUT *data* to *url* with exponential backoff on transient errors.

    If *data* is file-like (has ``seek``), it is rewound before each retry
    so the same slice can be re-uploaded.

    An optional *client* can be passed to reuse a single httpx.Client across
    multiple calls (e.g. multipart upload parts), avoiding per-call overhead.
    """
    initial_pos: int | None = None
    if hasattr(data, "seek") and hasattr(data, "tell"):
        initial_pos = data.tell()

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0:
            delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
            time.sleep(delay)
            # Rewind file-like data for retry
            if initial_pos is not None:
                data.seek(initial_pos)

        try:
            if client is not None:
                resp = client.put(url, headers=headers, content=data)
            else:
                with httpx.Client(
                    timeout=httpx.Timeout(timeout, connect=30.0)
                ) as _client:
                    resp = _client.put(url, headers=headers, content=data)

            if 200 <= resp.status_code < 300:
                return resp

            if resp.status_code in _RETRY_STATUS_CODES:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            # Non-retryable HTTP error
            raise RuntimeError(
                f"Upload failed: HTTP {resp.status_code}. {resp.text[:500]}"
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc

    raise RuntimeError(f"Upload failed after {max_retries} attempts: {last_error}")


# ── Progress wrapper ──────────────────────────────────────────────


class _ProgressReader:
    """Wraps a file object to invoke a callback as httpx reads chunks.

    Implements __iter__ so httpx accepts it as streaming content.
    """

    _CHUNK_SIZE = 64 * 1024  # 64 KiB

    def __init__(
        self,
        fp: Any,
        total: int,
        callback: ProgressCallback,
    ) -> None:
        self._fp = fp
        self._total = total
        self._callback = callback
        self._bytes_read = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fp, name)

    def seek(self, pos: int, *args: Any) -> Any:
        self._bytes_read = 0
        return self._fp.seek(pos, *args)

    def read(self, size: int = -1) -> bytes:
        data = self._fp.read(size)
        if data:
            self._bytes_read += len(data)
            self._callback(self._bytes_read, self._total)
        return data

    def __iter__(self) -> _ProgressReader:
        return self

    def __next__(self) -> bytes:
        data = self._fp.read(self._CHUNK_SIZE)
        if not data:
            raise StopIteration
        self._bytes_read += len(data)
        self._callback(self._bytes_read, self._total)
        return data


# ── Simple upload ─────────────────────────────────────────────────────


def upload_file_simple(
    file_path: Path,
    upload_info: UploadInfo,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Upload a file via a single presigned PUT URL.

    Args:
        file_path: Local file to upload.
        upload_info: Upload instructions with presigned_url and upload_headers.
        progress_callback: Optional callback(bytes_uploaded, total_bytes).

    Raises:
        RuntimeError: If upload fails after retries.
    """
    info = upload_info
    if info.presigned_url is None:
        raise RuntimeError("UploadInfo missing presigned_url for simple upload")

    file_size = file_path.stat().st_size
    url = info.presigned_url
    _require_https(url, "Simple upload presigned URL")

    headers = dict(info.upload_headers or {})
    # S3 presigned PUT requires Content-Length (does not support chunked encoding)
    headers["Content-Length"] = str(file_size)

    with open(file_path, "rb") as f:
        content: Any = f
        if progress_callback:
            content = _ProgressReader(f, file_size, progress_callback)
        _http_put_with_backoff(
            url,
            data=content,
            headers=headers,
            timeout=SIMPLE_UPLOAD_TIMEOUT,
        )

    # Ensure progress shows 100% on success
    if progress_callback:
        progress_callback(file_size, file_size)


# ── Multipart upload ─────────────────────────────────────────────────


def upload_file_multipart(
    file_path: Path,
    upload_info: UploadInfo,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Sequential multipart upload to S3 using a single file handle.

    Each part is streamed via SliceFileObj — no full-part buffer in memory.

    Args:
        file_path: Local file to upload.
        upload_info: Upload instructions with multipart fields.
        progress_callback: Optional callback(bytes_uploaded, total_bytes).

    Returns:
        Sorted list of {"PartNumber": int, "ETag": str} dicts.

    Raises:
        RuntimeError: If any part fails.
    """
    info = upload_info
    if not info.presigned_urls or not info.total_parts:
        raise RuntimeError("UploadInfo missing multipart fields")

    file_size = file_path.stat().st_size
    if not info.part_size:
        raise RuntimeError(
            "UploadInfo missing part_size for multipart upload; "
            "cannot safely compute byte offsets"
        )
    part_size = info.part_size

    # Validate file size is sufficient for the part configuration
    min_required_size = part_size * (info.total_parts - 1) + 1
    if file_size < min_required_size:
        raise RuntimeError(
            f"File size ({file_size} bytes) is too small for multipart upload "
            f"configuration (part_size={part_size}, total_parts={info.total_parts}). "
            f"Minimum required: {min_required_size} bytes."
        )

    # Build part_number → presigned_url mapping
    url_map: dict[int, str] = {}
    for entry in info.presigned_urls:
        pn = entry.get("partNumber")
        url = entry.get("presignedUrl")
        if pn is None or url is None:
            raise RuntimeError(f"Malformed presigned URL entry from server: {entry!r}")
        _require_https(url, f"Multipart presigned URL (part {pn})")
        url_map[pn] = url

    # Verify server returned URLs for every expected part
    missing = [i for i in range(1, info.total_parts + 1) if i not in url_map]
    if missing:
        raise RuntimeError(
            f"Server returned presigned URLs for {len(url_map)} parts, "
            f"but {info.total_parts} expected. Missing: {missing[:10]}"
        )

    completed_parts: list[dict] = []
    bytes_uploaded = 0

    with (
        open(file_path, "rb") as f,
        httpx.Client(
            timeout=httpx.Timeout(PART_UPLOAD_TIMEOUT, connect=30.0)
        ) as client,
    ):
        for i in range(info.total_parts):
            part_number = i + 1
            offset = i * part_size
            size = min(part_size, file_size - offset)

            with SliceFileObj(f, seek_from=offset, read_limit=size) as slice_obj:
                resp = _http_put_with_backoff(
                    url_map[part_number],
                    data=slice_obj,
                    timeout=PART_UPLOAD_TIMEOUT,
                    client=client,
                )

            etag = resp.headers.get("etag", "").strip('"')
            if not etag:
                raise RuntimeError(
                    f"Part {part_number}: S3 did not return an ETag header"
                )
            completed_parts.append({"PartNumber": part_number, "ETag": etag})

            bytes_uploaded += size
            if progress_callback:
                progress_callback(bytes_uploaded, file_size)

    completed_parts.sort(key=lambda p: p["PartNumber"])
    return completed_parts


# ── Progress bar helper ───────────────────────────────────────────────


def make_progress_bar(
    file_size: int,
) -> tuple[AbstractContextManager[Any], ProgressCallback]:
    """Create a progress bar and a matching callback.

    Returns (progress_context, callback) where progress_context is a
    context manager wrapping a rich Progress bar (or a no-op nullcontext
    for plain-text fallback).

    The caller should use::

        ctx, cb = make_progress_bar(size)
        with ctx:
            upload_fn(..., progress_callback=cb)
    """
    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TransferSpeedColumn,
        )
        from rich.text import Text

        class _SpeedCol(TransferSpeedColumn):
            def render(self, task: object) -> Text:
                speed = task.finished_speed or task.speed  # type: ignore[union-attr]
                if speed is None:
                    return Text("-", style="progress.data.speed")
                return super().render(task)  # type: ignore[arg-type]

        progress = Progress(
            "[progress.percentage]{task.percentage:>3.0f}%",
            BarColumn(),
            DownloadColumn(binary_units=True),
            _SpeedCol(),
        )
        task_id = progress.add_task("Uploading", total=file_size)

        def _rich_cb(uploaded: int, total: int) -> None:
            progress.update(task_id, completed=uploaded)

        return progress, _rich_cb
    except ImportError:
        pass

    # Plain-text fallback
    def _plain_cb(uploaded: int, total: int) -> None:
        pct = uploaded * 100 // total if total else 0
        sys.stdout.write(f"\rUploading: {pct}%")
        sys.stdout.flush()
        if uploaded >= total:
            sys.stdout.write("\n")

    return nullcontext(), _plain_cb

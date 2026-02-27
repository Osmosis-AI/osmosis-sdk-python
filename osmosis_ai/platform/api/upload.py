"""S3 upload with httpx — simple PUT and multipart support."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .models import UploadInfo

# ── Constants ─────────────────────────────────────────────────────────

MAX_CONCURRENCY = 6
PART_UPLOAD_TIMEOUT = 300.0  # 5 minutes per part
SIMPLE_UPLOAD_TIMEOUT = 600.0  # 10 minutes for simple upload

MAX_RETRIES_SIMPLE = 3
MAX_RETRIES_MULTIPART = 5
BACKOFF_BASE = 2  # seconds
BACKOFF_MAX = 30  # seconds
BACKOFF_MAX_SIMPLE = 10  # shorter cap for simple (single-request) uploads

ProgressCallback = Callable[[int, int], None]  # (bytes_completed, total_bytes)


# ── Retry helper ──────────────────────────────────────────────────────


def _backoff_delay(
    attempt: int, base: float = BACKOFF_BASE, cap: float = BACKOFF_MAX
) -> float:
    """Exponential backoff with full jitter."""
    exp = min(base * (2**attempt), cap)
    return random.uniform(0, exp)


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

    headers = dict(info.upload_headers or {})
    chunk_size = 256 * 1024  # 256 KB

    def _stream():
        uploaded = 0
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                uploaded += len(chunk)
                if progress_callback:
                    progress_callback(uploaded, file_size)
                yield chunk

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES_SIMPLE):
        if attempt > 0:
            delay = _backoff_delay(
                attempt - 1, base=BACKOFF_BASE, cap=BACKOFF_MAX_SIMPLE
            )
            time.sleep(delay)
            # Reset progress bar so the user sees the retry from 0%
            if progress_callback:
                progress_callback(0, file_size)

        try:
            with httpx.Client(
                timeout=httpx.Timeout(SIMPLE_UPLOAD_TIMEOUT, connect=30.0)
            ) as client:
                resp = client.put(url, headers=headers, content=_stream())
                if resp.status_code >= 300:
                    raise RuntimeError(
                        f"Upload failed: HTTP {resp.status_code}. {resp.text[:500]}"
                    )
            return  # success
        except (httpx.HTTPError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Upload failed after {MAX_RETRIES_SIMPLE} attempts: {last_error}"
    )


# ── Multipart upload ─────────────────────────────────────────────────


async def _upload_one_part(
    client: httpx.AsyncClient,
    url: str,
    data: bytes,
    part_number: int,
) -> dict:
    """Upload a single part with retry + exponential backoff + jitter.

    Returns:
        {"PartNumber": int, "ETag": str}
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES_MULTIPART):
        if attempt > 0:
            delay = _backoff_delay(attempt - 1)
            await asyncio.sleep(delay)

        try:
            resp = await client.put(url, content=data)
            if resp.status_code >= 300:
                raise RuntimeError(
                    f"Part {part_number} upload failed: HTTP {resp.status_code}"
                )
            etag = resp.headers.get("etag", "")
            if not etag:
                raise RuntimeError(
                    f"Part {part_number}: S3 did not return an ETag header"
                )
            return {"PartNumber": part_number, "ETag": etag}
        except (httpx.HTTPError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Part {part_number} failed after {MAX_RETRIES_MULTIPART} retries: {last_error}"
    )


async def upload_file_multipart(
    file_path: Path,
    upload_info: UploadInfo,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Concurrent multipart upload to S3.

    Semaphore acquired BEFORE reading each chunk and released AFTER upload
    completes, keeping peak memory at MAX_CONCURRENCY * part_size.

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

    # Build part_number → presigned_url mapping
    url_map: dict[int, str] = {}
    for entry in info.presigned_urls:
        pn = entry.get("partNumber")
        url = entry.get("presignedUrl")
        if pn is None or url is None:
            raise RuntimeError(f"Malformed presigned URL entry from server: {entry!r}")
        url_map[pn] = url

    # Verify server returned URLs for every expected part
    missing = [i for i in range(1, info.total_parts + 1) if i not in url_map]
    if missing:
        raise RuntimeError(
            f"Server returned presigned URLs for {len(url_map)} parts, "
            f"but {info.total_parts} expected. Missing: {missing[:10]}"
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed_parts: list[dict] = []
    lock = asyncio.Lock()
    bytes_uploaded = 0

    async def _do_part(
        http_client: httpx.AsyncClient,
        part_number: int,
        offset: int,
        size: int,
    ) -> None:
        nonlocal bytes_uploaded
        async with semaphore:
            # Read chunk while holding semaphore to control memory.
            # Use to_thread to avoid blocking the event loop on file I/O.
            def _read_chunk():
                with open(file_path, "rb") as f:
                    f.seek(offset)
                    return f.read(size)

            data = await asyncio.to_thread(_read_chunk)

            url = url_map[part_number]
            result = await _upload_one_part(http_client, url, data, part_number)

            async with lock:
                completed_parts.append(result)
                bytes_uploaded += size
                if progress_callback:
                    progress_callback(bytes_uploaded, file_size)

    timeout = httpx.Timeout(PART_UPLOAD_TIMEOUT, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks: list[asyncio.Task] = []
        for i in range(info.total_parts):
            part_number = i + 1
            offset = i * part_size
            size = min(part_size, file_size - offset)
            tasks.append(
                asyncio.create_task(_do_part(client, part_number, offset, size))
            )

        await asyncio.gather(*tasks)

    completed_parts.sort(key=lambda p: p["PartNumber"])
    return completed_parts


# ── Progress bar helper ───────────────────────────────────────────────


def make_progress_bar(
    file_size: int,
) -> tuple[AbstractContextManager[Any] | None, ProgressCallback]:
    """Create a progress bar and a matching callback.

    Returns (progress_context, callback) where progress_context is a
    context manager wrapping a rich Progress bar (or None for plain text).

    The caller should use::

        ctx, cb = make_progress_bar(size)
        if ctx:
            with ctx:
                upload_fn(..., progress_callback=cb)
        else:
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
            DownloadColumn(),
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

    return None, _plain_cb

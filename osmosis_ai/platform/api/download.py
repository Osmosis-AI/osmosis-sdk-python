"""Dataset download helpers for presigned object URLs."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from urllib.parse import urljoin

import httpx

from .upload import _require_https, make_progress_bar

DOWNLOAD_TIMEOUT = 600.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_DOWNLOAD_REDIRECTS = 20
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def _safe_download_filename(filename: str | None) -> str | None:
    if filename is None:
        return None
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        return None
    return name


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    message = Message()
    message["content-disposition"] = value
    return _safe_download_filename(message.get_filename())


def _resolve_download_destination(
    output: Path | None,
    filename: str,
    *,
    overwrite: bool,
    output_is_directory: bool = False,
) -> Path:
    safe_filename = _safe_download_filename(filename)
    if safe_filename is None:
        raise RuntimeError("Server did not provide a usable download filename")

    if output is None:
        destination = Path.cwd() / safe_filename
    elif output.exists() and output.is_dir():
        destination = output / safe_filename
    elif output_is_directory:
        if output.exists():
            raise RuntimeError(f"Output is not a directory: {output}")
        raise RuntimeError(f"Output directory not found: {output}")
    else:
        destination = output

    if destination.exists() and destination.is_dir():
        raise RuntimeError(f"Destination is a directory: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {destination}")
    if not destination.parent.exists():
        raise RuntimeError(f"Output directory not found: {destination.parent}")

    return destination


@contextlib.contextmanager
def _stream_download_response(
    url: str,
    timeout: httpx.Timeout,
) -> Iterator[httpx.Response]:
    current_url = url
    for redirect_count in range(MAX_DOWNLOAD_REDIRECTS + 1):
        _require_https(current_url, "Download presigned URL")
        with httpx.stream(
            "GET",
            current_url,
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            if response.status_code not in _REDIRECT_STATUS_CODES:
                yield response
                return

            location = response.headers.get("location")
            if location is None:
                yield response
                return

            if redirect_count >= MAX_DOWNLOAD_REDIRECTS:
                raise RuntimeError("Too many redirects while downloading file")

            current_url = urljoin(current_url, location)
            _require_https(current_url, "Download redirect URL")
            response.read()


def download_file(
    url: str,
    *,
    output: Path | None,
    default_filename: str,
    expected_size: int,
    overwrite: bool = False,
    output_is_directory: bool = False,
) -> Path:
    """Download a presigned URL to disk and return the final path."""

    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT, connect=30.0)
    with _stream_download_response(url, timeout) as response:
        if not 200 <= response.status_code < 300:
            body = response.read().decode("utf-8", errors="replace")[:500]
            detail = f" {body}" if body else ""
            raise RuntimeError(f"HTTP {response.status_code}.{detail}")

        header_filename = _filename_from_content_disposition(
            response.headers.get("content-disposition")
        )
        destination = _resolve_download_destination(
            output,
            header_filename or default_filename,
            overwrite=overwrite,
            output_is_directory=output_is_directory,
        )

        content_length = response.headers.get("content-length")
        total_size = expected_size
        if content_length is not None:
            with contextlib.suppress(ValueError):
                total_size = int(content_length)
        progress_total = total_size if total_size > 0 else 1
        progress_ctx, progress_cb = make_progress_bar(
            progress_total,
            description="Downloading",
        )

        tmp_path: Path | None = None
        bytes_downloaded = 0
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with progress_ctx:
                    for chunk in response.iter_bytes(DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        tmp_file.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress_cb(
                            min(bytes_downloaded, progress_total), progress_total
                        )

            Path(tmp_path).replace(destination)
            progress_cb(progress_total, progress_total)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise

    return destination

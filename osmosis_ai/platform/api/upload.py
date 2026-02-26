"""S3 presigned URL upload with progress reporting."""

from __future__ import annotations

import http.client
import sys
from pathlib import Path
from urllib.parse import urlparse


def upload_file_to_presigned_url(
    file_path: Path,
    presigned_url: str,
    content_type: str,
    *,
    extra_headers: dict[str, str] | None = None,
    show_progress: bool = True,
) -> None:
    """Upload a file to an S3 presigned PUT URL.

    Uses http.client for chunked reading to avoid loading the entire file into
    memory. Falls back to plain-text progress when rich is unavailable.

    Args:
        file_path: Path to the file to upload.
        presigned_url: The presigned PUT URL from S3.
        content_type: MIME type for the Content-Type header.
        extra_headers: Additional headers required by the presigned URL.
        show_progress: Whether to show a progress bar.

    Raises:
        RuntimeError: If the upload fails (non-2xx response).
    """
    file_size = file_path.stat().st_size
    parsed = urlparse(presigned_url)

    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=30
        )
    else:
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=30
        )

    path_and_query = parsed.path
    if parsed.query:
        path_and_query += f"?{parsed.query}"

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(file_size),
    }
    if extra_headers:
        headers.update(extra_headers)

    conn.putrequest("PUT", path_and_query)
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders()

    chunk_size = 256 * 1024  # 256KB
    uploaded = 0

    progress_bar = None
    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TransferSpeedColumn,
            )
            from rich.text import Text

            class _TransferSpeedColumn(TransferSpeedColumn):
                """Show '-' instead of '?' when speed is unknown."""

                def render(self, task: object) -> Text:
                    speed = task.finished_speed or task.speed  # type: ignore[union-attr]
                    if speed is None:
                        return Text("-", style="progress.data.speed")
                    return super().render(task)  # type: ignore[arg-type]

            progress_bar = Progress(
                "[progress.percentage]{task.percentage:>3.0f}%",
                BarColumn(),
                DownloadColumn(),
                _TransferSpeedColumn(),
            )
        except ImportError:
            pass  # Fall back to plain text

    try:
        if progress_bar:
            with progress_bar:
                task = progress_bar.add_task("Uploading", total=file_size)
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        conn.send(chunk)
                        uploaded += len(chunk)
                        progress_bar.update(task, completed=uploaded)
        elif show_progress:
            # Plain text fallback
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    conn.send(chunk)
                    uploaded += len(chunk)
                    pct = uploaded * 100 // file_size
                    sys.stdout.write(f"\rUploading: {pct}%")
                    sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    conn.send(chunk)

        # Increase socket timeout for the response phase — after a large
        # upload (up to 5 GB) S3 may take longer to finalise than the 30 s
        # connection timeout allows.
        if conn.sock is not None:
            conn.sock.settimeout(120)
        response = conn.getresponse()
        if response.status >= 300:
            body = response.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Upload failed: HTTP {response.status} {response.reason}. {body}"
            )
    finally:
        conn.close()

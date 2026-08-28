"""Shared manifest-to-disk download engine for remote runs."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.metrics_export import resolve_eval_output_dir
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.download import DownloadHTTPError, download_file_to
from osmosis_ai.platform.api.models import (
    RunDownloadFile,
    RunDownloadManifest,
    RunDownloadURLBatch,
)
from osmosis_ai.platform.api.upload import make_progress_bar
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli.utils import format_size

DOWNLOAD_CONFIRM_THRESHOLD_BYTES = 100 * 1024 * 1024
DOWNLOAD_URL_BATCH_SIZE = 500
DOWNLOAD_CONCURRENCY = 8
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_RETRY_BASE_SECONDS = 0.5

EVAL_DOWNLOAD_TYPES = ("metrics", "trajectories", "artifacts", "logs")
BENCHMARK_DOWNLOAD_TYPES = ("summary", "results", "artifacts", "logs")

ManifestLoader = Callable[[Sequence[str]], RunDownloadManifest]
URLLoader = Callable[[Sequence[RunDownloadFile]], RunDownloadURLBatch]
OutputResolver = Callable[..., Path]
PathCategory = Callable[[str], str | None]

_ROWS_RE = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EVAL_RESERVED_ARTIFACT_MANIFEST_RE = re.compile(
    r"artifacts/row_\d+_run_\d+/manifest\.json"
)


@dataclass(frozen=True)
class _PreparedFile:
    manifest: RunDownloadFile
    destination: Path
    category: str


@dataclass(frozen=True)
class _TransferResult:
    file: _PreparedFile
    bytes_written: int = 0
    error: str | None = None


def _safe_error_message(error: BaseException) -> str:
    """Keep presigned URLs and their credentials out of result envelopes."""
    return _URL_RE.sub("<redacted URL>", str(error))


def parse_download_types(value: str, *, allowed: Sequence[str]) -> tuple[str, ...]:
    """Parse a comma-separated selector, expanding the standalone ``all``."""
    requested = [part.strip().lower() for part in value.split(",")]
    if not requested or any(not part for part in requested):
        raise CLIError("--type must be a comma-separated list of output types.")
    if "all" in requested:
        if len(requested) != 1:
            raise CLIError("--type all cannot be combined with other output types.")
        return tuple(allowed)

    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise CLIError(
            f"Unknown --type value(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(allowed)}, all."
        )
    return tuple(dict.fromkeys(requested))


def validate_rows(value: str | None) -> str | None:
    """Validate ``3,7,10-20`` row-selection syntax without expanding it."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or _ROWS_RE.fullmatch(normalized) is None:
        raise CLIError('--rows must use syntax like "3,7,10-20".')
    for part in normalized.split(","):
        if "-" not in part:
            continue
        start, end = (int(item) for item in part.split("-", 1))
        if start > end:
            raise CLIError(f"Invalid --rows range {part!r}: start exceeds end.")
    return normalized


def _path_category(path: str) -> str | None:
    if _EVAL_RESERVED_ARTIFACT_MANIFEST_RE.fullmatch(path) is not None:
        # The per-run artifact manifest is a server-side index, not an output.
        return None
    if path == "metrics.json":
        return "metrics"
    if path == "logs.txt":
        return "logs"
    if path == "summary.jsonl" or path.startswith("trajectories/"):
        return "trajectories"
    if path.startswith("artifacts/"):
        return "artifacts"
    return None


def benchmark_path_category(path: str) -> str | None:
    """Map a fixed benchmark download path to its public selector."""
    if path == "summary.csv":
        return "summary"
    if path == "results.csv":
        return "results"
    if path == "logs.txt":
        return "logs"
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "artifacts":
        return "artifacts"
    return None


def _safe_relative_path(
    path: str,
    *,
    selected_types: set[str],
    path_category: PathCategory = _path_category,
) -> tuple[Path, str] | None:
    if not path or path.startswith("/") or "\\" in path:
        return None
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    category = path_category(path)
    if category is None or category not in selected_types:
        return None
    return Path(*parts), category


def _prepare_files(
    manifest: RunDownloadManifest,
    *,
    output_dir: Path,
    selected_types: Sequence[str],
    path_category: PathCategory = _path_category,
) -> tuple[list[_PreparedFile], list[dict[str, str]]]:
    prepared: list[_PreparedFile] = []
    rejected: list[dict[str, str]] = []
    local_paths: set[Path] = set()
    selected = set(selected_types)
    resolved_root = output_dir.resolve(strict=False)
    for item in manifest.files:
        safe = _safe_relative_path(
            item.path,
            selected_types=selected,
            path_category=path_category,
        )
        if safe is None:
            rejected.append(
                {
                    "path": item.path,
                    "error": "Manifest path is outside the fixed run layout.",
                }
            )
            continue
        relative, category = safe
        if relative in local_paths:
            rejected.append(
                {"path": item.path, "error": "Manifest contains a duplicate path."}
            )
            continue
        destination = output_dir / relative
        if not destination.resolve(strict=False).is_relative_to(resolved_root):
            rejected.append(
                {
                    "path": item.path,
                    "error": "Manifest path crosses a local symlink outside the run root.",
                }
            )
            continue
        local_paths.add(relative)
        prepared.append(
            _PreparedFile(
                manifest=item,
                destination=destination,
                category=category,
            )
        )
    return prepared, rejected


def _breakdown(files: Sequence[_PreparedFile]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for item in files:
        bucket = result.setdefault(item.category, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += item.manifest.size
    return result


def _request_urls_with_retry(
    loader: URLLoader,
    files: Sequence[RunDownloadFile],
) -> tuple[RunDownloadURLBatch | None, str | None]:
    error: Exception | None = None
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            return loader(files), None
        except AuthenticationExpiredError:
            raise
        except PlatformAPIError as exc:
            if (
                exc.status_code is not None
                and 400 <= exc.status_code < 500
                and exc.status_code != 429
            ):
                raise
            error = exc
        except Exception as exc:  # every failed batch is reported, not fatal
            error = exc
        if attempt + 1 < DOWNLOAD_MAX_ATTEMPTS:
            time.sleep(DOWNLOAD_RETRY_BASE_SECONDS * (2**attempt))
    return (
        None,
        _safe_error_message(error)
        if error is not None
        else "Could not request download URLs.",
    )


def _find_url(
    batch: RunDownloadURLBatch,
    file: RunDownloadFile,
) -> str | None:
    for item in batch.items:
        if item.identity == file.identity:
            return item.url
    path_matches = [item.url for item in batch.items if item.path == file.path]
    return path_matches[0] if len(path_matches) == 1 else None


def _download_with_retry(
    item: _PreparedFile,
    url: str,
    *,
    url_loader: URLLoader,
) -> _TransferResult:
    current_url = url
    error: Exception | None = None
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            written = download_file_to(
                current_url,
                item.destination,
                expected_size=item.manifest.size,
            )
            return _TransferResult(file=item, bytes_written=written)
        except Exception as exc:  # per-file failures do not stop other files
            error = exc
            if attempt + 1 >= DOWNLOAD_MAX_ATTEMPTS:
                break
            if isinstance(exc, DownloadHTTPError) and exc.status_code == 403:
                refreshed, refresh_error = _request_urls_with_retry(
                    url_loader, [item.manifest]
                )
                if refreshed is None:
                    error = RuntimeError(refresh_error or "Could not refresh URL")
                    break
                refreshed_url = _find_url(refreshed, item.manifest)
                if refreshed_url is None:
                    error = RuntimeError("Refreshed response omitted this file")
                    break
                current_url = refreshed_url
            time.sleep(DOWNLOAD_RETRY_BASE_SECONDS * (2**attempt))
    return _TransferResult(
        file=item,
        error=_safe_error_message(error) if error is not None else "Download failed.",
    )


def download_manifest_file(
    *,
    manifest: RunDownloadManifest,
    path: str,
    destination: Path,
    url_loader: URLLoader,
) -> int:
    """Download one uniquely named manifest file through the shared retry path."""
    matches = [item for item in manifest.files if item.path == path]
    if len(matches) != 1:
        raise CLIError(f"Download manifest did not contain exactly one {path!r} file.")

    item = _PreparedFile(
        manifest=matches[0],
        destination=destination,
        category=_path_category(path) or "unknown",
    )
    url_batch, batch_error = _request_urls_with_retry(url_loader, [item.manifest])
    if url_batch is None:
        raise CLIError(batch_error or f"Could not request a download URL for {path!r}.")
    url = _find_url(url_batch, item.manifest)
    if url is None:
        raise CLIError(f"Download URL response omitted {path!r}.")

    result = _download_with_retry(item, url, url_loader=url_loader)
    if result.error is not None:
        raise CLIError(f"Could not download {path!r}: {result.error}")
    return result.bytes_written


def _chunks[T](items: Sequence[T], size: int) -> Sequence[Sequence[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_download(
    *,
    run_id: str,
    run_name: str | None,
    run_status: str,
    selected_types: Sequence[str],
    output: str | None,
    overwrite: bool,
    yes: bool,
    workspace_directory: Path,
    result_context: dict[str, Any],
    manifest_loader: ManifestLoader,
    url_loader: URLLoader,
    selection: dict[str, Any] | None = None,
    output_resolver: OutputResolver = resolve_eval_output_dir,
    path_category: PathCategory = _path_category,
    operation: str = "eval.download",
    resource_key: str = "eval_run",
) -> OperationResult:
    """Plan, confirm, and execute a run download using the manifest contract."""
    output_dir = output_resolver(
        run_name,
        run_id,
        workspace_directory=workspace_directory,
        output=output,
        create=False,
    )
    output_ctx = get_output_context()
    with output_ctx.status("Fetching download manifest..."):
        manifest = manifest_loader(selected_types)

    prepared, rejected = _prepare_files(
        manifest,
        output_dir=output_dir,
        selected_types=selected_types,
        path_category=path_category,
    )
    if not prepared:
        if rejected:
            raise CLIError(
                "The download manifest contained no usable run-scoped paths."
            )
        raise CLIError("No downloadable files matched the requested selection.")

    pending: list[_PreparedFile] = []
    skipped = 0
    for item in prepared:
        try:
            up_to_date = (
                not overwrite
                and item.destination.is_file()
                and item.destination.stat().st_size == item.manifest.size
            )
        except OSError:
            up_to_date = False
        if up_to_date:
            skipped += 1
        else:
            pending.append(item)

    total_bytes = sum(item.manifest.size for item in prepared)
    pending_bytes = sum(item.manifest.size for item in pending)
    breakdown = _breakdown(prepared)
    console.print(
        f"Download plan: {len(pending):,} files, {format_size(pending_bytes)} "
        f"({skipped:,} already up to date)",
        markup=False,
    )
    console.print(f"Destination: {output_dir}", markup=False, style="dim")

    if pending_bytes > DOWNLOAD_CONFIRM_THRESHOLD_BYTES:
        require_confirmation(
            "Download these run outputs?",
            yes=yes,
            default=False,
            summary=[
                ("Files", f"{len(pending):,}"),
                ("Download size", format_size(pending_bytes)),
                ("Destination", str(output_dir)),
            ],
            notes=[f"Selected types: {', '.join(selected_types)}"],
        )

    output_resolver(
        run_name,
        run_id,
        workspace_directory=workspace_directory,
        output=output,
    )

    downloaded = 0
    bytes_downloaded = 0
    failures = list(rejected)
    if pending:
        progress_total = max(pending_bytes, 1)
        progress_ctx, progress_cb = make_progress_bar(
            progress_total, description="Downloading"
        )
        processed_bytes = 0
        with progress_ctx:
            for batch in _chunks(pending, DOWNLOAD_URL_BATCH_SIZE):
                url_batch, batch_error = _request_urls_with_retry(
                    url_loader, [item.manifest for item in batch]
                )
                if url_batch is None:
                    for item in batch:
                        failures.append(
                            {
                                "path": item.manifest.path,
                                "error": batch_error
                                or "Could not request download URL.",
                            }
                        )
                        processed_bytes += item.manifest.size
                        progress_cb(
                            min(processed_bytes, progress_total), progress_total
                        )
                    continue

                ready: list[tuple[_PreparedFile, str]] = []
                for item in batch:
                    url = _find_url(url_batch, item.manifest)
                    if url is None:
                        failures.append(
                            {
                                "path": item.manifest.path,
                                "error": "Download URL response omitted this file.",
                            }
                        )
                        processed_bytes += item.manifest.size
                        progress_cb(
                            min(processed_bytes, progress_total), progress_total
                        )
                    else:
                        ready.append((item, url))

                with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as executor:
                    futures = [
                        executor.submit(
                            copy_context().run,
                            _download_with_retry,
                            item,
                            url,
                            url_loader=url_loader,
                        )
                        for item, url in ready
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        processed_bytes += result.file.manifest.size
                        progress_cb(
                            min(processed_bytes, progress_total), progress_total
                        )
                        if result.error is not None:
                            failures.append(
                                {
                                    "path": result.file.manifest.path,
                                    "error": result.error,
                                }
                            )
                        else:
                            downloaded += 1
                            bytes_downloaded += result.bytes_written
            progress_cb(progress_total, progress_total)

    partial = bool(failures)
    message = (
        f"Downloaded {downloaded:,} files ({format_size(bytes_downloaded)}) "
        f"to {output_dir}"
    )
    if skipped:
        message += f" ({skipped:,} already up to date)"
    if partial:
        message += f" ({len(failures):,} failed; re-run to retry)"

    resource: dict[str, Any] = {
        resource_key: {"id": run_id, "name": run_name},
        "status": run_status,
        "selected_types": list(selected_types),
        "files_downloaded": downloaded,
        "files_skipped": skipped,
        "files_failed": failures,
        "bytes_downloaded": bytes_downloaded,
        "total_files": len(prepared),
        "total_bytes": total_bytes,
        "breakdown": breakdown,
        "manifest_totals": manifest.totals,
        "output_path": str(output_dir),
        **result_context,
    }
    if selection:
        resource.update(selection)

    return OperationResult(
        operation=operation,
        status="partial" if partial else "success",
        resource=resource,
        message=message,
        display_next_steps=(
            [f"Failed: {item['path']} — {item['error']}" for item in failures]
            if failures
            else []
        ),
        exit_code=1 if partial else 0,
    )

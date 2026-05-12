"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    CommandResult,
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    OperationResult,
    get_output_context,
    serialize_dataset,
)
from osmosis_ai.cli.paths import parse_cli_path
from osmosis_ai.cli.prompts import confirm
from osmosis_ai.platform.api.models import STATUSES_IN_PROGRESS
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .constants import (
    MAX_FILE_SIZE,
    REQUIRED_COLUMNS,
    VALID_EXTENSIONS,
)
from .utils import (
    _require_subscription,
    build_dataset_detail_rows,
    format_dataset_status,
    format_dim_date,
    format_size,
    platform_call,
    platform_entity_url,
    require_workspace_context,
)


def _abort_upload(
    client: Any,
    dataset_id: str,
    *,
    workspace_id: str,
    credentials: Any | None = None,
) -> None:
    """Best-effort abort of an in-progress upload."""
    try:
        client.abort_upload(
            dataset_id,
            credentials=credentials,
            workspace_id=workspace_id,
        )
    except Exception as exc:
        console.print(
            f"Warning: failed to abort upload: {exc}",
            style="dim yellow",
        )


# ── Complete-upload retry (P0 reliability fix) ───────────────────────
# Inspired by huggingface_hub's http_backoff pattern: exponential
# backoff on transient errors (network/timeout/5xx).  This is the most
# dangerous failure point — the file is already on S3, so a failed
# complete call leaves an orphan that costs money and confuses users.
#
# Note: the platform has server-side safety nets for orphaned S3 objects
# (weekly cron Lambda + S3 lifecycle rules), so this retry is the
# primary client-side defense, not the only one.

from osmosis_ai.platform.api.upload import BACKOFF_BASE, BACKOFF_CAP, MAX_RETRIES

PARQUET_VALIDATION_SKIPPED_WARNING = (
    "pyarrow not installed; parquet content validation skipped. "
    "Install with: pip install 'osmosis-ai[platform]'"
)


def _workspace_result_context(workspace: Any) -> dict[str, Any]:
    return {
        "workspace": {"id": workspace.workspace_id, "name": workspace.workspace_name},
        "project_root": str(workspace.project_root),
    }


def _complete_with_retry(
    client: Any,
    dataset_id: str,
    parts: list[dict] | None = None,
    *,
    workspace_id: str,
    credentials: Any | None = None,
) -> Any:
    """Call ``client.complete_upload`` with automatic retry on transient errors.

    On final failure, prints recovery information so the user can retry
    manually with ``osmosis dataset complete <id>``.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
            console.print(
                f"Retrying complete (attempt {attempt} of {MAX_RETRIES - 1})...",
                style="dim",
            )
            time.sleep(delay)
        try:
            return platform_call(
                "Finalizing upload...",
                lambda: client.complete_upload(
                    dataset_id,
                    parts=parts,
                    credentials=credentials,
                    workspace_id=workspace_id,
                ),
                output_console=console,
            )
        except AuthenticationExpiredError:
            raise  # Not transient — surface immediately
        except PlatformAPIError as e:
            # Retry on server errors (5xx), rate limits (429), or no status code (network)
            if (
                e.status_code is not None
                and e.status_code < 500
                and e.status_code != 429
            ):
                raise  # 4xx (except 429) — not transient
            last_error = e
        except Exception as e:
            # Network / timeout / unexpected — retryable
            last_error = e

    # All retries exhausted
    console.print()
    console.print(
        f"Upload succeeded but failed to notify server after {MAX_RETRIES} attempts.",
        style="bold red",
    )
    console.print(f"  Dataset ID: {dataset_id}", style="yellow")
    console.print(
        "  The file may need to be re-uploaded. Please run the upload command again.",
        style="yellow",
    )
    console.print()
    raise CLIError(
        f"Failed to complete upload: {last_error}",
        code="PLATFORM_ERROR",
        details={"dataset_id": dataset_id},
    ) from last_error


def _check_file_basics(file: str) -> tuple[Path, str, int]:
    """Validate file existence, extension, and size.

    Returns (resolved_path, extension, file_size) or raises CLIError.
    """
    file_path = Path(file).resolve()

    try:
        file_size = file_path.stat().st_size
    except FileNotFoundError:
        raise CLIError(f"File not found: {file_path}") from None

    ext = file_path.suffix.lstrip(".").lower()
    if ext not in VALID_EXTENSIONS:
        raise CLIError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
    if file_size == 0:
        raise CLIError("File is empty.")
    if file_size > MAX_FILE_SIZE:
        raise CLIError(
            f"File too large ({format_size(file_size)}). Maximum: {format_size(MAX_FILE_SIZE)}"
        )
    return file_path, ext, file_size


def _dataset_name_from_path(file_path: Path) -> str:
    """Derive the platform dataset name from a local file path."""
    return file_path.stem


def _detail_fields(rows: list[tuple[str, str]]) -> list[DetailField]:
    """Convert existing label/value rows into renderer detail fields."""
    return [DetailField(label=label, value=value) for label, value in rows]


def _perform_upload(
    *,
    file_path: Path,
    ext: str,
    file_size: int,
    workspace_id: str,
    credentials: Any | None = None,
) -> Any:
    """Core upload: create dataset record → S3 upload → complete.

    Returns the completed DatasetFile. Raises CLIError on failure.
    Shared by the ``upload`` CLI command and the workspace interactive flow.
    """
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.upload import (
        make_progress_bar,
        upload_file_multipart,
        upload_file_simple,
    )

    client = OsmosisClient()
    dataset_name = _dataset_name_from_path(file_path)

    dataset = platform_call(
        "Creating dataset...",
        lambda: client.create_dataset(
            dataset_name,
            file_size,
            ext,
            credentials=credentials,
            workspace_id=workspace_id,
        ),
        output_console=console,
    )

    upload_info = dataset.upload
    if upload_info is None:
        raise CLIError("Server did not return upload instructions.")

    is_multipart = upload_info.method == "multipart"
    if is_multipart:
        parts_label = (
            f", {upload_info.total_parts} parts" if upload_info.total_parts else ""
        )
        console.print(
            f"Uploading {console.escape(file_path.name)} "
            f"({format_size(file_size)}, multipart{parts_label})..."
        )
    else:
        console.print(
            f"Uploading {console.escape(file_path.name)} ({format_size(file_size)})..."
        )

    ctx, progress_cb = make_progress_bar(file_size)

    if is_multipart:
        try:
            with ctx:
                parts = upload_file_multipart(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                workspace_id=workspace_id,
            )
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                workspace_id=workspace_id,
            )
            raise CLIError(f"Upload failed: {e}") from e
        completed = _complete_with_retry(
            client,
            dataset.id,
            parts=parts,
            credentials=credentials,
            workspace_id=workspace_id,
        )
    else:
        try:
            with ctx:
                upload_file_simple(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            console.print("\nUpload interrupted.")
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                workspace_id=workspace_id,
            )
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                workspace_id=workspace_id,
            )
            raise CLIError(f"Upload failed: {e}") from e
        completed = _complete_with_retry(
            client,
            dataset.id,
            credentials=credentials,
            workspace_id=workspace_id,
        )

    return completed or dataset


def upload(
    file: str,
) -> CommandResult:
    """Upload a dataset file."""
    workspace = require_workspace_context()
    ws_name = workspace.workspace_name
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    _require_subscription(workspace_id=workspace_id, workspace_name=ws_name)
    output = get_output_context()

    file_path, ext, file_size = _check_file_basics(file)

    # Validate file contents before uploading
    errors = _validate_file(file_path, ext)
    if errors:
        raise CLIError(
            "File validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Confirm upload target
    console.print()
    console.table(
        [
            ("Workspace", console.escape(ws_name) if ws_name else "unknown"),
            ("File", f"{console.escape(file_path.name)} ({format_size(file_size)})"),
        ],
        title="Upload Target",
    )
    console.print()
    if output.interactive:
        proceed = confirm("Proceed with upload?", default=True)
        if proceed is None or not proceed:
            return OperationResult(
                operation="dataset.upload",
                status="cancelled",
                message="Upload cancelled.",
            )

    dataset = _perform_upload(
        file_path=file_path,
        ext=ext,
        file_size=file_size,
        credentials=credentials,
        workspace_id=workspace_id,
    )

    url = platform_entity_url(ws_name, "datasets", dataset.id)
    resource = serialize_dataset(dataset)
    resource.update(_workspace_result_context(workspace))
    return OperationResult(
        operation="dataset.upload",
        status="success",
        resource=resource,
        message=f"Dataset uploaded: {dataset.file_name}",
        display_next_steps=[
            f"Processing will continue on the platform. Check status at: {url}"
        ],
        next_steps_structured=[
            {
                "label": "View dataset",
                "url": url,
            }
        ],
    )


def list_datasets(limit: int = DEFAULT_PAGE_SIZE, all_: bool = False) -> CommandResult:
    """List datasets."""
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    with console.spinner("Fetching datasets..."):
        if fetch_all:
            datasets, total_count = fetch_all_pages(
                lambda lim, off: client.list_datasets(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    workspace_id=workspace_id,
                ),
                items_attr="datasets",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_datasets(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                workspace_id=workspace_id,
            )
            datasets = page.datasets
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Datasets",
        items=[serialize_dataset(d) for d in datasets],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=_workspace_result_context(workspace),
        columns=[
            ListColumn(key="file_name", label="File"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="file_size", label="Size"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
        display_items=[
            {
                **serialize_dataset(d),
                "status": format_dataset_status(d),
                "file_size": format_size(d.file_size),
                "created_at": format_dim_date(d.created_at),
            }
            for d in datasets
        ],
    )


def info(
    name: str,
) -> CommandResult:
    """Show dataset details and processing status."""
    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            workspace_id=workspace_id,
        ),
        output_console=console,
    )

    rows = build_dataset_detail_rows(ds)
    data = serialize_dataset(ds)
    data.update(_workspace_result_context(workspace))
    return DetailResult(
        title="Dataset",
        data=data,
        fields=_detail_fields(rows),
    )


def preview(
    name: str,
    rows: int = 5,
) -> CommandResult:
    """Preview dataset rows."""
    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            workspace_id=workspace_id,
        ),
        output_console=console,
    )

    if ds.data_preview is None:
        if ds.status in STATUSES_IN_PROGRESS:
            message = f"Dataset is still processing (status: {ds.status})."
        else:
            message = "No preview available for this dataset."
        return DetailResult(
            title="Dataset Preview",
            data={
                "dataset": serialize_dataset(ds),
                "rows": None,
                "requested_rows": rows,
                "returned_rows": 0,
                "available": False,
                "message": message,
                **_workspace_result_context(workspace),
            },
            fields=[
                DetailField(label="Dataset", value=ds.file_name),
                DetailField(label="Status", value=ds.status),
                DetailField(label="Preview", value=message),
            ],
        )

    data_rows = ds.data_preview
    if isinstance(data_rows, list):
        limit = min(rows, len(data_rows))
        preview_rows = data_rows[:limit]
        fields = [
            DetailField(label="Dataset", value=ds.file_name),
            DetailField(label="Rows", value=f"{len(preview_rows)} of {len(data_rows)}"),
        ]
        fields.extend(
            DetailField(label=f"Row {idx}", value=str(row))
            for idx, row in enumerate(preview_rows, 1)
        )
    else:
        preview_rows = data_rows
        fields = [
            DetailField(label="Dataset", value=ds.file_name),
            DetailField(label="Preview", value=str(data_rows)),
        ]

    return DetailResult(
        title="Dataset Preview",
        data={
            "dataset": serialize_dataset(ds),
            "rows": preview_rows,
            "requested_rows": rows,
            "returned_rows": len(preview_rows) if isinstance(preview_rows, list) else 1,
            "available": True,
            **_workspace_result_context(workspace),
        },
        fields=fields,
    )


def _default_download_filename(file_name: str, presigned_url: str) -> str:
    """Choose a useful filename when the platform response omits one."""
    if Path(file_name).suffix:
        return file_name

    suffix = PurePosixPath(unquote(urlparse(presigned_url).path)).suffix
    if suffix.lstrip(".").lower() in VALID_EXTENSIONS:
        return f"{file_name}{suffix}"
    return file_name


def download(
    name: str,
    output: str | None = None,
    overwrite: bool = False,
) -> CommandResult:
    """Download a dataset file."""
    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.download import download_file

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            workspace_id=workspace_id,
        ),
        output_console=console,
    )
    if ds.status != "uploaded":
        if ds.status in STATUSES_IN_PROGRESS:
            raise CLIError(
                f"Dataset is still processing (status: {ds.status}). "
                "Try again after it finishes uploading."
            )
        raise CLIError(f"Dataset is not available for download (status: {ds.status}).")

    info = platform_call(
        "Preparing download...",
        lambda: client.get_dataset_download_url(
            name,
            credentials=credentials,
            workspace_id=workspace_id,
        ),
        output_console=console,
    )
    default_filename = _default_download_filename(
        info.file_name or ds.file_name,
        info.presigned_url,
    )
    parsed_output = parse_cli_path(output, expand_user=True) if output else None
    output_path = parsed_output.path if parsed_output else None

    try:
        destination = download_file(
            info.presigned_url,
            output=output_path,
            default_filename=default_filename,
            expected_size=ds.file_size,
            overwrite=overwrite,
            output_is_directory=(
                parsed_output.has_trailing_separator if parsed_output else False
            ),
        )
    except FileExistsError as exc:
        raise CLIError(f"{exc} Use --overwrite to replace it.") from None
    except Exception as exc:
        raise CLIError(f"Download failed: {exc}") from exc

    resource = serialize_dataset(ds)
    resource["output_path"] = str(destination)
    resource.update(_workspace_result_context(workspace))
    return OperationResult(
        operation="dataset.download",
        status="success",
        resource=resource,
        message=f"Dataset downloaded: {destination}",
    )


def validate(
    file: str,
) -> CommandResult:
    """Validate a dataset file locally."""
    file_path, ext, file_size = _check_file_basics(file)

    # Format validation
    errors, warnings = _validate_file_with_warnings(file_path, ext)

    if errors:
        raise CLIError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return DetailResult(
        title="Dataset Validation",
        data={
            "valid": True,
            "file": str(file_path),
            "file_name": file_path.name,
            "extension": ext,
            "file_size": file_size,
            "errors": [],
            "warnings": warnings,
        },
        fields=[
            DetailField(label="File", value=file_path.name),
            DetailField(label="Size", value=format_size(file_size)),
            DetailField(label="Status", value=f"Valid {ext} file"),
            *[DetailField(label="Warning", value=warning) for warning in warnings],
        ],
    )


def _validate_file_with_warnings(
    file_path: Path, ext: str
) -> tuple[list[str], list[str]]:
    """Validate file contents and surface non-fatal validation skips."""
    warnings: list[str] = []
    if ext == "parquet":
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            warnings.append(PARQUET_VALIDATION_SKIPPED_WARNING)
            return [], warnings
    return _validate_file(file_path, ext), warnings


def _validate_file(file_path: Path, ext: str) -> list[str]:
    """Validate file contents based on extension."""
    if ext == "jsonl":
        return _validate_jsonl(file_path)
    elif ext == "csv":
        return _validate_csv(file_path)
    elif ext == "parquet":
        return _validate_parquet(file_path)
    return []


def _check_required_columns(columns: Iterable[str]) -> list[str]:
    """Check that required columns are present."""
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        return [
            f"Missing required columns: {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(REQUIRED_COLUMNS))}"
        ]
    return []


def _read_tail_lines(
    file_path: Path, n: int, chunk_size: int = 1024 * 1024
) -> list[str]:
    """Read approximately the last *n* lines of a text file.

    Seeks to the end and reads a chunk backwards. If the file is smaller
    than *chunk_size* the entire content is returned (caller should handle
    overlap with head validation).
    """
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return []
        read_size = min(chunk_size, file_size)
        f.seek(file_size - read_size)
        data = f.read(read_size)

    # If we read a chunk that doesn't start at the beginning of the file,
    # it may begin mid-way through a multi-byte UTF-8 character.  Strip
    # leading continuation bytes (10xxxxxx, i.e. 0x80-0xBF) so the
    # decode doesn't fail on a perfectly valid UTF-8 file.
    if read_size < file_size:
        start = 0
        while start < len(data) and 0x80 <= data[start] <= 0xBF:
            start += 1
        data = data[start:]

    text = data.decode("utf-8")

    lines = text.split("\n")
    # If we didn't read from the start, the first "line" may be partial
    if read_size < file_size:
        lines = lines[1:]
    # Drop trailing empty string produced by a final newline
    if lines and not lines[-1]:
        lines = lines[:-1]
    return lines[-n:]


def _validate_parquet(file_path: Path) -> list[str]:
    """Validate parquet file structure and required columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        console.print(
            f"Note: {PARQUET_VALIDATION_SKIPPED_WARNING}",
            style="dim yellow",
        )
        return []

    errors = []
    try:
        with pq.ParquetFile(file_path) as pf:
            if len(pf.schema) == 0:
                errors.append("Parquet file has no columns")
            elif pf.metadata.num_rows == 0:
                errors.append("Parquet file has no rows")
            else:
                errors.extend(_check_required_columns(pf.schema_arrow.names))
    except Exception as e:
        errors.append(f"Invalid parquet file: {e}")
    return errors


def _validate_jsonl(file_path: Path) -> list[str]:
    """Validate JSONL: required columns + first/last 100 lines."""
    import json

    errors = []
    columns_checked = False
    file_fully_read = False

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 100:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if not columns_checked and isinstance(obj, dict):
                    errors.extend(_check_required_columns(obj.keys()))
                    columns_checked = True
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    return errors
        else:
            # Loop completed without break — entire file was read
            file_fully_read = True

    if file_fully_read or len(errors) >= 5:
        return errors

    # Validate last 100 lines
    try:
        tail_lines = _read_tail_lines(file_path, 100)
    except UnicodeDecodeError:
        errors.append("File contains invalid UTF-8 encoding near end of file")
        return errors
    for line in tail_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if not columns_checked and isinstance(obj, dict):
                errors.extend(_check_required_columns(obj.keys()))
                columns_checked = True
        except json.JSONDecodeError as e:
            errors.append(f"Near end of file: invalid JSON - {e}")
            if len(errors) >= 5:
                errors.append("... (showing first 5 errors)")
                break

    return errors


def _validate_csv(file_path: Path) -> list[str]:
    """Validate CSV: required columns + first/last 100 rows."""
    import csv

    errors = []
    num_cols = 0
    file_fully_read = False

    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return ["File has no header row"]

            errors.extend(_check_required_columns(header))
            num_cols = len(header)

            for i, row in enumerate(reader, 2):
                if i > 101:
                    break
                if len(row) != num_cols:
                    errors.append(
                        f"Row {i}: expected {num_cols} columns, got {len(row)}"
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        return errors
            else:
                file_fully_read = True
    except UnicodeDecodeError as e:
        errors.append(f"File encoding error: {e}")
        return errors
    except csv.Error as e:
        errors.append(f"CSV parse error: {e}")
        return errors

    if file_fully_read or len(errors) >= 5:
        return errors

    # Validate last 100 rows
    try:
        tail_lines = _read_tail_lines(file_path, 100)
    except UnicodeDecodeError:
        errors.append("File contains invalid UTF-8 encoding near end of file")
        return errors
    try:
        reader = csv.reader(tail_lines)
        for row in reader:
            if len(row) != num_cols:
                errors.append(
                    f"Near end of file: expected {num_cols} columns, got {len(row)}"
                )
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    break
    except csv.Error as e:
        errors.append(f"Near end of file: CSV parse error - {e}")

    return errors

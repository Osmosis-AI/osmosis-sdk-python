"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import confirm, is_interactive
from osmosis_ai.platform.api.models import STATUSES_IN_PROGRESS
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL

from .constants import (
    MAX_FILE_SIZE,
    REQUIRED_COLUMNS,
    VALID_EXTENSIONS,
)
from .project import (
    _require_auth,
    _require_subscription,
    _resolve_project,
    _resolve_project_id,
)
from .utils import format_dataset_status, format_processing_step, format_size

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, list, status, preview, delete, validate)."
)


def _abort_multipart(
    client: Any,
    dataset_id: str,
    upload_id: str | None,
    *,
    credentials: Any | None = None,
) -> None:
    """Best-effort abort of a multipart upload."""
    if upload_id:
        with contextlib.suppress(Exception):
            client.abort_upload(dataset_id, upload_id, credentials=credentials)


# ── Complete-upload retry (P0 reliability fix) ───────────────────────
# Inspired by huggingface_hub's http_backoff pattern: exponential
# backoff on transient errors (network/timeout/5xx).  This is the most
# dangerous failure point — the file is already on S3, so a failed
# complete call leaves an orphan that costs money and confuses users.

_COMPLETE_MAX_RETRIES = 5
_COMPLETE_BACKOFF_BASE = 1.0  # seconds
_COMPLETE_BACKOFF_CAP = 8.0  # seconds


def _complete_with_retry(
    client: Any,
    dataset_id: str,
    s3_key: str,
    extension: str | None = None,
    upload_id: str | None = None,
    parts: list[dict] | None = None,
    credentials: Any | None = None,
) -> Any:
    """Call ``client.complete_upload`` with automatic retry on transient errors.

    On final failure, prints recovery information so the user can retry
    manually with ``osmosis dataset complete <id>``.
    """
    last_error: Exception | None = None
    for attempt in range(_COMPLETE_MAX_RETRIES):
        if attempt > 0:
            delay = min(
                _COMPLETE_BACKOFF_BASE * (2 ** (attempt - 1)), _COMPLETE_BACKOFF_CAP
            )
            console.print(
                f"Retrying complete ({attempt}/{_COMPLETE_MAX_RETRIES - 1})...",
                style="dim",
            )
            time.sleep(delay)
        try:
            return client.complete_upload(
                dataset_id,
                s3_key,
                extension,
                upload_id=upload_id,
                parts=parts,
                credentials=credentials,
            )
        except AuthenticationExpiredError:
            raise  # Not transient — surface immediately
        except PlatformAPIError as e:
            # Only retry on server errors (5xx) or no status code (network)
            if e.status_code is not None and e.status_code < 500:
                raise  # 4xx — not transient
            last_error = e
        except Exception as e:
            # Network / timeout / unexpected — retryable
            last_error = e

    # All retries exhausted
    console.print()
    console.print(
        "Upload succeeded but failed to notify server after "
        f"{_COMPLETE_MAX_RETRIES} attempts.",
        style="bold red",
    )
    console.print(f"  Dataset ID: {dataset_id}", style="yellow")
    console.print("  Please try uploading the file again.", style="yellow")
    console.print()
    raise CLIError(f"Failed to complete upload: {last_error}") from last_error


@app.command("upload")
def upload(
    file: str = typer.Argument(..., help="Path to the file to upload."),
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
) -> None:
    """Upload a dataset file."""
    ws_name, credentials = _require_auth()
    _require_subscription(workspace_name=ws_name)

    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.upload import (
        make_progress_bar,
        upload_file_multipart,
        upload_file_simple,
    )

    file_path = Path(file).resolve()
    if not file_path.exists():
        raise CLIError(f"File not found: {file_path}")

    ext = file_path.suffix.lstrip(".").lower()
    if ext not in VALID_EXTENSIONS:
        raise CLIError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    file_size = file_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise CLIError(
            f"File too large ({format_size(file_size)}). Maximum: {format_size(MAX_FILE_SIZE)}"
        )
    if file_size == 0:
        raise CLIError("File is empty.")

    # Validate file contents before uploading
    errors = _validate_file(file_path, ext)
    if errors:
        raise CLIError(
            "File validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    proj = _resolve_project(project, workspace_name=ws_name)
    project_id = proj["id"]
    project_name = proj.get("project_name", "")

    # Confirm upload target
    console.print()
    console.table(
        [
            ("Workspace", ws_name or "unknown"),
            ("Project", project_name or project_id),
            ("File", f"{file_path.name} ({format_size(file_size)})"),
        ],
        title="Upload Target",
    )
    console.print()
    if is_interactive():
        proceed = confirm("Proceed with upload?", default=True)
        if proceed is None or not proceed:
            console.print("Upload cancelled.", style="dim")
            return

    client = OsmosisClient()

    # Step 1: Create dataset record + get upload instructions
    dataset = client.create_dataset(
        project_id,
        file_path.name,
        file_size,
        ext,
        credentials=credentials,
    )

    upload_info = dataset.upload
    if upload_info is None:
        raise CLIError("Server did not return upload instructions.")

    # Step 2: Upload file to S3
    is_multipart = upload_info.method == "multipart"
    if is_multipart:
        parts_label = (
            f", {upload_info.total_parts} parts" if upload_info.total_parts else ""
        )
        console.print(
            f"Uploading {file_path.name} ({format_size(file_size)}, multipart{parts_label})..."
        )
    else:
        console.print(f"Uploading {file_path.name} ({format_size(file_size)})...")

    ctx, progress_cb = make_progress_bar(file_size)

    if is_multipart:
        try:
            with ctx:
                parts = upload_file_multipart(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            _abort_multipart(
                client,
                dataset.id,
                upload_info.upload_id,
                credentials=credentials,
            )
            raise CLIError("Upload cancelled by user.") from None
        except RuntimeError as e:
            _abort_multipart(
                client,
                dataset.id,
                upload_info.upload_id,
                credentials=credentials,
            )
            raise CLIError(f"Upload failed: {e}") from e
        except Exception:
            _abort_multipart(
                client,
                dataset.id,
                upload_info.upload_id,
                credentials=credentials,
            )
            raise
        _complete_with_retry(
            client,
            dataset.id,
            upload_info.s3_key,
            ext,
            upload_id=upload_info.upload_id,
            parts=parts,
            credentials=credentials,
        )
    else:
        try:
            with ctx:
                upload_file_simple(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            console.print("\nUpload interrupted.")
            with contextlib.suppress(Exception):
                client.delete_dataset(dataset.id, credentials=credentials)
            raise CLIError("Upload cancelled by user.") from None
        except RuntimeError as e:
            raise CLIError(f"Upload failed: {e}") from e
        _complete_with_retry(
            client,
            dataset.id,
            upload_info.s3_key,
            ext,
            credentials=credentials,
        )

    console.print(f"Upload complete. Dataset ID: {dataset.id}", style="green")
    url = f"{PLATFORM_URL}/{ws_name}/{project_name}/training-data/{dataset.id}"
    console.print(f"Processing will continue on the platform. Check status at: {url}")


@app.command("list")
def list_datasets(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
) -> None:
    """List datasets."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()
    result = client.list_datasets(project_id, credentials=credentials)

    if not result.datasets:
        console.print("No datasets found.")
        return

    console.print(f"Datasets ({result.total_count}):", style="bold")
    for d in result.datasets:
        status_info = format_dataset_status(d)
        console.print(
            f"  {d.id[:8]}  {d.file_name}  {format_size(d.file_size)}  {status_info}"
        )

    if result.has_more:
        console.print(f"  ... and {result.total_count - len(result.datasets)} more")


@app.command("status")
def status(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Check dataset processing status."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    from .utils import resolve_id_prefix

    client = OsmosisClient()

    dataset_id = id
    if len(id) < 32:
        project_id = _resolve_project_id(project, workspace_name=ws_name)
        result = client.list_datasets(project_id, credentials=credentials)
        dataset_id = resolve_id_prefix(
            id, result.datasets, entity_name="dataset", has_more=result.has_more
        )

    ds = client.get_dataset(dataset_id, credentials=credentials)

    rows = [
        ("File", ds.file_name),
        ("ID", ds.id),
        ("Size", format_size(ds.file_size)),
        ("Status", ds.status),
    ]
    step = format_processing_step(ds)
    if step:
        rows.append(("Step", step))
    if ds.error:
        rows.append(("Error", ds.error))

    console.table(rows, title="Dataset Status")


@app.command("preview")
def preview(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Preview dataset rows."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    from .utils import resolve_id_prefix

    client = OsmosisClient()

    dataset_id = id
    if len(id) < 32:
        project_id = _resolve_project_id(project, workspace_name=ws_name)
        result = client.list_datasets(project_id, credentials=credentials)
        dataset_id = resolve_id_prefix(
            id, result.datasets, entity_name="dataset", has_more=result.has_more
        )

    ds = client.get_dataset(dataset_id, credentials=credentials)

    if ds.data_preview is None:
        if ds.status in STATUSES_IN_PROGRESS:
            console.print(
                f"Dataset is still processing (status: {ds.status}).",
                style="yellow",
            )
        else:
            console.print("No preview available for this dataset.", style="dim")
        return

    data_rows = ds.data_preview
    if isinstance(data_rows, list):
        limit = min(rows, len(data_rows))
        for row in data_rows[:limit]:
            console.print(str(row))
    else:
        console.print(str(data_rows))


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> None:
    """Validate a dataset file locally."""
    file_path = Path(file).resolve()
    if not file_path.exists():
        raise CLIError(f"File not found: {file_path}")

    ext = file_path.suffix.lstrip(".").lower()
    if ext not in VALID_EXTENSIONS:
        raise CLIError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise CLIError("File is empty.")
    if file_size > MAX_FILE_SIZE:
        raise CLIError(
            f"File too large ({format_size(file_size)}). Maximum: {format_size(MAX_FILE_SIZE)}"
        )

    # Format validation
    errors = _validate_file(file_path, ext)

    if errors:
        console.print("Validation errors:")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(1)

    console.print(
        f"Valid {ext} file: {file_path.name} ({format_size(file_size)})",
        style="green",
    )


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

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise

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
        return []  # pyarrow not installed, skip content validation

    errors = []
    try:
        pf = pq.ParquetFile(file_path)
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

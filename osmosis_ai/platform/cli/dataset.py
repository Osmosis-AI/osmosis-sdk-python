"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import confirm, is_interactive
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
    _require_auth,
    _require_subscription,
    build_dataset_detail_rows,
    format_dataset_status,
    format_dim_date,
    format_size,
    platform_entity_url,
)


def _abort_upload(
    client: Any,
    dataset_id: str,
    *,
    credentials: Any | None = None,
) -> None:
    """Best-effort abort of an in-progress upload."""
    try:
        client.abort_upload(dataset_id, credentials=credentials)
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


def _complete_with_retry(
    client: Any,
    dataset_id: str,
    parts: list[dict] | None = None,
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
            return client.complete_upload(
                dataset_id,
                parts=parts,
                credentials=credentials,
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
    raise CLIError(f"Failed to complete upload: {last_error}") from last_error


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


def _perform_upload(
    *,
    file_path: Path,
    ext: str,
    file_size: int,
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

    dataset = client.create_dataset(
        file_path.name,
        file_size,
        ext,
        credentials=credentials,
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
            _abort_upload(client, dataset.id, credentials=credentials)
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(client, dataset.id, credentials=credentials)
            raise CLIError(f"Upload failed: {e}") from e
        _complete_with_retry(
            client,
            dataset.id,
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
            _abort_upload(client, dataset.id, credentials=credentials)
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(client, dataset.id, credentials=credentials)
            raise CLIError(f"Upload failed: {e}") from e
        _complete_with_retry(
            client,
            dataset.id,
            credentials=credentials,
        )

    return dataset


def upload(
    file: str,
) -> None:
    """Upload a dataset file."""
    ws_name, credentials = _require_auth()
    _require_subscription(workspace_name=ws_name)

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
            ("Workspace", ws_name or "unknown"),
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

    dataset = _perform_upload(
        file_path=file_path,
        ext=ext,
        file_size=file_size,
        credentials=credentials,
    )

    console.print(f"Dataset uploaded: {console.escape(file_path.name)}", style="green")
    url = platform_entity_url(ws_name, "datasets", dataset.id)
    console.print(f"Processing will continue on the platform. Check status at: {url}")


def list_datasets(limit: int = DEFAULT_PAGE_SIZE, all_: bool = False) -> None:
    """List datasets."""
    from osmosis_ai.platform.cli.utils import (
        paginated_fetch,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    with console.spinner("Fetching datasets..."):
        datasets, total_count, _has_more = paginated_fetch(
            lambda lim, off: client.list_datasets(
                limit=lim, offset=off, credentials=credentials
            ),
            items_attr="datasets",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    if not datasets:
        console.print("No datasets found.")
        return

    console.print(f"Datasets ({total_count}):", style="bold")
    for d in datasets:
        status_info = format_dataset_status(d)
        name = console.escape(d.file_name)
        date = format_dim_date(d.created_at)
        console.print(
            f"  {name}  {format_size(d.file_size)}  {status_info}  {date}",
            highlight=False,
        )

    print_pagination_footer(len(datasets), total_count, "datasets")


def status(
    name: str,
) -> None:
    """Check dataset processing status."""
    _ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = client.get_dataset(name, credentials=credentials)

    rows = build_dataset_detail_rows(ds)
    console.table(rows, title="Dataset Status")


def preview(
    name: str,
    rows: int = 5,
) -> None:
    """Preview dataset rows."""
    _ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = client.get_dataset(name, credentials=credentials)

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


def delete(
    name: str,
    yes: bool = False,
) -> None:
    """Delete a dataset."""
    _ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    # Blocking preflight: abort if active training runs use this dataset
    try:
        affected = client.get_dataset_affected_resources(name, credentials=credentials)
    except Exception as e:
        raise CLIError(f"Unable to verify dataset dependencies: {e}") from e

    if affected.has_blocking_runs:
        lines = ["Cannot delete — active training runs depend on this dataset:"]
        for run in affected.affected_training_runs:
            run_name = (
                console.escape(run.training_run_name)
                if run.training_run_name
                else "(unnamed)"
            )
            lines.append(f"  {run_name}")
        lines.append("\nStop these training runs first, then retry.")
        raise CLIError("\n".join(lines))

    if not yes:
        ds = client.get_dataset(name, credentials=credentials)
        console.print(f"  Dataset: {ds.file_name} ({format_size(ds.file_size)})")

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(f'Delete dataset "{name}"? This cannot be undone.', yes=yes)

    client.delete_dataset(name, credentials=credentials)
    console.print(f'Dataset "{console.escape(name)}" deleted.', style="green")


def validate(
    file: str,
) -> None:
    """Validate a dataset file locally."""
    file_path, ext, file_size = _check_file_basics(file)

    # Format validation
    errors = _validate_file(file_path, ext)

    if errors:
        raise CLIError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

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
            "Note: pyarrow not installed — parquet content validation skipped. "
            "Install with: pip install 'osmosis-ai[platform]'",
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

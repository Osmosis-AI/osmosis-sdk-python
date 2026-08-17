"""Dataset resolution, content-addressed caching, and row normalization.

``experiment.dataset`` is always a platform dataset name -- never a filesystem
path -- so one TOML means the same thing to ``eval submit`` and ``eval run``
(design ``local-eval-run-plan.md`` §6). ``--dataset-file`` is the explicit local
override.

Row normalization is a **shared contract** with the cloud eval controller:
``row_index`` is the position within the *selected* set, not the original
dataset offset, because the controller selects rows first and then enumerates
them. ``source_row_index`` is retained alongside it for local ``--rows`` UX and
provenance, and never reaches the platform.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from osmosis_ai.eval.local.state import atomic_write_json
from osmosis_ai.rollout.types.sample import MessageDict

#: Dataset extensions the platform validator accepts.
VALID_EXTENSIONS: frozenset[str] = frozenset({"csv", "jsonl", "parquet"})

#: Column that switches a dataset into metadata mode (``docs/datasets.md``).
METADATA_COLUMN = "metadata"

_ROW_SELECTOR_RE = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
_HASH_CHUNK_SIZE = 1024 * 1024
_CSV_FIELD_SIZE_LIMIT = 16 * 1024 * 1024

DatasetSource = Literal["explicit", "cache", "download"]


class DatasetResolutionError(RuntimeError):
    """The dataset for this run could not be resolved or parsed."""


# --------------------------------------------------------------------------- #
# Normalized rows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalDatasetRow:
    """One normalized dataset row bound to a work item's row identity.

    ``row_index`` is the platform-visible position inside the selected set;
    ``source_row_index`` is the 0-based offset in the dataset file.
    """

    row_index: int
    source_row_index: int
    initial_messages: list[MessageDict]
    label: str | None
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class RowSelection:
    """The rows this run will execute, plus the dataset's full size."""

    rows: tuple[EvalDatasetRow, ...]
    total_dataset_rows: int

    @property
    def source_row_indices(self) -> tuple[int, ...]:
        return tuple(row.source_row_index for row in self.rows)


# --------------------------------------------------------------------------- #
# --rows selector
# --------------------------------------------------------------------------- #


def parse_row_selector(value: str) -> tuple[int, ...]:
    """Parse ``"3,7,10-20"`` into deduplicated ascending source-row indices."""
    text = value.strip()
    if not _ROW_SELECTOR_RE.fullmatch(text):
        raise DatasetResolutionError(
            f"--rows must be comma-separated indices or ranges, for example "
            f"'3,7,10-20': got {value!r}"
        )
    selected: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise DatasetResolutionError(
                    f"--rows range {part!r} is inverted; write it as {end}-{start}"
                )
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return tuple(sorted(selected))


# --------------------------------------------------------------------------- #
# Streaming readers
# --------------------------------------------------------------------------- #


def dataset_extension(path: Path) -> str:
    extension = path.suffix.lstrip(".").lower()
    if extension not in VALID_EXTENSIONS:
        raise DatasetResolutionError(
            f"{path} has extension {extension or '(none)'!r}; expected one of "
            f"{sorted(VALID_EXTENSIONS)}"
        )
    return extension


def iter_raw_rows(path: Path, extension: str) -> Iterator[Mapping[str, Any]]:
    """Stream raw rows so a large dataset never has to fit in memory."""
    if extension == "jsonl":
        yield from _iter_jsonl_rows(path)
    elif extension == "csv":
        yield from _iter_csv_rows(path)
    elif extension == "parquet":
        yield from _iter_parquet_rows(path)
    else:  # pragma: no cover - dataset_extension() gates this
        raise DatasetResolutionError(f"unsupported dataset extension {extension!r}")


def _iter_jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetResolutionError(
                    f"{path}:{lineno}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(payload, dict):
                raise DatasetResolutionError(
                    f"{path}:{lineno}: each line must be a JSON object"
                )
            yield payload


def _iter_csv_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    previous_limit = csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetResolutionError(f"{path} has no CSV header row")
            for row in reader:
                yield {key: value for key, value in row.items() if key is not None}
    finally:
        csv.field_size_limit(previous_limit)


def _iter_parquet_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise DatasetResolutionError(
            'Reading Parquet datasets requires: pip install "osmosis-ai[parquet]"'
        ) from exc
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        yield from batch.to_pylist()


# --------------------------------------------------------------------------- #
# Row normalization
# --------------------------------------------------------------------------- #


def _text_value(raw: Mapping[str, Any], column: str) -> str | None:
    value = raw.get(column)
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)


def _parse_metadata(value: Any, *, where: str) -> dict[str, Any] | None:
    """Accept a JSON object, or an encoded JSON object from a CSV/JSONL string."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetResolutionError(
                f"{where}: metadata is not valid JSON ({exc.msg})"
            ) from exc
        if not isinstance(decoded, dict):
            raise DatasetResolutionError(f"{where}: metadata must be a JSON object")
        return decoded
    raise DatasetResolutionError(f"{where}: metadata must be a JSON object")


def normalize_row(
    raw: Mapping[str, Any], *, row_index: int, source_row_index: int
) -> EvalDatasetRow:
    """Normalize one raw row into the shared prompt/metadata shape.

    Prompt mode needs ``user_prompt``; metadata mode (a ``metadata`` column)
    makes the prompt columns optional. ``system_prompt`` is always optional and
    ``label`` is an accepted alias for ``ground_truth``. Columns outside this
    contract are ignored -- they are not part of what the platform normalizes,
    so consuming them locally would fork behavior.
    """
    where = f"row {source_row_index}"
    metadata = _parse_metadata(raw.get(METADATA_COLUMN), where=where)
    metadata_mode = METADATA_COLUMN in raw

    system_prompt = _text_value(raw, "system_prompt")
    user_prompt = _text_value(raw, "user_prompt")
    if user_prompt is None and not metadata_mode:
        raise DatasetResolutionError(
            f"{where}: user_prompt is required for a dataset with no "
            f"{METADATA_COLUMN!r} column"
        )

    label = _text_value(raw, "ground_truth")
    if label is None:
        label = _text_value(raw, "label")
    if label is None and not metadata_mode:
        raise DatasetResolutionError(
            f"{where}: ground_truth (or label) is required for a dataset with "
            f"no {METADATA_COLUMN!r} column"
        )

    messages: list[MessageDict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})

    if not messages and metadata is None:
        # A CSV with a metadata header supplies every column for every row, so an
        # empty cell would otherwise waive the prompt requirement and dispatch a
        # rollout with no input at all.
        raise DatasetResolutionError(
            f"{where}: has no prompt and no metadata; a row needs at least a "
            f"user_prompt or a non-empty {METADATA_COLUMN!r} object"
        )

    return EvalDatasetRow(
        row_index=row_index,
        source_row_index=source_row_index,
        initial_messages=messages,
        label=label,
        metadata=metadata,
    )


def select_rows(
    path: Path,
    *,
    extension: str | None = None,
    limit: int | None = None,
    row_selector: Sequence[int] | None = None,
) -> RowSelection:
    """Select and normalize the rows this run executes, in one streaming pass.

    Without ``--rows``, mirror cloud selection: a positive ``evaluation.limit``
    takes the first N rows, and otherwise every row runs in dataset order.
    ``--rows`` is an explicit local override of that selection; an out-of-range
    index fails here, before the run directory is created.
    """
    resolved_extension = extension or dataset_extension(path)
    wanted = set(row_selector) if row_selector is not None else None
    rows: list[EvalDatasetRow] = []
    total = 0
    for source_row_index, raw in enumerate(iter_raw_rows(path, resolved_extension)):
        total += 1
        if wanted is not None:
            if source_row_index not in wanted:
                continue
        elif limit is not None and limit > 0 and len(rows) >= limit:
            # Keep counting so progress.json can report the dataset's real size.
            continue
        rows.append(
            normalize_row(raw, row_index=len(rows), source_row_index=source_row_index)
        )

    if wanted is not None:
        out_of_range = sorted(index for index in wanted if index >= total)
        if out_of_range:
            raise DatasetResolutionError(
                f"--rows selects {out_of_range} but {path.name} has {total} rows "
                f"(valid indices are 0-{max(total - 1, 0)})"
            )
    if not rows:
        raise DatasetResolutionError(f"{path} selected no rows")
    return RowSelection(rows=tuple(rows), total_dataset_rows=total)


# --------------------------------------------------------------------------- #
# Content-addressed cache
# --------------------------------------------------------------------------- #


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def default_dataset_cache_root() -> Path:
    return Path.home() / ".cache" / "osmosis" / "datasets"


def _workspace_cache_key(git_identity: str) -> str:
    """A readable but collision-free directory name for a workspace identity."""
    safe = (
        re.sub(r"[^A-Za-z0-9_.-]", "_", git_identity)[:48].strip("._-") or "workspace"
    )
    return f"{safe}-{hashlib.sha256(git_identity.encode()).hexdigest()[:8]}"


@dataclass(frozen=True)
class CachedDataset:
    """A verified cache entry: the bytes plus the metadata that describes them."""

    path: Path
    sha256: str
    extension: str
    dataset_id: str
    dataset_name: str
    version: str | None
    row_count: int | None
    organization_id: str | None


class DatasetCache:
    """Content-addressed dataset cache under ``<root>/<workspace>/<dataset-id>/``.

    A hit requires the recorded ``version`` to match the platform's and the
    stored bytes to hash to their recorded digest, so a truncated or swapped
    file can never masquerade as the dataset a manifest pinned.
    """

    def __init__(self, root: Path, *, git_identity: str) -> None:
        self._root = root / _workspace_cache_key(git_identity)

    def directory_for(self, dataset_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", dataset_id)
        return self._root / safe_id

    def _metadata_path(self, dataset_id: str) -> Path:
        return self.directory_for(dataset_id) / "metadata.json"

    def lookup(
        self, dataset_id: str, *, expected_version: str | None
    ) -> CachedDataset | None:
        metadata_path = self._metadata_path(dataset_id)
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        digest = payload.get("sha256")
        extension = payload.get("extension")
        if not isinstance(digest, str) or not isinstance(extension, str):
            return None
        if expected_version is not None and payload.get("version") != expected_version:
            return None
        data_path = self.directory_for(dataset_id) / f"{digest}.{extension}"
        if not data_path.is_file() or sha256_of_file(data_path) != digest:
            return None
        return CachedDataset(
            path=data_path,
            sha256=digest,
            extension=extension,
            dataset_id=dataset_id,
            dataset_name=str(payload.get("dataset_name") or dataset_id),
            version=payload.get("version"),
            row_count=payload.get("row_count"),
            organization_id=payload.get("organization_id"),
        )

    def store(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        source: Path,
        extension: str,
        version: str | None,
        row_count: int | None,
        organization_id: str | None,
    ) -> CachedDataset:
        digest = sha256_of_file(source)
        directory = self.directory_for(dataset_id)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{digest}.{extension}"
        if source != data_path:
            source.replace(data_path)
        atomic_write_json(
            self._metadata_path(dataset_id),
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "organization_id": organization_id,
                "version": version,
                "sha256": digest,
                "extension": extension,
                "row_count": row_count,
                "cached_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        return CachedDataset(
            path=data_path,
            sha256=digest,
            extension=extension,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            version=version,
            row_count=row_count,
            organization_id=organization_id,
        )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedDataset:
    """The dataset bytes this run is pinned to.

    ``sha256`` is the fingerprint the manifest stores: never the path, name, or
    mtime, so moving or renaming a file never invalidates a resumable run.
    """

    path: Path
    sha256: str
    extension: str
    source: DatasetSource
    dataset_name: str
    dataset_id: str | None = None
    version: str | None = None


def resolve_explicit_dataset_file(path: Path) -> ResolvedDataset:
    """Resolve ``--dataset-file``: the highest-precedence resolver step."""
    resolved = path.expanduser()
    if not resolved.is_file():
        raise DatasetResolutionError(f"--dataset-file {path} is not a file")
    extension = dataset_extension(resolved)
    return ResolvedDataset(
        path=resolved,
        sha256=sha256_of_file(resolved),
        extension=extension,
        source="explicit",
        dataset_name=resolved.name,
    )


@dataclass(frozen=True)
class DatasetDescription:
    """What the platform knows about a dataset, reduced to what caching needs."""

    dataset_id: str
    dataset_name: str
    extension: str
    version: str | None = None
    row_count: int | None = None
    organization_id: str | None = None
    file_size: int | None = None


class DatasetFetcher(Protocol):
    """The platform surface dataset resolution needs. Fakeable in tests."""

    def describe(self, dataset_name: str) -> DatasetDescription: ...

    def download_to(self, dataset_name: str, destination: Path) -> None: ...


def resolve_platform_dataset(
    dataset_name: str,
    *,
    cache: DatasetCache,
    fetcher: DatasetFetcher,
    on_event: Callable[[str], None] | None = None,
) -> ResolvedDataset:
    """Resolve a platform dataset name through the cache, downloading on a miss.

    The cache only ever serves bytes the platform has just confirmed the version
    of: there is no offline mode, so an unreachable platform is an error rather
    than a run pinned to whatever happened to be on disk.
    """

    def _note(message: str) -> None:
        if on_event is not None:
            on_event(message)

    try:
        description = fetcher.describe(dataset_name)
    except DatasetResolutionError:
        raise
    except Exception as exc:
        raise DatasetResolutionError(
            f"could not resolve dataset {dataset_name!r} from the platform "
            f"({exc}); pass --dataset-file to run against a local file"
        ) from exc

    cached = cache.lookup(description.dataset_id, expected_version=description.version)
    if cached is not None:
        _note(f"dataset cache hit for {dataset_name!r}")
        return _resolved_from_cache(cached)

    directory = cache.directory_for(description.dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".download.{description.extension}"
    _note(f"downloading dataset {dataset_name!r}")
    try:
        fetcher.download_to(dataset_name, staging)
        stored = cache.store(
            dataset_id=description.dataset_id,
            dataset_name=description.dataset_name,
            source=staging,
            extension=description.extension,
            version=description.version,
            row_count=description.row_count,
            organization_id=description.organization_id,
        )
    finally:
        staging.unlink(missing_ok=True)
    return _resolved_from_cache(stored, source="download")


def _resolved_from_cache(
    entry: CachedDataset, *, source: DatasetSource = "cache"
) -> ResolvedDataset:
    return ResolvedDataset(
        path=entry.path,
        sha256=entry.sha256,
        extension=entry.extension,
        source=source,
        dataset_name=entry.dataset_name,
        dataset_id=entry.dataset_id,
        version=entry.version,
    )


class PlatformDatasetFetcher:
    """:class:`DatasetFetcher` over the platform CLI API.

    The dataset endpoints accept a dataset *name* wherever they document a file
    id, so no name-to-id lookup pass is needed.
    """

    def __init__(self, *, credentials: Any, git_identity: str) -> None:
        self._credentials = credentials
        self._git_identity = git_identity

    def describe(self, dataset_name: str) -> DatasetDescription:
        from osmosis_ai.platform.api.client import OsmosisClient
        from osmosis_ai.platform.api.models import (
            STATUSES_IN_PROGRESS,
            STATUSES_SUCCESS,
        )

        record = OsmosisClient().get_dataset(
            dataset_name,
            credentials=self._credentials,
            git_identity=self._git_identity,
        )
        if record.status not in STATUSES_SUCCESS:
            hint = (
                " Try again after it finishes uploading."
                if record.status in STATUSES_IN_PROGRESS
                else ""
            )
            raise DatasetResolutionError(
                f"dataset {dataset_name!r} is not available "
                f"(status: {record.status}).{hint}"
            )
        return DatasetDescription(
            dataset_id=record.id,
            dataset_name=record.file_name or dataset_name,
            extension=_platform_extension(record),
            version=record.updated_at or None,
            row_count=record.row_count,
            organization_id=record.organization_id,
            file_size=record.file_size,
        )

    def download_to(self, dataset_name: str, destination: Path) -> None:
        from osmosis_ai.platform.api.client import OsmosisClient
        from osmosis_ai.platform.api.download import download_file_to

        info = OsmosisClient().get_dataset_download_url(
            dataset_name,
            credentials=self._credentials,
            git_identity=self._git_identity,
        )
        download_file_to(info.presigned_url, destination)


def _platform_extension(record: Any) -> str:
    for candidate in (record.file_format, record.original_file_format):
        if (
            isinstance(candidate, str)
            and candidate.lstrip(".").lower() in VALID_EXTENSIONS
        ):
            return candidate.lstrip(".").lower()
    name = record.file_name or ""
    suffix = Path(name).suffix.lstrip(".").lower()
    if suffix in VALID_EXTENSIONS:
        return suffix
    raise DatasetResolutionError(
        f"dataset {name or record.id!r} has no recognizable format; expected one "
        f"of {sorted(VALID_EXTENSIONS)}"
    )


def format_row_selector(indices: Sequence[int]) -> str:
    """Render row indices as a compact normalized selector, e.g. ``"0-9,12"``.

    Used in the manifest's resolved-input lock: a compact string keeps the lock
    readable and its diff meaningful even for a large selection, where a literal
    index list would dominate the file.
    """
    ordered = sorted(set(indices))
    if not ordered:
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)

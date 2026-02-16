"""Dataset loading and request conversion for local rollout commands.

Shared by:
- `osmosis test`
- `osmosis eval`
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict, cast

from osmosis_ai.rollout.core.schemas import RolloutRequest
from osmosis_ai.rollout.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
)

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["ground_truth", "user_prompt", "system_prompt"]
REQUIRED_COLUMNS_SET = frozenset(REQUIRED_COLUMNS)


class DatasetRow(TypedDict):
    """Type definition for a normalized dataset row."""

    ground_truth: str
    user_prompt: str
    system_prompt: str


class DatasetReader:
    """Reader for JSON / JSONL / Parquet datasets used by local workflows."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self._extension = self.file_path.suffix.lower()
        if self._extension not in (".json", ".jsonl", ".parquet"):
            raise DatasetParseError(
                f"Unsupported file format: {self._extension}. "
                f"Supported formats: .json, .jsonl, .parquet"
            )

        self._row_count: int | None = None

    def read(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DatasetRow]:
        """Read rows from the dataset file."""
        return list(self.iter_rows(limit=limit, offset=offset))

    def iter_rows(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> Iterator[DatasetRow]:
        """Yield validated rows lazily for memory-efficient iteration."""
        count = 0
        skipped = 0

        for row in self._iter_raw_rows():
            if skipped < offset:
                skipped += 1
                continue

            if limit is not None and count >= limit:
                break

            row_index = offset + count
            validated = self._validate_row(row, row_index)
            yield validated
            count += 1

    def _iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        if self._extension == ".json":
            yield from self._parse_json()
        elif self._extension == ".jsonl":
            yield from self._iter_jsonl()
        elif self._extension == ".parquet":
            yield from self._parse_parquet()

    def __len__(self) -> int:
        if self._row_count is not None:
            return self._row_count

        if self._extension == ".json":
            self._row_count = len(self._parse_json())
        elif self._extension == ".jsonl":
            self._row_count = self._count_jsonl_rows()
        elif self._extension == ".parquet":
            self._row_count = self._count_parquet_rows()
        else:
            self._row_count = 0

        return self._row_count

    def _parse_json(self) -> list[dict[str, Any]]:
        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetParseError(f"Invalid JSON file: {e}") from e
        except OSError as e:
            raise DatasetParseError(f"Error reading file: {e}") from e

        if not isinstance(data, list):
            raise DatasetParseError(
                f"JSON file must contain an array of objects, got {type(data).__name__}"
            )

        return data

    def _iter_jsonl(self) -> Iterator[dict[str, Any]]:
        try:
            with open(self.file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetParseError(
                            f"Invalid JSON at line {line_num}: {e}"
                        ) from e
                    yield row
        except OSError as e:
            raise DatasetParseError(f"Error reading file: {e}") from e

    def _count_jsonl_rows(self) -> int:
        count = 0
        try:
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count

    def _parse_parquet(self) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DatasetParseError(
                "Parquet support requires pyarrow. Install with: pip install pyarrow"
            ) from e

        try:
            table = pq.read_table(self.file_path)
            return table.to_pylist()
        except Exception as e:
            raise DatasetParseError(f"Error reading Parquet file: {e}") from e

    def _count_parquet_rows(self) -> int:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DatasetParseError(
                "Parquet support requires pyarrow. Install with: pip install pyarrow"
            ) from e

        try:
            metadata = pq.read_metadata(self.file_path)
            return metadata.num_rows
        except Exception as e:
            raise DatasetParseError(f"Error reading Parquet metadata: {e}") from e

    def _validate_row(self, row: Any, row_index: int) -> DatasetRow:
        if not isinstance(row, dict):
            raise DatasetValidationError(
                f"Row {row_index}: Expected object, got {type(row).__name__}"
            )

        lower_keys = {k.lower(): k for k in row}

        missing = [
            required for required in REQUIRED_COLUMNS if required not in lower_keys
        ]
        if missing:
            raise DatasetValidationError(
                f"Row {row_index}: Missing required columns: {missing}"
            )

        result: dict[str, Any] = {}
        for required in REQUIRED_COLUMNS:
            original_key = lower_keys[required]
            value = row[original_key]
            if value is None:
                raise DatasetValidationError(
                    f"Row {row_index}: '{required}' cannot be null"
                )
            if not isinstance(value, str):
                raise DatasetValidationError(
                    f"Row {row_index}: '{required}' must be a string, "
                    f"got {type(value).__name__}"
                )
            if not value.strip():
                raise DatasetValidationError(
                    f"Row {row_index}: '{required}' cannot be empty"
                )
            result[required] = value

        for key, value in row.items():
            if key.lower() not in REQUIRED_COLUMNS_SET:
                result[key] = value

        return cast(DatasetRow, result)


def dataset_row_to_request(
    row: DatasetRow,
    row_index: int,
    max_turns: int = 10,
    max_tokens_total: int = 4096,
    completion_params: dict[str, Any] | None = None,
    rollout_id_prefix: str = "local",
    rollout_id: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> RolloutRequest:
    """Convert a dataset row to a local RolloutRequest."""
    metadata: dict[str, Any] = {
        "ground_truth": row["ground_truth"],
        "row_index": row_index,
    }

    for key, value in row.items():
        if key.lower() not in REQUIRED_COLUMNS_SET:
            metadata[key] = value

    if metadata_overrides:
        metadata.update(metadata_overrides)

    resolved_rollout_id = rollout_id or f"{rollout_id_prefix}-{row_index}"

    return RolloutRequest(
        rollout_id=resolved_rollout_id,
        server_url="http://local-rollout.local",
        messages=[
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ],
        max_turns=max_turns,
        max_tokens_total=max_tokens_total,
        completion_params=completion_params or {},
        metadata=metadata,
    )


__all__ = [
    "REQUIRED_COLUMNS",
    "DatasetReader",
    "DatasetRow",
    "dataset_row_to_request",
]

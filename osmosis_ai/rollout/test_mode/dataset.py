"""Dataset reading and validation for test mode.

This module provides functionality to read and validate test datasets
in JSON, JSONL, and Parquet formats. It ensures datasets conform to
the expected schema with required columns.

Supported Formats:
    - .json - Array of objects [{...}, {...}]
    - .jsonl - JSON Lines format (one JSON object per line)
    - .parquet - Apache Parquet format

Required Columns (case-insensitive):
    - ground_truth: Expected output for reward calculation
    - user_prompt: User's input message
    - system_prompt: System prompt for the agent

Example:
    from osmosis_ai.rollout.test_mode.dataset import DatasetReader

    reader = DatasetReader("./test_data.jsonl")
    rows = reader.read(limit=10)

    for row in rows:
        print(row["user_prompt"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TypedDict, cast

from osmosis_ai.rollout.core.schemas import RolloutRequest
from osmosis_ai.rollout.test_mode.exceptions import (
    DatasetParseError,
    DatasetValidationError,
)

logger = logging.getLogger(__name__)

# Required columns (consistent with Osmosis Platform app)
REQUIRED_COLUMNS = ["ground_truth", "user_prompt", "system_prompt"]
# Pre-computed set for efficient membership testing (all lowercase)
REQUIRED_COLUMNS_SET = frozenset(REQUIRED_COLUMNS)


class DatasetRow(TypedDict):
    """Type definition for a dataset row.

    Attributes:
        ground_truth: Expected output for reward calculation.
        user_prompt: User's input message.
        system_prompt: System prompt for the agent.

    Note:
        Additional columns from the dataset are preserved
        but not typed here.
    """

    ground_truth: str
    user_prompt: str
    system_prompt: str


class DatasetReader:
    """Reader for test datasets in various formats.

    Supports JSON, JSONL, and Parquet formats with automatic
    format detection based on file extension.

    Validation Rules:
        - Each row must be a valid JSON object (not array or primitive)
        - All three required columns must exist (case-insensitive matching)
        - Each required field value must be a non-empty string
        - null, undefined, non-string types are rejected

    Example:
        reader = DatasetReader("./test_data.jsonl")

        # Get total row count
        print(f"Total rows: {len(reader)}")

        # Read first 10 rows
        rows = reader.read(limit=10)

        # Read with offset for pagination
        rows = reader.read(limit=10, offset=20)
    """

    def __init__(self, file_path: str) -> None:
        """Initialize the dataset reader.

        Args:
            file_path: Path to the dataset file.

        Raises:
            DatasetParseError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self._extension = self.file_path.suffix.lower()
        if self._extension not in (".json", ".jsonl", ".parquet"):
            raise DatasetParseError(
                f"Unsupported file format: {self._extension}. "
                f"Supported formats: .json, .jsonl, .parquet"
            )

        # Cache for row count (computed lazily)
        self._row_count: Optional[int] = None

    def read(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[DatasetRow]:
        """Read rows from the dataset file.

        Args:
            limit: Maximum number of rows to read. None means all rows.
            offset: Number of rows to skip before reading (0-indexed).

        Returns:
            List of DatasetRow dicts with normalized column names.

        Raises:
            DatasetValidationError: If required columns are missing or values invalid.
            DatasetParseError: If file parsing fails.
        """
        return list(self.iter_rows(limit=limit, offset=offset))

    def iter_rows(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[DatasetRow]:
        """Iterate over rows from the dataset file.

        Memory-efficient alternative to read() for large datasets.

        Args:
            limit: Maximum number of rows to yield. None means all rows.
            offset: Number of rows to skip before yielding.

        Yields:
            DatasetRow dicts with normalized column names.

        Raises:
            DatasetValidationError: If required columns are missing or values invalid.
            DatasetParseError: If file parsing fails.
        """
        count = 0
        skipped = 0

        for i, row in enumerate(self._iter_raw_rows()):
            if skipped < offset:
                skipped += 1
                continue

            if limit is not None and count >= limit:
                break

            # Use offset + count for consistent row indexing with read()
            row_index = offset + count
            validated = self._validate_row(row, row_index)
            yield validated
            count += 1

    def _iter_raw_rows(self) -> Iterator[Dict[str, Any]]:
        """Iterate over raw rows without validation."""
        if self._extension == ".json":
            yield from self._parse_json()
        elif self._extension == ".jsonl":
            yield from self._iter_jsonl()
        elif self._extension == ".parquet":
            yield from self._parse_parquet()

    def __len__(self) -> int:
        """Return total row count.

        Note:
            For JSON files, this requires parsing the entire file.
            For JSONL, this counts lines efficiently.
            For Parquet, this uses metadata when available.
        """
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

    def _parse_json(self) -> List[Dict[str, Any]]:
        """Parse a JSON file containing an array of objects."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetParseError(f"Invalid JSON file: {e}")
        except OSError as e:
            raise DatasetParseError(f"Error reading file: {e}")

        if not isinstance(data, list):
            raise DatasetParseError(
                f"JSON file must contain an array of objects, got {type(data).__name__}"
            )

        return data

    def _parse_jsonl(self) -> List[Dict[str, Any]]:
        """Parse a JSONL file (one JSON object per line)."""
        return list(self._iter_jsonl())

    def _iter_jsonl(self) -> Iterator[Dict[str, Any]]:
        """Iterate over JSONL file line by line."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetParseError(
                            f"Invalid JSON at line {line_num}: {e}"
                        )

                    yield row
        except OSError as e:
            raise DatasetParseError(f"Error reading file: {e}")

    def _count_jsonl_rows(self) -> int:
        """Count rows in JSONL file efficiently."""
        count = 0
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count

    def _parse_parquet(self) -> List[Dict[str, Any]]:
        """Parse a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise DatasetParseError(
                "Parquet support requires pyarrow. "
                "Install with: pip install 'osmosis-ai[test-mode]'"
            )

        try:
            table = pq.read_table(self.file_path)
            return table.to_pylist()
        except Exception as e:
            raise DatasetParseError(f"Error reading Parquet file: {e}")

    def _count_parquet_rows(self) -> int:
        """Count rows in Parquet file using metadata."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise DatasetParseError(
                "Parquet support requires pyarrow. "
                "Install with: pip install 'osmosis-ai[test-mode]'"
            )

        try:
            metadata = pq.read_metadata(self.file_path)
            return metadata.num_rows
        except Exception as e:
            raise DatasetParseError(f"Error reading Parquet metadata: {e}") from e

    def _validate_row(self, row: Any, row_index: int) -> DatasetRow:
        """Validate a single row and normalize column names.

        Args:
            row: Raw row data from file.
            row_index: Index of the row for error messages.

        Returns:
            Validated DatasetRow with normalized column names.

        Raises:
            DatasetValidationError: If validation fails.
        """
        if not isinstance(row, dict):
            raise DatasetValidationError(
                f"Row {row_index}: Expected object, got {type(row).__name__}"
            )

        # Build a lowercase -> original key mapping
        lower_keys = {k.lower(): k for k in row.keys()}

        # Check required columns exist (case-insensitive)
        missing = []
        for required in REQUIRED_COLUMNS:
            if required.lower() not in lower_keys:
                missing.append(required)

        if missing:
            raise DatasetValidationError(
                f"Row {row_index}: Missing required columns: {missing}"
            )

        # Extract and validate required fields
        result: Dict[str, Any] = {}

        for required in REQUIRED_COLUMNS:
            original_key = lower_keys[required.lower()]
            value = row[original_key]

            # Validate value is a non-empty string
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

            # Store with normalized (lowercase) key
            result[required] = value

        # Preserve additional columns with original casing
        for key, value in row.items():
            if key.lower() not in REQUIRED_COLUMNS_SET:
                result[key] = value

        # Use cast instead of TypedDict constructor to preserve extra columns.
        # TypedDict is just a dict at runtime, so extra keys are allowed.
        return cast(DatasetRow, result)


def dataset_row_to_request(
    row: DatasetRow,
    row_index: int,
    max_turns: int = 10,
    max_tokens_total: int = 4096,
    completion_params: Optional[Dict[str, Any]] = None,
) -> RolloutRequest:
    """Convert a dataset row to a RolloutRequest.

    Args:
        row: Dataset row with user_prompt, system_prompt, ground_truth.
        row_index: Index of this row in the dataset (for rollout_id).
        max_turns: Advisory max LLM calls for agent loop.
        max_tokens_total: Advisory max total tokens (NOT enforced in test mode).
        completion_params: Sampling parameters (temperature, etc.).

    Returns:
        RolloutRequest ready to be passed to agent_loop.get_tools() and run().

    Note:
        server_url is a placeholder since ExternalLLMClient handles LLM calls
        directly without going through TrainGate.

        ground_truth is stored in metadata for reward function access.
        Users access it via: ctx.request.metadata["ground_truth"]
    """
    # Build metadata with ground_truth and test mode marker
    metadata: Dict[str, Any] = {
        "ground_truth": row["ground_truth"],
        "test_mode": True,
        "row_index": row_index,
    }

    # Preserve any extra columns from dataset
    for key, value in row.items():
        if key not in REQUIRED_COLUMNS_SET:
            metadata[key] = value

    return RolloutRequest(
        rollout_id=f"test-{row_index}",
        server_url="http://test-mode.local",  # Placeholder, not used in test mode
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

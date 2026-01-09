"""Tests for test_mode dataset reading and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from osmosis_ai.rollout.test_mode.dataset import (
    REQUIRED_COLUMNS,
    DatasetReader,
    DatasetRow,
    dataset_row_to_request,
)
from osmosis_ai.rollout.test_mode.exceptions import (
    DatasetParseError,
    DatasetValidationError,
)

# Check if pyarrow is available for Parquet tests
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class TestDatasetReader:
    """Tests for DatasetReader class."""

    def test_read_json_file(self, tmp_path: Path) -> None:
        """Test reading a valid JSON file."""
        data = [
            {
                "user_prompt": "What is 2+2?",
                "system_prompt": "You are a calculator.",
                "ground_truth": "4",
            },
            {
                "user_prompt": "What is 3+3?",
                "system_prompt": "You are a calculator.",
                "ground_truth": "6",
            },
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 2
        assert rows[0]["user_prompt"] == "What is 2+2?"
        assert rows[1]["ground_truth"] == "6"

    def test_read_jsonl_file(self, tmp_path: Path) -> None:
        """Test reading a valid JSONL file."""
        lines = [
            '{"user_prompt": "Hello", "system_prompt": "Be helpful", "ground_truth": "Hi"}',
            '{"user_prompt": "Bye", "system_prompt": "Be helpful", "ground_truth": "Goodbye"}',
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 2
        assert rows[0]["user_prompt"] == "Hello"
        assert rows[1]["ground_truth"] == "Goodbye"

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_read_parquet_file(self, tmp_path: Path) -> None:
        """Test reading a valid Parquet file."""
        # Create test data using pyarrow
        table = pa.table(
            {
                "user_prompt": ["What is 2+2?", "What is 3+3?"],
                "system_prompt": ["You are a calculator.", "You are a calculator."],
                "ground_truth": ["4", "6"],
            }
        )
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 2
        assert rows[0]["user_prompt"] == "What is 2+2?"
        assert rows[1]["ground_truth"] == "6"

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_read_parquet_with_extra_columns(self, tmp_path: Path) -> None:
        """Test that extra columns are preserved in Parquet files."""
        table = pa.table(
            {
                "user_prompt": ["Question"],
                "system_prompt": ["System"],
                "ground_truth": ["Answer"],
                "difficulty": ["easy"],
                "category": ["math"],
            }
        )
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        assert rows[0]["difficulty"] == "easy"
        assert rows[0]["category"] == "math"

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_parquet_len_uses_metadata(self, tmp_path: Path) -> None:
        """Test that __len__ for Parquet uses metadata efficiently."""
        table = pa.table(
            {
                "user_prompt": [f"Q{i}" for i in range(100)],
                "system_prompt": ["sys"] * 100,
                "ground_truth": [f"A{i}" for i in range(100)],
            }
        )
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        reader = DatasetReader(str(file_path))
        # This should use metadata, not parse entire file
        assert len(reader) == 100

    def test_read_with_limit(self, tmp_path: Path) -> None:
        """Test reading with limit parameter."""
        data = [
            {"user_prompt": f"Q{i}", "system_prompt": "sys", "ground_truth": f"A{i}"}
            for i in range(10)
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read(limit=3)

        assert len(rows) == 3
        assert rows[0]["user_prompt"] == "Q0"
        assert rows[2]["user_prompt"] == "Q2"

    def test_read_with_offset(self, tmp_path: Path) -> None:
        """Test reading with offset parameter."""
        data = [
            {"user_prompt": f"Q{i}", "system_prompt": "sys", "ground_truth": f"A{i}"}
            for i in range(10)
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read(offset=5)

        assert len(rows) == 5
        assert rows[0]["user_prompt"] == "Q5"
        assert rows[4]["user_prompt"] == "Q9"

    def test_read_with_limit_and_offset(self, tmp_path: Path) -> None:
        """Test reading with both limit and offset."""
        data = [
            {"user_prompt": f"Q{i}", "system_prompt": "sys", "ground_truth": f"A{i}"}
            for i in range(10)
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read(limit=3, offset=2)

        assert len(rows) == 3
        assert rows[0]["user_prompt"] == "Q2"
        assert rows[2]["user_prompt"] == "Q4"

    def test_case_insensitive_column_names(self, tmp_path: Path) -> None:
        """Test that column names are matched case-insensitively."""
        data = [
            {
                "USER_PROMPT": "Question",
                "System_Prompt": "System",
                "GROUND_TRUTH": "Answer",
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        # Should normalize to lowercase keys
        assert rows[0]["user_prompt"] == "Question"
        assert rows[0]["system_prompt"] == "System"
        assert rows[0]["ground_truth"] == "Answer"

    def test_preserve_extra_columns(self, tmp_path: Path) -> None:
        """Test that extra columns are preserved in the result."""
        data = [
            {
                "user_prompt": "Question",
                "system_prompt": "System",
                "ground_truth": "Answer",
                "difficulty": "easy",
                "category": "math",
                "custom_field": 123,
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        # Extra columns should be preserved with original casing
        assert rows[0]["difficulty"] == "easy"
        assert rows[0]["category"] == "math"
        assert rows[0]["custom_field"] == 123

    def test_len_returns_row_count(self, tmp_path: Path) -> None:
        """Test that __len__ returns correct row count."""
        data = [
            {"user_prompt": f"Q{i}", "system_prompt": "sys", "ground_truth": f"A{i}"}
            for i in range(5)
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        assert len(reader) == 5

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            DatasetReader("/nonexistent/path/data.json")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Test that DatasetParseError is raised for unsupported format."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("a,b,c\n1,2,3")

        with pytest.raises(DatasetParseError) as exc_info:
            DatasetReader(str(file_path))
        assert "Unsupported file format" in str(exc_info.value)

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test that DatasetParseError is raised for invalid JSON."""
        file_path = tmp_path / "test.json"
        file_path.write_text("not valid json")

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetParseError) as exc_info:
            reader.read()
        assert "Invalid JSON" in str(exc_info.value)

    def test_json_not_array(self, tmp_path: Path) -> None:
        """Test that DatasetParseError is raised when JSON is not an array."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"key": "value"}')

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetParseError) as exc_info:
            reader.read()
        assert "array of objects" in str(exc_info.value)

    def test_missing_required_column(self, tmp_path: Path) -> None:
        """Test that DatasetValidationError is raised for missing column."""
        data = [
            {
                "user_prompt": "Question",
                "system_prompt": "System",
                # missing ground_truth
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "Missing required columns" in str(exc_info.value)
        assert "ground_truth" in str(exc_info.value)

    def test_null_value_rejected(self, tmp_path: Path) -> None:
        """Test that null values are rejected."""
        data = [
            {
                "user_prompt": None,
                "system_prompt": "System",
                "ground_truth": "Answer",
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "cannot be null" in str(exc_info.value)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        """Test that empty strings are rejected."""
        data = [
            {
                "user_prompt": "  ",  # whitespace only
                "system_prompt": "System",
                "ground_truth": "Answer",
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "cannot be empty" in str(exc_info.value)

    def test_non_string_value_rejected(self, tmp_path: Path) -> None:
        """Test that non-string values are rejected for required columns."""
        data = [
            {
                "user_prompt": 123,  # number instead of string
                "system_prompt": "System",
                "ground_truth": "Answer",
            }
        ]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "must be a string" in str(exc_info.value)

    def test_row_not_object_rejected(self, tmp_path: Path) -> None:
        """Test that non-object rows are rejected."""
        data = ["just a string", "another string"]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "Expected object" in str(exc_info.value)


class TestDatasetRowToRequest:
    """Tests for dataset_row_to_request function."""

    def test_basic_conversion(self) -> None:
        """Test basic conversion from DatasetRow to RolloutRequest."""
        row: Dict[str, Any] = {
            "user_prompt": "What is 2+2?",
            "system_prompt": "You are a calculator.",
            "ground_truth": "4",
        }

        request = dataset_row_to_request(row, row_index=0)  # type: ignore[arg-type]

        assert request.rollout_id == "test-0"
        assert request.server_url == "http://test-mode.local"
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"] == "You are a calculator."
        assert request.messages[1]["role"] == "user"
        assert request.messages[1]["content"] == "What is 2+2?"

    def test_ground_truth_in_metadata(self) -> None:
        """Test that ground_truth is stored in metadata."""
        row: Dict[str, Any] = {
            "user_prompt": "Question",
            "system_prompt": "System",
            "ground_truth": "Expected Answer",
        }

        request = dataset_row_to_request(row, row_index=5)  # type: ignore[arg-type]

        assert request.metadata["ground_truth"] == "Expected Answer"
        assert request.metadata["test_mode"] is True
        assert request.metadata["row_index"] == 5

    def test_extra_columns_in_metadata(self) -> None:
        """Test that extra columns are preserved in metadata."""
        row: Dict[str, Any] = {
            "user_prompt": "Question",
            "system_prompt": "System",
            "ground_truth": "Answer",
            "difficulty": "hard",
            "category": "science",
        }

        request = dataset_row_to_request(row, row_index=0)  # type: ignore[arg-type]

        assert request.metadata["difficulty"] == "hard"
        assert request.metadata["category"] == "science"

    def test_max_turns_parameter(self) -> None:
        """Test that max_turns is passed correctly."""
        row: Dict[str, Any] = {
            "user_prompt": "Question",
            "system_prompt": "System",
            "ground_truth": "Answer",
        }

        request = dataset_row_to_request(row, row_index=0, max_turns=20)  # type: ignore[arg-type]

        assert request.max_turns == 20

    def test_completion_params(self) -> None:
        """Test that completion_params are passed correctly."""
        row: Dict[str, Any] = {
            "user_prompt": "Question",
            "system_prompt": "System",
            "ground_truth": "Answer",
        }

        request = dataset_row_to_request(
            row,  # type: ignore[arg-type]
            row_index=0,
            completion_params={"temperature": 0.5, "max_tokens": 1000},
        )

        assert request.completion_params["temperature"] == 0.5
        assert request.completion_params["max_tokens"] == 1000

"""Tests for eval dataset reading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.eval.common.dataset import (
    DatasetReader,
    DatasetRow,
    dataset_row_to_prompt,
)
from osmosis_ai.eval.common.errors import (
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

    def test_read_csv_file(self, tmp_path: Path) -> None:
        """Test reading a valid CSV file."""
        content = (
            "user_prompt,system_prompt,ground_truth\n"
            "What is 2+2?,You are a calculator.,4\n"
            "What is 3+3?,You are a calculator.,6\n"
        )
        file_path = tmp_path / "test.csv"
        file_path.write_text(content)

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 2
        assert rows[0]["user_prompt"] == "What is 2+2?"
        assert rows[1]["ground_truth"] == "6"

    def test_csv_with_extra_columns(self, tmp_path: Path) -> None:
        """Test that extra columns are preserved in CSV files."""
        content = (
            "user_prompt,system_prompt,ground_truth,difficulty,category\n"
            "Question,System,Answer,easy,math\n"
        )
        file_path = tmp_path / "test.csv"
        file_path.write_text(content)

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        assert rows[0]["difficulty"] == "easy"
        assert rows[0]["category"] == "math"

    def test_csv_with_quoted_fields(self, tmp_path: Path) -> None:
        """Test that CSV fields containing commas and quotes are parsed correctly."""
        content = (
            "system_prompt,user_prompt,ground_truth\n"
            '"You are a helpful, friendly assistant.",What is 2 + 2?,4\n'
            '"Say ""hello"".",Greet me,hello\n'
        )
        file_path = tmp_path / "test.csv"
        file_path.write_text(content)

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 2
        assert rows[0]["system_prompt"] == "You are a helpful, friendly assistant."
        assert rows[1]["system_prompt"] == 'Say "hello".'

    def test_csv_missing_required_column(self, tmp_path: Path) -> None:
        """Test that DatasetValidationError is raised for CSV with missing column."""
        content = "user_prompt,system_prompt\nQuestion,System\n"
        file_path = tmp_path / "test.csv"
        file_path.write_text(content)

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "Missing required columns" in str(exc_info.value)
        assert "ground_truth" in str(exc_info.value)

    def test_csv_len(self, tmp_path: Path) -> None:
        """Test that __len__ returns correct row count for CSV files."""
        lines = ["user_prompt,system_prompt,ground_truth"]
        for i in range(5):
            lines.append(f"Q{i},sys,A{i}")
        file_path = tmp_path / "test.csv"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        assert len(reader) == 5

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
        lines = [
            json.dumps(
                {
                    "user_prompt": f"Q{i}",
                    "system_prompt": "sys",
                    "ground_truth": f"A{i}",
                }
            )
            for i in range(10)
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read(limit=3)

        assert len(rows) == 3
        assert rows[0]["user_prompt"] == "Q0"
        assert rows[2]["user_prompt"] == "Q2"

    def test_read_with_offset(self, tmp_path: Path) -> None:
        """Test reading with offset parameter."""
        lines = [
            json.dumps(
                {
                    "user_prompt": f"Q{i}",
                    "system_prompt": "sys",
                    "ground_truth": f"A{i}",
                }
            )
            for i in range(10)
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read(offset=5)

        assert len(rows) == 5
        assert rows[0]["user_prompt"] == "Q5"
        assert rows[4]["user_prompt"] == "Q9"

    def test_read_with_limit_and_offset(self, tmp_path: Path) -> None:
        """Test reading with both limit and offset."""
        lines = [
            json.dumps(
                {
                    "user_prompt": f"Q{i}",
                    "system_prompt": "sys",
                    "ground_truth": f"A{i}",
                }
            )
            for i in range(10)
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read(limit=3, offset=2)

        assert len(rows) == 3
        assert rows[0]["user_prompt"] == "Q2"
        assert rows[2]["user_prompt"] == "Q4"

    def test_case_insensitive_column_names(self, tmp_path: Path) -> None:
        """Test that column names are matched case-insensitively."""
        lines = [
            '{"USER_PROMPT": "Question", "System_Prompt": "System", "GROUND_TRUTH": "Answer"}'
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        # Should normalize to lowercase keys
        assert rows[0]["user_prompt"] == "Question"
        assert rows[0]["system_prompt"] == "System"
        assert rows[0]["ground_truth"] == "Answer"

    def test_preserve_extra_columns(self, tmp_path: Path) -> None:
        """Test that extra columns are preserved in the result."""
        lines = [
            json.dumps(
                {
                    "user_prompt": "Question",
                    "system_prompt": "System",
                    "ground_truth": "Answer",
                    "difficulty": "easy",
                    "category": "math",
                    "custom_field": 123,
                }
            )
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        rows = reader.read()

        assert len(rows) == 1
        # Extra columns should be preserved with original casing
        assert rows[0]["difficulty"] == "easy"
        assert rows[0]["category"] == "math"
        assert rows[0]["custom_field"] == 123

    def test_len_returns_row_count(self, tmp_path: Path) -> None:
        """Test that __len__ returns correct row count."""
        lines = [
            json.dumps(
                {
                    "user_prompt": f"Q{i}",
                    "system_prompt": "sys",
                    "ground_truth": f"A{i}",
                }
            )
            for i in range(5)
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        assert len(reader) == 5

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            DatasetReader("/nonexistent/path/data.jsonl")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Test that DatasetParseError is raised for unsupported format."""
        file_path = tmp_path / "test.json"
        file_path.write_text('[{"a": 1}]')

        with pytest.raises(DatasetParseError) as exc_info:
            DatasetReader(str(file_path))
        assert "Unsupported file format" in str(exc_info.value)

    def test_invalid_jsonl(self, tmp_path: Path) -> None:
        """Test that DatasetParseError is raised for invalid JSONL."""
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("not valid json")

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetParseError) as exc_info:
            reader.read()
        assert "Invalid JSON" in str(exc_info.value)

    def test_missing_required_column(self, tmp_path: Path) -> None:
        """Test that DatasetValidationError is raised for missing column."""
        lines = [
            json.dumps(
                {
                    "user_prompt": "Question",
                    "system_prompt": "System",
                    # missing ground_truth
                }
            )
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "Missing required columns" in str(exc_info.value)
        assert "ground_truth" in str(exc_info.value)

    def test_null_value_rejected(self, tmp_path: Path) -> None:
        """Test that null values are rejected."""
        lines = [
            json.dumps(
                {
                    "user_prompt": None,
                    "system_prompt": "System",
                    "ground_truth": "Answer",
                }
            )
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "cannot be null" in str(exc_info.value)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        """Test that empty strings are rejected."""
        lines = [
            json.dumps(
                {
                    "user_prompt": "  ",  # whitespace only
                    "system_prompt": "System",
                    "ground_truth": "Answer",
                }
            )
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "cannot be empty" in str(exc_info.value)

    def test_non_string_value_rejected(self, tmp_path: Path) -> None:
        """Test that non-string values are rejected for required columns."""
        lines = [
            json.dumps(
                {
                    "user_prompt": 123,  # number instead of string
                    "system_prompt": "System",
                    "ground_truth": "Answer",
                }
            )
        ]
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "must be a string" in str(exc_info.value)

    def test_row_not_object_rejected(self, tmp_path: Path) -> None:
        """Test that non-object rows are rejected."""
        lines = ['"just a string"', '"another string"']
        file_path = tmp_path / "test.jsonl"
        file_path.write_text("\n".join(lines))

        reader = DatasetReader(str(file_path))
        with pytest.raises(DatasetValidationError) as exc_info:
            reader.read()
        assert "Expected object" in str(exc_info.value)


def test_dataset_row_to_prompt() -> None:
    """Test conversion from DatasetRow to a plain prompt list."""
    row: DatasetRow = {
        "system_prompt": "You are a math tutor.",
        "user_prompt": "What is 2+2?",
        "ground_truth": "4",
    }
    prompt = dataset_row_to_prompt(row)
    assert prompt == [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
    ]

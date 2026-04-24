"""Tests for osmosis_ai.eval.rubric — dataset loader, report, and RubricCommand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.rubric.cli import RubricCommand
from osmosis_ai.eval.rubric.dataset import (
    RubricRecord,
    load_rubric_dataset,
)
from osmosis_ai.eval.rubric.report import (
    ConsoleReportRenderer,
    JsonReportWriter,
    RecordResult,
    RubricReport,
    calculate_statistics,
)
from osmosis_ai.eval.rubric.types import RubricResult

# =============================================================================
# load_rubric_dataset Tests
# =============================================================================


class TestLoadRubricDataset:
    """Tests for the JSONL dataset loader."""

    def test_messages_format_loads_correctly(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)

        assert len(records) == 1
        assert records[0].solution_str == "Hi there!"
        assert records[0].ground_truth is None
        assert records[0].original_input is None
        assert records[0].metadata is None
        assert records[0].record_id is None

    def test_solution_str_format_auto_converts(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {"solution_str": "The answer is 42."}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)

        assert len(records) == 1
        assert records[0].solution_str == "The answer is 42."

    def test_missing_messages_and_solution_str_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {"some_other_field": "value"}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(
            CLIError, match=r"must include 'messages'.*or 'solution_str'"
        ):
            load_rubric_dataset(data_file)

    def test_invalid_json_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("not valid json\n", encoding="utf-8")

        with pytest.raises(CLIError, match="Invalid JSON on line 1"):
            load_rubric_dataset(data_file)

    def test_empty_file_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("", encoding="utf-8")

        with pytest.raises(CLIError, match="No JSON records found"):
            load_rubric_dataset(data_file)

    def test_blank_lines_only_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("\n\n\n", encoding="utf-8")

        with pytest.raises(CLIError, match="No JSON records found"):
            load_rubric_dataset(data_file)

    def test_optional_fields_parsed(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Answer"}],
            "ground_truth": "Expected answer",
            "original_input": "What is the question?",
            "metadata": {"key": "value"},
            "id": "abc-123",
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)

        assert records[0].ground_truth == "Expected answer"
        assert records[0].original_input == "What is the question?"
        assert records[0].metadata == {"key": "value"}
        assert records[0].record_id == "abc-123"

    def test_conversation_id_used_as_record_id(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Answer"}],
            "conversation_id": "conv-456",
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)
        assert records[0].record_id == "conv-456"

    def test_id_takes_precedence_over_conversation_id(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Answer"}],
            "id": "primary-id",
            "conversation_id": "secondary-id",
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)
        assert records[0].record_id == "primary-id"

    def test_multiple_records(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"messages": [{"role": "assistant", "content": f"Answer {i}"}]})
            for i in range(3)
        ]
        data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)
        assert len(records) == 3

    def test_non_dict_json_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("[1, 2, 3]\n", encoding="utf-8")

        with pytest.raises(CLIError, match="Expected JSON object"):
            load_rubric_dataset(data_file)

    def test_empty_messages_list_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {"messages": []}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(
            CLIError, match=r"must include 'messages'.*or 'solution_str'"
        ):
            load_rubric_dataset(data_file)

    def test_whitespace_only_solution_str_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {"solution_str": "   "}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(
            CLIError, match=r"must include 'messages'.*or 'solution_str'"
        ):
            load_rubric_dataset(data_file)

    def test_blank_lines_skipped(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        content = (
            json.dumps({"messages": [{"role": "assistant", "content": "A"}]})
            + "\n"
            + "\n"
            + json.dumps({"messages": [{"role": "assistant", "content": "B"}]})
            + "\n"
        )
        data_file.write_text(content, encoding="utf-8")

        records = load_rubric_dataset(data_file)
        assert len(records) == 2


# =============================================================================
# RubricRecord.label Tests
# =============================================================================


class TestRubricRecordLabel:
    """Tests for RubricRecord.label() method."""

    def test_with_record_id_returns_record_id(self):
        record = RubricRecord(
            solution_str="test",
            ground_truth=None,
            original_input=None,
            metadata=None,
            record_id="my-record-id",
        )
        assert record.label(5) == "my-record-id"

    def test_without_record_id_returns_indexed_label(self):
        record = RubricRecord(
            solution_str="test",
            ground_truth=None,
            original_input=None,
            metadata=None,
            record_id=None,
        )
        assert record.label(3) == "record[3]"

    def test_without_record_id_index_zero(self):
        record = RubricRecord(
            solution_str="test",
            ground_truth=None,
            original_input=None,
            metadata=None,
            record_id=None,
        )
        assert record.label(0) == "record[0]"


# =============================================================================
# calculate_statistics Tests
# =============================================================================


class TestCalculateStatistics:
    """Tests for the calculate_statistics helper."""

    def test_empty_scores(self):
        stats = calculate_statistics([])
        assert stats == {
            "average": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    def test_single_score(self):
        stats = calculate_statistics([0.75])
        assert stats["average"] == 0.75
        assert stats["variance"] == 0.0
        assert stats["stdev"] == 0.0
        assert stats["min"] == 0.75
        assert stats["max"] == 0.75

    def test_multiple_scores(self):
        stats = calculate_statistics([0.0, 1.0])
        assert stats["average"] == 0.5
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0
        assert stats["stdev"] == 0.5


# =============================================================================
# ConsoleReportRenderer Tests
# =============================================================================


class TestConsoleReportRenderer:
    """Tests for the console report renderer."""

    def test_renders_basic_report(self, tmp_path: Path):
        lines: list[str] = []
        renderer = ConsoleReportRenderer(printer=lines.append)

        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="Score quality",
            data_path=tmp_path / "data.jsonl",
            number=1,
            results=[
                RecordResult(
                    record_index=1,
                    label="rec-1",
                    scores=[0.85],
                    explanations=["Good"],
                    errors=[],
                    statistics=calculate_statistics([0.85]),
                )
            ],
            overall_statistics=calculate_statistics([0.85]),
        )
        renderer.render(report)

        output = "\n".join(lines)
        assert "Model: openai/gpt-5.4" in output
        assert "Evaluated 1 record(s)" in output
        assert "[rec-1]" in output
        assert "score=0.8500" in output
        assert "explanation: Good" in output
        assert "Overall Statistics:" in output

    def test_renders_multi_run_summary(self, tmp_path: Path):
        lines: list[str] = []
        renderer = ConsoleReportRenderer(printer=lines.append)

        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="Score quality",
            data_path=tmp_path / "data.jsonl",
            number=2,
            results=[
                RecordResult(
                    record_index=1,
                    label="rec-1",
                    scores=[0.8, 0.9],
                    explanations=["Good", "Better"],
                    errors=[],
                    statistics=calculate_statistics([0.8, 0.9]),
                )
            ],
            overall_statistics=calculate_statistics([0.8, 0.9]),
        )
        renderer.render(report)

        output = "\n".join(lines)
        assert "Summary: avg=" in output

    def test_renders_errors(self, tmp_path: Path):
        lines: list[str] = []
        renderer = ConsoleReportRenderer(printer=lines.append)

        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="Score quality",
            data_path=tmp_path / "data.jsonl",
            number=1,
            results=[
                RecordResult(
                    record_index=1,
                    label="rec-1",
                    scores=[],
                    explanations=[],
                    errors=["Something went wrong"],
                    statistics=calculate_statistics([]),
                )
            ],
            overall_statistics=calculate_statistics([]),
        )
        renderer.render(report)

        output = "\n".join(lines)
        assert "ERROR: Something went wrong" in output


# =============================================================================
# JsonReportWriter Tests
# =============================================================================


class TestJsonReportWriter:
    """Tests for the JSON report writer."""

    def test_writes_valid_json(self, tmp_path: Path):
        writer = JsonReportWriter()
        output_path = tmp_path / "output" / "result.json"

        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="Score quality",
            data_path=tmp_path / "data.jsonl",
            number=1,
            results=[
                RecordResult(
                    record_index=1,
                    label="rec-1",
                    scores=[0.85],
                    explanations=["Good"],
                    errors=[],
                    statistics=calculate_statistics([0.85]),
                )
            ],
            overall_statistics=calculate_statistics([0.85]),
        )

        result_path = writer.write(report, output_path)

        assert result_path == output_path
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["model"] == "openai/gpt-5.4"
        assert data["rubric"] == "Score quality"
        assert data["number"] == 1
        assert len(data["records"]) == 1
        assert data["records"][0]["scores"] == [0.85]
        assert "generated_at" in data
        assert "overall_statistics" in data


# =============================================================================
# RubricCommand._resolve_rubric_text Tests
# =============================================================================


class TestResolveRubricText:
    """Tests for RubricCommand._resolve_rubric_text."""

    def test_inline_text_returned_stripped(self):
        result = RubricCommand._resolve_rubric_text("  Score quality  ")
        assert result == "Score quality"

    def test_file_reference_reads_content(self, tmp_path: Path):
        rubric_file = tmp_path / "rubric.txt"
        rubric_file.write_text("  Score factual accuracy.  \n", encoding="utf-8")

        result = RubricCommand._resolve_rubric_text(f"@{rubric_file}")
        assert result == "Score factual accuracy."

    def test_nonexistent_file_raises(self):
        with pytest.raises(CLIError, match="does not exist"):
            RubricCommand._resolve_rubric_text("@/nonexistent/rubric.txt")


# =============================================================================
# RubricCommand.run End-to-End Tests
# =============================================================================


_EVALUATE_RUBRIC_PATCH = "osmosis_ai.eval.rubric.cli.evaluate_rubric"


class TestRubricCommandRun:
    """End-to-end tests for RubricCommand.run."""

    def test_run_succeeds(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        mock_result = RubricResult(score=0.9, explanation="Correct", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ):
            RubricCommand().run(
                data=str(data_file),
                rubric="Score accuracy",
                model="openai/gpt-5.4",
                api_key="test-key",
            )

    def test_run_with_output_writes_json(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
        output_json = tmp_path / "result.json"

        mock_result = RubricResult(score=0.8, explanation="Good", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ):
            RubricCommand().run(
                data=str(data_file),
                rubric="Score quality",
                model="openai/gpt-5.4",
                api_key="test-key",
                output_path=str(output_json),
            )
        assert output_json.exists()
        data = json.loads(output_json.read_text(encoding="utf-8"))
        assert data["model"] == "openai/gpt-5.4"
        assert len(data["records"]) == 1
        assert data["records"][0]["scores"] == [0.8]

    def test_run_multiple_records_and_runs(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"messages": [{"role": "assistant", "content": f"Answer {i}"}]})
            for i in range(2)
        ]
        data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mock_result = RubricResult(score=0.7, explanation="OK", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ) as mock_eval:
            RubricCommand().run(
                data=str(data_file),
                rubric="Score it",
                model="openai/gpt-5.4",
                api_key="test-key",
                number=3,
            )
        # 2 records * 3 runs each = 6 calls total
        assert mock_eval.call_count == 6

    def test_run_with_rubric_file(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Test"}],
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        rubric_file = tmp_path / "rubric.txt"
        rubric_file.write_text("Score the response quality.", encoding="utf-8")

        mock_result = RubricResult(score=0.5, explanation="Average", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ) as mock_eval:
            RubricCommand().run(
                data=str(data_file),
                rubric=f"@{rubric_file}",
                model="openai/gpt-5.4",
                api_key="test-key",
            )
        # Verify the rubric text from file was passed to evaluate_rubric
        call_kwargs = mock_eval.call_args.kwargs
        assert call_kwargs["rubric"] == "Score the response quality."

    def test_run_nonexistent_data_raises(self):
        with pytest.raises(CLIError, match="does not exist"):
            RubricCommand().run(
                data="/nonexistent/data.jsonl",
                rubric="Score it",
                model="openai/gpt-5.4",
                api_key="test-key",
            )

    def test_run_directory_data_raises(self, tmp_path: Path):
        with pytest.raises(CLIError, match="Expected a file but received directory"):
            RubricCommand().run(
                data=str(tmp_path),
                rubric="Score it",
                model="openai/gpt-5.4",
                api_key="test-key",
            )

    def test_run_with_output_dir_creates_default_filename(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Hello"}],
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        output_dir = tmp_path / "output_dir"
        output_dir.mkdir()

        mock_result = RubricResult(score=0.6, explanation="Fair", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ):
            RubricCommand().run(
                data=str(data_file),
                rubric="Score quality",
                model="openai/gpt-5.4",
                api_key="test-key",
                output_path=str(output_dir),
            )
        expected_file = output_dir / "rubric_eval_result.json"
        assert expected_file.exists()

    def test_run_with_trailing_separator_uses_directory_mode(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {
            "messages": [{"role": "assistant", "content": "Hello"}],
        }
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        output_dir = tmp_path / "new_output_dir"

        mock_result = RubricResult(score=0.6, explanation="Fair", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ):
            RubricCommand().run(
                data=str(data_file),
                rubric="Score quality",
                model="openai/gpt-5.4",
                api_key="test-key",
                output_path=f"{output_dir}{os.sep}",
            )

        assert (output_dir / "rubric_eval_result.json").exists()
        assert not output_dir.is_file()

    def test_run_solution_str_records(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        record = {"solution_str": "The answer is 42."}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        mock_result = RubricResult(score=0.95, explanation="Excellent", raw={})

        with patch(
            _EVALUATE_RUBRIC_PATCH, new_callable=AsyncMock, return_value=mock_result
        ) as mock_eval:
            RubricCommand().run(
                data=str(data_file),
                rubric="Score accuracy",
                model="openai/gpt-5.4",
                api_key="test-key",
            )
        call_kwargs = mock_eval.call_args.kwargs
        assert call_kwargs["solution_str"] == "The answer is 42."

"""Tests for osmosis_ai.eval.rubric — dataset loader, report, and RubricCommand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat, override_output_context
from osmosis_ai.eval.rubric.cli import RubricCommand
from osmosis_ai.eval.rubric.dataset import RubricRecord, load_rubric_dataset
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

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            (
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "Hi there!",
            ),
            (
                [
                    {"role": "assistant", "content": "First"},
                    {"role": "user", "content": "Again"},
                    {"role": "assistant", "content": "Last"},
                ],
                "Last",
            ),
            (
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Part one"},
                            {"type": "image_url", "image_url": "ignored"},
                            {"type": "text", "text": "Part two"},
                        ],
                    }
                ],
                "Part one\nPart two",
            ),
        ],
    )
    def test_messages_format_loads_correctly(
        self, tmp_path: Path, messages: list[dict], expected: str
    ) -> None:
        data_file = tmp_path / "data.jsonl"
        record = {"messages": messages}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records = load_rubric_dataset(data_file)

        assert len(records) == 1
        assert records[0].solution_str == expected
        assert records[0].ground_truth is None
        assert records[0].original_input is None
        assert records[0].metadata is None
        assert records[0].record_id is None

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

    def test_non_dict_json_raises(self, tmp_path: Path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("[1, 2, 3]\n", encoding="utf-8")

        with pytest.raises(CLIError, match="Expected JSON object"):
            load_rubric_dataset(data_file)

    @pytest.mark.parametrize(
        ("messages", "expected_error"),
        [
            ([], r"must include 'messages'.*or 'solution_str'"),
            ([{"role": "user", "content": "Hello"}], "at least one assistant message"),
        ],
    )
    def test_invalid_messages_raise(
        self, tmp_path: Path, messages: list[dict], expected_error: str
    ) -> None:
        data_file = tmp_path / "data.jsonl"
        record = {"messages": messages}
        data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(CLIError, match=expected_error):
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


@pytest.mark.parametrize(
    ("record_id", "index", "expected"),
    [("my-record-id", 5, "my-record-id"), (None, 3, "record[3]")],
)
def test_rubric_record_label(record_id: str | None, index: int, expected: str) -> None:
    record = RubricRecord("test", None, None, None, record_id)
    assert record.label(index) == expected


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
# Console report Tests
# =============================================================================


class TestConsoleReport:
    def test_renders_exact_output(self, tmp_path: Path):
        data_path = tmp_path / "data.jsonl"
        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="评分质量",
            data_path=data_path,
            number=2,
            results=[
                RecordResult(
                    record_index=1,
                    label="rec-1",
                    scores=[0.8, 0.9],
                    explanations=["Good", "Better"],
                    errors=[],
                    statistics=calculate_statistics([0.8, 0.9]),
                ),
                RecordResult(
                    record_index=2,
                    label="rec-2",
                    scores=[],
                    explanations=[],
                    errors=["Something went wrong"],
                    statistics=calculate_statistics([]),
                ),
            ],
            overall_statistics=calculate_statistics([0.8, 0.9]),
        )
        lines: list[str] = []
        ConsoleReportRenderer(lines.append).render(report)

        assert lines == [
            "Model: openai/gpt-5.4",
            f"Evaluated 2 record(s) from {data_path}",
            "Runs per record: 2",
            "",
            "[rec-1]",
            "  Run 01: score=0.8000",
            "    explanation: Good",
            "  Run 02: score=0.9000",
            "    explanation: Better",
            "  Summary: avg=0.8500 stdev=0.0500 min=0.8000 max=0.9000",
            "",
            "[rec-2]",
            "  ERROR: Something went wrong",
            "",
            "Overall Statistics:",
            "  average:  0.8500",
            "  stdev:    0.0500",
            "  min/max:  0.8000 / 0.9000",
        ]


# =============================================================================
# JSON report Tests
# =============================================================================


class TestJsonReport:
    def test_writes_valid_json(self, tmp_path: Path):
        output_path = tmp_path / "output" / "result.json"

        report = RubricReport(
            model="openai/gpt-5.4",
            rubric_text="评分质量",
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

        result_path = JsonReportWriter().write(report, output_path)

        assert result_path == output_path
        assert output_path.exists()

        raw = output_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert raw == json.dumps(data, indent=2, ensure_ascii=False)
        assert list(data) == [
            "generated_at",
            "model",
            "rubric",
            "data_path",
            "number",
            "overall_statistics",
            "records",
        ]
        assert data["generated_at"].endswith("+00:00")
        assert data["model"] == "openai/gpt-5.4"
        assert data["rubric"] == "评分质量"
        assert data["data_path"] == str(report.data_path)
        assert data["number"] == 1
        assert data["overall_statistics"] == calculate_statistics([0.85])
        assert len(data["records"]) == 1
        assert data["records"][0] == {
            "index": 1,
            "label": "rec-1",
            "scores": [0.85],
            "explanations": ["Good"],
            "errors": [],
            "statistics": calculate_statistics([0.85]),
        }
        assert data["records"][0] == report.results[0].to_payload()


# =============================================================================
# RubricCommand._resolve_rubric_text Tests
# =============================================================================


class TestResolveRubricText:
    """Tests for RubricCommand._resolve_rubric_text."""

    def test_inline_text_returned_stripped(self):
        result = RubricCommand._resolve_rubric_text("  Score quality  ")
        assert result == "Score quality"

    def test_nonexistent_file_raises(self):
        with pytest.raises(CLIError, match="does not exist"):
            RubricCommand._resolve_rubric_text("@/nonexistent/rubric.txt")


# =============================================================================
# RubricCommand.run End-to-End Tests
# =============================================================================


_EVALUATE_RUBRIC_PATCH = "osmosis_ai.eval.rubric.cli.evaluate_rubric"


class TestRubricCommandRun:
    """End-to-end tests for RubricCommand.run."""

    @pytest.fixture(autouse=True)
    def _json_output(self):
        with override_output_context(format=OutputFormat.json):
            yield

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

        mock_result = RubricResult(score=0.8, explanation="Good")

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

        mock_result = RubricResult(score=0.7, explanation="OK")

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
        rubric_file.write_text("  Score the response quality.  \n", encoding="utf-8")

        mock_result = RubricResult(score=0.5, explanation="Average")

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

        mock_result = RubricResult(score=0.6, explanation="Fair")

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

        mock_result = RubricResult(score=0.6, explanation="Fair")

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

        mock_result = RubricResult(score=0.95, explanation="Excellent")

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

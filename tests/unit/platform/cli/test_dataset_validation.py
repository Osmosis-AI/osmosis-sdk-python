"""Tests for dataset local validation (Step 0).

Covers: required column checks, head+tail sampling, format-specific validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.platform.cli.dataset import (
    _check_required_columns,
    _read_tail_lines,
    _validate_csv,
    _validate_jsonl,
    _validate_parquet,
)

# ---------------------------------------------------------------------------
# _check_required_columns
# ---------------------------------------------------------------------------


class TestCheckRequiredColumns:
    def test_all_present(self):
        assert (
            _check_required_columns(["system_prompt", "user_prompt", "ground_truth"])
            == []
        )

    def test_extra_columns_ok(self):
        cols = ["system_prompt", "user_prompt", "ground_truth", "extra", "another"]
        assert _check_required_columns(cols) == []

    def test_missing_one(self):
        errors = _check_required_columns(["system_prompt", "user_prompt"])
        assert len(errors) == 1
        assert "ground_truth" in errors[0]

    def test_missing_all(self):
        errors = _check_required_columns(["foo", "bar"])
        assert len(errors) == 1
        assert "ground_truth" in errors[0]
        assert "system_prompt" in errors[0]
        assert "user_prompt" in errors[0]

    def test_empty(self):
        errors = _check_required_columns([])
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# _read_tail_lines
# ---------------------------------------------------------------------------


class TestReadTailLines:
    def test_small_file(self, tmp_path: Path):
        f = tmp_path / "small.txt"
        f.write_text("a\nb\nc\n")
        assert _read_tail_lines(f, 100) == ["a", "b", "c"]

    def test_returns_last_n(self, tmp_path: Path):
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line{i}" for i in range(200)) + "\n")
        tail = _read_tail_lines(f, 5)
        assert tail == [f"line{i}" for i in range(195, 200)]

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert _read_tail_lines(f, 10) == []

    def test_no_trailing_newline(self, tmp_path: Path):
        f = tmp_path / "no_nl.txt"
        f.write_text("a\nb\nc")  # no trailing newline
        assert _read_tail_lines(f, 100) == ["a", "b", "c"]

    def test_chunk_boundary(self, tmp_path: Path):
        """When chunk_size < file_size, partial first line is dropped."""
        f = tmp_path / "big.txt"
        lines = [f"line-{i:04d}" for i in range(50)]
        f.write_text("\n".join(lines) + "\n")
        # Use a tiny chunk so we don't read from the start
        tail = _read_tail_lines(f, 3, chunk_size=64)
        assert len(tail) == 3
        # Should be the last 3 lines
        assert tail == lines[-3:]


# ---------------------------------------------------------------------------
# _validate_jsonl
# ---------------------------------------------------------------------------


def _make_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestValidateJsonl:
    def test_valid_file(self, tmp_path: Path):
        rows = [
            {"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"}
            for _ in range(10)
        ]
        f = _make_jsonl(tmp_path / "ok.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_missing_required_columns(self, tmp_path: Path):
        rows = [{"foo": "bar"} for _ in range(5)]
        f = _make_jsonl(tmp_path / "bad_cols.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("Missing required columns" in e for e in errors)

    def test_invalid_json_head(self, tmp_path: Path):
        f = tmp_path / "bad_head.jsonl"
        lines = ['{"system_prompt":"s","user_prompt":"u","ground_truth":"g"}'] * 5
        lines[2] = "not json"
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_jsonl(f)
        assert any("Line 3" in e for e in errors)

    def test_invalid_json_tail(self, tmp_path: Path):
        """Errors in the last 100 lines are caught even if head is clean."""
        f = tmp_path / "bad_tail.jsonl"
        good = '{"system_prompt":"s","user_prompt":"u","ground_truth":"g"}'
        lines = [good] * 200
        lines[-1] = "broken json{{"
        lines[-3] = "{bad"
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_jsonl(f)
        assert any("Near end of file" in e for e in errors)

    def test_small_file_no_tail_pass(self, tmp_path: Path):
        """Files <= 100 lines skip tail validation (already fully read)."""
        rows = [
            {"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"}
            for _ in range(50)
        ]
        f = _make_jsonl(tmp_path / "small.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_error_limit(self, tmp_path: Path):
        """At most 5 errors + summary are reported."""
        f = tmp_path / "many_errors.jsonl"
        lines = ["not json"] * 10
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_jsonl(f)
        # 5 real errors + 1 "showing first 5 errors"
        assert len(errors) == 6
        assert "showing first 5" in errors[-1]

    def test_blank_lines_skipped(self, tmp_path: Path):
        f = tmp_path / "blanks.jsonl"
        good = '{"system_prompt":"s","user_prompt":"u","ground_truth":"g"}'
        f.write_text(f"\n{good}\n\n{good}\n")
        assert _validate_jsonl(f) == []

    def test_columns_checked_from_tail_when_head_blank(self, tmp_path: Path):
        """If head is all blank lines, columns are still checked via tail."""
        f = tmp_path / "blank_head.jsonl"
        blank_lines = [""] * 101  # > 100 so tail validation triggers
        data_lines = ['{"foo": "bar"}'] * 50
        f.write_text("\n".join(blank_lines + data_lines) + "\n")
        errors = _validate_jsonl(f)
        assert any("Missing required columns" in e for e in errors)


# ---------------------------------------------------------------------------
# _validate_csv
# ---------------------------------------------------------------------------


class TestValidateCsv:
    def test_valid_file(self, tmp_path: Path):
        f = tmp_path / "ok.csv"
        f.write_text("system_prompt,user_prompt,ground_truth\ns,u,g\ns2,u2,g2\n")
        assert _validate_csv(f) == []

    def test_extra_columns_ok(self, tmp_path: Path):
        f = tmp_path / "extra.csv"
        f.write_text("system_prompt,user_prompt,ground_truth,extra\ns,u,g,e\n")
        assert _validate_csv(f) == []

    def test_missing_required_columns(self, tmp_path: Path):
        f = tmp_path / "bad_cols.csv"
        f.write_text("foo,bar\n1,2\n")
        errors = _validate_csv(f)
        assert any("Missing required columns" in e for e in errors)

    def test_no_header(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        errors = _validate_csv(f)
        assert errors == ["File has no header row"]

    def test_inconsistent_columns_head(self, tmp_path: Path):
        f = tmp_path / "bad_head.csv"
        f.write_text("system_prompt,user_prompt,ground_truth\ns,u,g\ns,u\n")
        errors = _validate_csv(f)
        assert any("expected 3 columns, got 2" in e for e in errors)

    def test_inconsistent_columns_tail(self, tmp_path: Path):
        f = tmp_path / "bad_tail.csv"
        header = "system_prompt,user_prompt,ground_truth"
        good = "s,u,g"
        lines = [header] + [good] * 200
        lines[-1] = "s,u"  # bad row at end
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_csv(f)
        assert any("Near end of file" in e for e in errors)

    def test_small_file_no_tail(self, tmp_path: Path):
        f = tmp_path / "small.csv"
        header = "system_prompt,user_prompt,ground_truth"
        good = "s,u,g"
        f.write_text("\n".join([header] + [good] * 20) + "\n")
        assert _validate_csv(f) == []

    def test_exactly_100_data_rows_no_tail_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """100 data rows are fully covered by head validation for CSV."""
        import osmosis_ai.platform.cli.dataset as dataset_module

        f = tmp_path / "exactly_100.csv"
        header = "system_prompt,user_prompt,ground_truth"
        good = "s,u,g"
        f.write_text("\n".join([header] + [good] * 100) + "\n")

        def _should_not_be_called(*_args, **_kwargs):
            raise AssertionError("Tail validation should not run for 100 data rows")

        monkeypatch.setattr(dataset_module, "_read_tail_lines", _should_not_be_called)

        assert _validate_csv(f) == []


# ---------------------------------------------------------------------------
# _validate_parquet
# ---------------------------------------------------------------------------


class TestValidateParquet:
    @pytest.fixture()
    def _has_pyarrow(self):
        pytest.importorskip("pyarrow")

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_valid_file(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"],
                "user_prompt": ["u"],
                "ground_truth": ["g"],
            }
        )
        f = tmp_path / "ok.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_missing_required_columns(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"foo": [1], "bar": [2]})
        f = tmp_path / "bad_cols.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("Missing required columns" in e for e in errors)

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_extra_columns_ok(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"],
                "user_prompt": ["u"],
                "ground_truth": ["g"],
                "extra": ["e"],
            }
        )
        f = tmp_path / "extra.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_corrupt_file(self, tmp_path: Path):
        f = tmp_path / "corrupt.parquet"
        f.write_bytes(b"not a parquet file")
        errors = _validate_parquet(f)
        assert any("Invalid parquet file" in e for e in errors)

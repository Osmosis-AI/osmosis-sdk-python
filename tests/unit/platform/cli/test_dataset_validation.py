"""Tests for dataset local validation (Step 0).

Covers: required column checks, head+tail sampling, format-specific validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.platform.cli.dataset import (
    PARQUET_VALIDATION_SKIPPED_WARNING,
    _check_metadata_value,
    _check_required_columns,
    _metadata_is_absent,
    _read_tail_lines,
    _validate_csv,
    _validate_file_with_warnings,
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
        errors = _check_required_columns(["system_prompt"])
        assert len(errors) == 1
        assert "user_prompt" in errors[0]

    def test_missing_all(self):
        errors = _check_required_columns(["foo", "bar"])
        assert len(errors) == 1
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
        f.write_text(f"\n{good}\n\n{good}\n{good}\n\n{good}\n")
        assert _validate_jsonl(f) == []

    def test_sparse_large_file_enforces_min_rows(self, tmp_path: Path):
        """Large files with mostly blank lines must still meet MIN_ROW_COUNT."""
        f = tmp_path / "sparse.jsonl"
        good = '{"system_prompt":"s","user_prompt":"u","ground_truth":"g"}'
        blank_lines = [""] * 101
        data_lines = [good] * 2
        f.write_text("\n".join(blank_lines + data_lines) + "\n")
        errors = _validate_jsonl(f)
        assert any("Dataset too small" in e for e in errors)

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
        rows = "s,u,g\n" * 5
        f.write_text(f"system_prompt,user_prompt,ground_truth\n{rows}")
        assert _validate_csv(f) == []

    def test_extra_columns_ok(self, tmp_path: Path):
        f = tmp_path / "extra.csv"
        rows = "s,u,g,e\n" * 5
        f.write_text(f"system_prompt,user_prompt,ground_truth,extra\n{rows}")
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
    def test_validate_file_with_warnings_skips_parquet_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Structured validation should surface one warning without console output."""
        import builtins
        from io import StringIO

        from osmosis_ai.cli.console import Console

        real_import = builtins.__import__

        def _block_pyarrow(name: str, *args, **kwargs):
            if name == "pyarrow.parquet" or name == "pyarrow":
                raise ImportError("mocked missing pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_pyarrow)

        import osmosis_ai.platform.cli.dataset as dataset_mod

        buf = StringIO()
        fake_console = Console(file=buf, force_terminal=False)
        monkeypatch.setattr(dataset_mod, "console", fake_console)

        f = tmp_path / "test.parquet"
        f.write_bytes(b"fake parquet data")

        errors, warnings = _validate_file_with_warnings(f, "parquet")

        assert errors == []
        assert warnings == [PARQUET_VALIDATION_SKIPPED_WARNING]
        assert buf.getvalue() == ""

    def test_missing_pyarrow_prints_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When pyarrow is not installed, a warning is printed and [] is returned."""
        import builtins
        from io import StringIO

        from osmosis_ai.cli.console import Console

        real_import = builtins.__import__

        def _block_pyarrow(name: str, *args, **kwargs):
            if name == "pyarrow.parquet" or name == "pyarrow":
                raise ImportError("mocked missing pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_pyarrow)

        import osmosis_ai.platform.cli.dataset as dataset_mod

        buf = StringIO()
        fake_console = Console(file=buf, force_terminal=False)
        monkeypatch.setattr(dataset_mod, "console", fake_console)

        f = tmp_path / "test.parquet"
        f.write_bytes(b"fake parquet data")
        result = _validate_parquet(f)
        assert result == []
        output = buf.getvalue()
        assert "pyarrow not installed" in output
        assert "pip install" in output

    @pytest.fixture()
    def _has_pyarrow(self):
        pytest.importorskip("pyarrow")

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_valid_file(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
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
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "extra": ["e"] * 5,
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


# ---------------------------------------------------------------------------
# Optional "metadata" column helpers
# ---------------------------------------------------------------------------


class TestMetadataHelpers:
    def test_absent_none(self):
        assert _metadata_is_absent(None) is True

    def test_absent_empty_string(self):
        assert _metadata_is_absent("") is True

    def test_absent_whitespace_string(self):
        assert _metadata_is_absent("   ") is True

    def test_not_absent_dict(self):
        assert _metadata_is_absent({"a": 1}) is False

    def test_not_absent_nonempty_string(self):
        assert _metadata_is_absent('{"a": 1}') is False

    def test_dict_passes(self):
        assert _check_metadata_value({"tools": ["x"]}, location="Line 1") == []

    def test_object_string_passes(self):
        assert _check_metadata_value('{"tools": ["x"]}', location="Line 1") == []

    def test_absent_passes(self):
        assert _check_metadata_value(None, location="Line 1") == []
        assert _check_metadata_value("", location="Line 1") == []

    def test_number_rejected(self):
        errors = _check_metadata_value(3, location="Line 1")
        assert len(errors) == 1
        assert "invalid metadata" in errors[0]

    def test_array_rejected(self):
        errors = _check_metadata_value([1, 2], location="Line 1")
        assert len(errors) == 1
        assert "invalid metadata" in errors[0]

    def test_json_array_string_rejected(self):
        errors = _check_metadata_value("[1, 2]", location="Line 1")
        assert len(errors) == 1
        assert "JSON object" in errors[0]

    def test_garbage_string_rejected(self):
        errors = _check_metadata_value("not json{{", location="Line 1")
        assert len(errors) == 1
        assert "not valid JSON" in errors[0]

    def test_root_empty_object_passes_per_cell(self):
        # {} mixed with keyed objects is valid; only an all-empty column is
        # rejected, which is a cross-row check (see TestMetadataCrossRow).
        assert _check_metadata_value({}, location="Line 1") == []
        assert _check_metadata_value("{}", location="Line 1") == []

    def test_nested_empty_object_rejected(self):
        errors = _check_metadata_value({"a": {}}, location="Line 1")
        assert len(errors) == 1
        assert "empty nested object" in errors[0]

    def test_nested_empty_object_string_rejected(self):
        errors = _check_metadata_value('{"a": {}}', location="Line 1")
        assert len(errors) == 1
        assert "empty nested object" in errors[0]

    def test_nested_empty_object_in_list_rejected(self):
        errors = _check_metadata_value({"items": [{}]}, location="Line 1")
        assert len(errors) == 1
        assert "empty nested object" in errors[0]

    def test_oversized_int_rejected(self):
        # Valid JSON, but Arrow cannot store an int beyond 64 bits.
        errors = _check_metadata_value({"big": 2**100}, location="Line 1")
        assert len(errors) == 1
        assert "too large" in errors[0]

    def test_oversized_int_in_string_rejected(self):
        errors = _check_metadata_value(
            '{"big": 99999999999999999999999999999999}', location="Line 1"
        )
        assert len(errors) == 1
        assert "too large" in errors[0]

    def test_int64_boundary_accepted(self):
        value = {"max": 2**63 - 1, "min": -(2**63)}
        assert _check_metadata_value(value, location="Line 1") == []


# ---------------------------------------------------------------------------
# Metadata validation: JSONL
# ---------------------------------------------------------------------------


def _base_row() -> dict:
    return {"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"}


class TestMetadataJsonl:
    def test_object_metadata_accepted(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": {"k": "v"}} for _ in range(5)]
        f = _make_jsonl(tmp_path / "obj.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_object_string_metadata_accepted(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": '{"k": "v"}'} for _ in range(5)]
        f = _make_jsonl(tmp_path / "str.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_absent_metadata_accepted(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": None} for _ in range(5)]
        rows.extend(_base_row() for _ in range(5))  # column missing entirely
        f = _make_jsonl(tmp_path / "absent.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_number_metadata_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": 3} for _ in range(5)]
        f = _make_jsonl(tmp_path / "num.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("invalid metadata" in e for e in errors)

    def test_array_metadata_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": [1, 2]} for _ in range(5)]
        f = _make_jsonl(tmp_path / "arr.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("invalid metadata" in e for e in errors)

    def test_json_array_string_metadata_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": "[1, 2]"} for _ in range(5)]
        f = _make_jsonl(tmp_path / "arr_str.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("JSON object" in e for e in errors)

    def test_garbage_string_metadata_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": "garbage{{"} for _ in range(5)]
        f = _make_jsonl(tmp_path / "garbage.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("not valid JSON" in e for e in errors)

    def test_metadata_tail_validation(self, tmp_path: Path):
        f = tmp_path / "tail.jsonl"
        good = json.dumps({**_base_row(), "metadata": {"k": "v"}})
        lines = [good] * 200
        lines[-1] = json.dumps({**_base_row(), "metadata": 3})
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_jsonl(f)
        assert any("invalid metadata" in e for e in errors)


# ---------------------------------------------------------------------------
# Metadata validation: cross-row rules (mirror the platform normalizer)
# ---------------------------------------------------------------------------


class TestMetadataCrossRow:
    def test_jsonl_all_empty_objects_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": {}} for _ in range(5)]
        f = _make_jsonl(tmp_path / "all_empty.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("all sampled metadata objects are empty" in e for e in errors)

    def test_jsonl_all_empty_object_strings_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": "{}"} for _ in range(5)]
        f = _make_jsonl(tmp_path / "all_empty_str.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("all sampled metadata objects are empty" in e for e in errors)

    def test_jsonl_empty_mixed_with_keyed_accepted(self, tmp_path: Path):
        # The platform only rejects a column whose non-null cells are ALL empty.
        rows = [{**_base_row(), "metadata": {}} for _ in range(3)]
        rows.extend({**_base_row(), "metadata": {"k": "v"}} for _ in range(3))
        f = _make_jsonl(tmp_path / "mixed_empty.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_jsonl_nested_empty_object_rejected(self, tmp_path: Path):
        rows = [{**_base_row(), "metadata": {"a": {}}} for _ in range(5)]
        f = _make_jsonl(tmp_path / "nested_empty.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("empty nested object" in e for e in errors)

    def test_jsonl_inconsistent_value_types_rejected(self, tmp_path: Path):
        rows = [
            {**_base_row(), "metadata": {"k": 1}},
            {**_base_row(), "metadata": {"k": "x"}},
        ]
        rows.extend({**_base_row(), "metadata": {"k": 2}} for _ in range(3))
        f = _make_jsonl(tmp_path / "mixed_types.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("inconsistent across rows" in e and "$.k" in e for e in errors)

    def test_jsonl_int_and_float_are_consistent(self, tmp_path: Path):
        # JSON has one "number" type; the platform promotes int+float to double.
        rows = [
            {**_base_row(), "metadata": {"k": 1}},
            {**_base_row(), "metadata": {"k": 1.5}},
        ]
        rows.extend({**_base_row(), "metadata": {"k": 2}} for _ in range(3))
        f = _make_jsonl(tmp_path / "num_mix.jsonl", rows)
        assert _validate_jsonl(f) == []

    def test_jsonl_bool_vs_number_rejected(self, tmp_path: Path):
        rows = [
            {**_base_row(), "metadata": {"k": True}},
            {**_base_row(), "metadata": {"k": 1}},
        ]
        f = _make_jsonl(tmp_path / "bool_num.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("inconsistent across rows" in e for e in errors)

    def test_jsonl_nested_type_mismatch_rejected(self, tmp_path: Path):
        rows = [
            {**_base_row(), "metadata": {"outer": {"k": 1}}},
            {**_base_row(), "metadata": {"outer": {"k": [1]}}},
        ]
        f = _make_jsonl(tmp_path / "nested_types.jsonl", rows)
        errors = _validate_jsonl(f)
        assert any("$.outer.k" in e for e in errors)

    def test_jsonl_type_mismatch_across_head_and_tail(self, tmp_path: Path):
        # The tracker spans head and tail samples of large files.
        f = tmp_path / "head_tail_types.jsonl"
        head = json.dumps({**_base_row(), "metadata": {"k": 1}})
        tail = json.dumps({**_base_row(), "metadata": {"k": "x"}})
        f.write_text("\n".join([head] * 150 + [tail]) + "\n")
        errors = _validate_jsonl(f)
        assert any("inconsistent across rows" in e for e in errors)

    def test_csv_all_empty_object_strings_rejected(self, tmp_path: Path):
        f = tmp_path / "all_empty.csv"
        header = "system_prompt,user_prompt,ground_truth,metadata"
        row = "s,u,g,{}"
        f.write_text("\n".join([header] + [row] * 5) + "\n")
        errors = _validate_csv(f)
        assert any("all sampled metadata objects are empty" in e for e in errors)

    def test_csv_inconsistent_value_types_rejected(self, tmp_path: Path):
        f = tmp_path / "mixed_types.csv"
        header = "system_prompt,user_prompt,ground_truth,metadata"
        rows = ['s,u,g,"{""k"": 1}"', 's,u,g,"{""k"": ""x""}"']
        f.write_text("\n".join([header, *rows]) + "\n")
        errors = _validate_csv(f)
        assert any("inconsistent across rows" in e for e in errors)

    def test_csv_nested_empty_object_rejected(self, tmp_path: Path):
        f = tmp_path / "nested_empty.csv"
        header = "system_prompt,user_prompt,ground_truth,metadata"
        row = 's,u,g,"{""a"": {}}"'
        f.write_text("\n".join([header] + [row] * 5) + "\n")
        errors = _validate_csv(f)
        assert any("empty nested object" in e for e in errors)


# ---------------------------------------------------------------------------
# Metadata validation: CSV
# ---------------------------------------------------------------------------


class TestMetadataCsv:
    def _header(self) -> str:
        return "system_prompt,user_prompt,ground_truth,metadata"

    def test_object_string_metadata_accepted(self, tmp_path: Path):
        f = tmp_path / "ok.csv"
        row = 's,u,g,"{""k"": ""v""}"'
        f.write_text("\n".join([self._header()] + [row] * 5) + "\n")
        assert _validate_csv(f) == []

    def test_absent_metadata_accepted(self, tmp_path: Path):
        f = tmp_path / "absent.csv"
        row = "s,u,g,"
        f.write_text("\n".join([self._header()] + [row] * 5) + "\n")
        assert _validate_csv(f) == []

    def test_json_array_string_metadata_rejected(self, tmp_path: Path):
        f = tmp_path / "arr.csv"
        row = 's,u,g,"[1, 2]"'
        f.write_text("\n".join([self._header()] + [row] * 5) + "\n")
        errors = _validate_csv(f)
        assert any("JSON object" in e for e in errors)

    def test_garbage_string_metadata_rejected(self, tmp_path: Path):
        f = tmp_path / "garbage.csv"
        row = "s,u,g,garbage{{"
        f.write_text("\n".join([self._header()] + [row] * 5) + "\n")
        errors = _validate_csv(f)
        assert any("not valid JSON" in e for e in errors)

    def test_metadata_tail_validation(self, tmp_path: Path):
        f = tmp_path / "tail.csv"
        good = 's,u,g,"{""k"": ""v""}"'
        bad = "s,u,g,garbage{{"
        lines = [self._header()] + [good] * 200
        lines[-1] = bad
        f.write_text("\n".join(lines) + "\n")
        errors = _validate_csv(f)
        assert any("Near end of file" in e and "metadata" in e for e in errors)

    def test_no_metadata_column_ok(self, tmp_path: Path):
        f = tmp_path / "no_meta.csv"
        header = "system_prompt,user_prompt,ground_truth"
        f.write_text("\n".join([header] + ["s,u,g"] * 5) + "\n")
        assert _validate_csv(f) == []


# ---------------------------------------------------------------------------
# Metadata validation: Parquet
# ---------------------------------------------------------------------------


class TestMetadataParquet:
    @pytest.fixture()
    def _has_pyarrow(self):
        pytest.importorskip("pyarrow")

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_struct_dtype_accepted(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": [{"k": "v"}] * 5,
            }
        )
        f = tmp_path / "struct.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_object_string_dtype_accepted(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": ['{"k": "v"}'] * 5,
            }
        )
        f = tmp_path / "str.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_absent_string_metadata_accepted(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        # String-typed column with absent (null) cells: parseability check
        # skips absent cells, so the column passes.
        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": pa.array([None] * 5, type=pa.string()),
            }
        )
        f = tmp_path / "absent.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_null_dtype_metadata_accepted(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        # An all-null metadata column (null dtype) means every cell is absent.
        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": pa.array([None] * 5, type=pa.null()),
            }
        )
        f = tmp_path / "null.parquet"
        pq.write_table(table, f)
        assert _validate_parquet(f) == []

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_int_dtype_rejected(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": [1, 2, 3, 4, 5],
            }
        )
        f = tmp_path / "int.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("Invalid metadata column" in e for e in errors)

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_json_array_string_dtype_rejected(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": ["[1, 2]"] * 5,
            }
        )
        f = tmp_path / "arr_str.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("JSON object" in e for e in errors)

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_all_empty_object_strings_rejected(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": ["{}"] * 5,
            }
        )
        f = tmp_path / "all_empty.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("all sampled metadata objects are empty" in e for e in errors)

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_inconsistent_value_types_rejected(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 2,
                "user_prompt": ["u"] * 2,
                "ground_truth": ["g"] * 2,
                "metadata": ['{"k": 1}', '{"k": "x"}'],
            }
        )
        f = tmp_path / "mixed_types.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("inconsistent across rows" in e for e in errors)

    @pytest.mark.usefixtures("_has_pyarrow")
    def test_nested_empty_object_string_rejected(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "system_prompt": ["s"] * 5,
                "user_prompt": ["u"] * 5,
                "ground_truth": ["g"] * 5,
                "metadata": ['{"a": {}}'] * 5,
            }
        )
        f = tmp_path / "nested_empty.parquet"
        pq.write_table(table, f)
        errors = _validate_parquet(f)
        assert any("empty nested object" in e for e in errors)

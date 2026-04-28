"""Plain renderer behavior tests."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.renderer import render
from osmosis_ai.cli.output.result import (
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    MessageResult,
    OperationResult,
)


def _render(result: Any) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.plain) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            render(result, ctx)
    return out.getvalue(), err.getvalue()


def test_detail_renders_label_value_lines() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1", "file_name": "train.jsonl"},
        fields=[
            DetailField(label="ID", value="ds_1"),
            DetailField(label="File", value="train.jsonl"),
        ],
    )
    stdout, stderr = _render(result)
    assert stdout == "ID: ds_1\nFile: train.jsonl\n"
    assert stderr == ""


def test_list_renders_tab_separated_without_heading() -> None:
    result = ListResult(
        title="Datasets",
        items=[
            {"id": "ds_1", "status": "uploaded"},
            {"id": "ds_2", "status": "pending"},
        ],
        total_count=2,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="id", label="ID"),
            ListColumn(key="status", label="Status"),
        ],
    )
    stdout, _ = _render(result)
    assert stdout.splitlines() == ["ds_1\tuploaded", "ds_2\tpending"]


def test_list_uses_display_items_for_plain_human_values() -> None:
    result = ListResult(
        title="Datasets",
        items=[
            {
                "file_name": "train[ok].jsonl",
                "file_size": 12345,
                "created_at": "2026-04-26T00:00:00Z",
            }
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="file_name", label="File"),
            ListColumn(key="file_size", label="Size"),
            ListColumn(key="created_at", label="Created"),
        ],
        display_items=[
            {
                "file_name": "train[ok].jsonl",
                "file_size": "12.1 KB",
                "created_at": "[dim]2026-04-26[/dim]",
            }
        ],
    )
    stdout, _ = _render(result)
    assert stdout == "train[ok].jsonl\t12.1 KB\t2026-04-26\n"


def test_list_skips_columns_marked_plain_false() -> None:
    result = ListResult(
        title="Datasets",
        items=[{"id": "ds_1", "status": "uploaded", "internal_id": "int_1"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="id", label="ID"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="internal_id", label="Internal ID", plain=False),
        ],
    )
    stdout, _ = _render(result)
    assert stdout.strip() == "ds_1\tuploaded"


def test_list_normalises_tabs_and_newlines_in_values() -> None:
    result = ListResult(
        title="X",
        items=[{"name": "first\tline\nbroken", "id": "1"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="name", label="Name"),
            ListColumn(key="id", label="ID"),
        ],
    )
    stdout, _ = _render(result)
    assert "\t" in stdout
    assert stdout.count("\t") == 1
    assert "\n" in stdout
    assert stdout.count("\n") == 1


def test_detail_unescapes_rich_escaped_display_values() -> None:
    result = DetailResult(
        title="Dataset",
        data={"file_name": "train[ok].jsonl"},
        fields=[
            DetailField(label="File", value="train\\[ok].jsonl"),
        ],
    )
    stdout, _ = _render(result)
    assert stdout == "File: train[ok].jsonl\n"


def test_operation_renders_concise_success_line() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        resource={"id": "dep_1", "checkpoint_name": "run-step-40"},
        message="Checkpoint deployed.",
        display_next_steps=["Test it: osmosis deployment info run-step-40"],
    )
    stdout, _ = _render(result)
    lines = stdout.splitlines()
    assert lines[0] == "Checkpoint deployed."
    assert any("Test it:" in line for line in lines)


def test_message_renders_text_only() -> None:
    stdout, _ = _render(MessageResult(message="Logged out."))
    assert stdout == "Logged out.\n"


def test_no_ansi_in_plain_stdout() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1"},
        fields=[DetailField(label="ID", value="ds_1")],
    )
    stdout, _ = _render(result)
    assert "\x1b[" not in stdout
    assert "\u2500" not in stdout

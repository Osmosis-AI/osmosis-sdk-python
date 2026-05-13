"""Rich renderer integration tests."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.renderer import render
from osmosis_ai.cli.output.result import (
    DetailField,
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    MessageResult,
    OperationResult,
)


def _render(result: Any) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.rich) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            render(result, ctx)
    return out.getvalue(), err.getvalue()


def test_rich_detail_uses_label_value_table() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1"},
        fields=[
            DetailField(label="ID", value="ds_1"),
            DetailField(label="File", value="train.jsonl"),
        ],
    )
    stdout, _ = _render(result)
    assert "ID" in stdout and "ds_1" in stdout
    assert "File" in stdout and "train.jsonl" in stdout


def test_rich_list_includes_column_headers() -> None:
    result = ListResult(
        title="Datasets",
        items=[{"id": "ds_1", "status": "uploaded"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="id", label="ID"),
            ListColumn(key="status", label="Status"),
        ],
    )
    stdout, _ = _render(result)
    assert "ID" in stdout and "Status" in stdout
    assert "ds_1" in stdout and "uploaded" in stdout


def test_rich_list_prints_display_hints() -> None:
    result = ListResult(
        title="Training Runs",
        items=[{"name": "run-a"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[ListColumn(key="name", label="Name", ratio=4, overflow="fold")],
        display_hints=["Use osmosis train status <name> for details."],
    )
    stdout, _ = _render(result)
    assert "Use osmosis train status <name> for details." in stdout


def test_rich_list_prints_display_hints_with_literal_brackets() -> None:
    result = ListResult(
        title="Training Runs",
        items=[{"name": "run-a"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[ListColumn(key="name", label="Name")],
        display_hints=["Use [name] literally"],
    )
    stdout, _ = _render(result)
    assert "[name]" in stdout


def test_rich_detail_prints_sections_after_table() -> None:
    from rich.table import Table

    section_table = Table(show_header=True)
    section_table.add_column("Checkpoint")
    section_table.add_row("ckpt-a")
    result = DetailResult(
        title="Training Run",
        data={"id": "run_1"},
        fields=[DetailField(label="Name", value="run-a")],
        sections=[
            DetailSection(
                rich=section_table,
                plain_lines=["Checkpoint: ckpt-a"],
            )
        ],
        display_hints=["Deploy: osmosis deploy ckpt-a"],
    )
    stdout, _ = _render(result)
    assert "Name" in stdout
    assert "ckpt-a" in stdout
    assert "Deploy: osmosis deploy ckpt-a" in stdout


def test_rich_list_renders_raw_cell_markup_as_text() -> None:
    result = ListResult(
        title="Models",
        items=[{"name": "[red]model[/red]"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[ListColumn(key="name", label="Name")],
    )
    stdout, _ = _render(result)
    assert "[red]model[/red]" in stdout


def test_rich_message_renders_message_text() -> None:
    stdout, _ = _render(MessageResult(message="Logged out."))
    assert "Logged out." in stdout


def test_rich_operation_renders_message() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        message="Deployed.",
        display_next_steps=["Try it now"],
    )
    stdout, _ = _render(result)
    assert "Deployed." in stdout
    assert "Try it now" in stdout


def test_rich_operation_renders_message_markup_as_text() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        message='Checkpoint "[red]danger[/red]" deployed.',
    )
    stdout, _ = _render(result)
    assert 'Checkpoint "[red]danger[/red]" deployed.' in stdout

"""Rich renderer integration tests."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.renderer import _can_protect_primary_column, render
from osmosis_ai.cli.output.result import (
    DetailField,
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    ListSection,
    MessageResult,
    OperationResult,
    SectionedListResult,
)


def _render(result: Any) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.rich) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            render(result, ctx)
    return out.getvalue(), err.getvalue()


def _render_at_width(result: Any, monkeypatch: Any, width: int) -> tuple[str, str]:
    from osmosis_ai.cli.console import Console as CliConsole

    monkeypatch.setattr(
        "osmosis_ai.cli.console.Console",
        lambda: CliConsole(force_terminal=True, no_color=True, width=width),
    )
    return _render(result)


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
        display_hints=["Use osmosis train info <name> for details."],
    )
    stdout, _ = _render(result)
    assert "Use osmosis train info <name> for details." in stdout


def test_rich_list_does_not_expand_table_to_terminal_width(monkeypatch) -> None:
    from rich.table import Table as RichTable

    created_tables = []

    class RecordingTable(RichTable):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_tables.append(self)

    monkeypatch.setattr("rich.table.Table", RecordingTable)
    result = ListResult(
        title="Training Runs",
        items=[{"name": "a-very-long-training-run-name", "status": "running"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
        ],
    )

    _render(result)

    assert created_tables
    assert created_tables[0].expand is False


def test_rich_list_prioritizes_name_column_when_width_is_tight(monkeypatch) -> None:
    name = "run-name-that-should-fit-before-other-columns"
    result = ListResult(
        title="Training Runs",
        items=[
            {
                "name": name,
                "status": "running",
                "reward": "0.123",
                "created_at": "2026-05-18",
                "id": "abcdef12",
            }
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="reward", label="Reward", no_wrap=True, ratio=1),
            ListColumn(key="created_at", label="Created", no_wrap=True, ratio=1),
            ListColumn(key="id", label="ID", no_wrap=True, ratio=1),
        ],
    )

    stdout, _ = _render_at_width(result, monkeypatch, 90)

    assert name in stdout
    assert "…" in stdout


def test_primary_column_protection_accounts_for_min_width() -> None:
    assert (
        _can_protect_primary_column(
            [
                ListColumn(key="name", label="Name", min_width=20),
                ListColumn(key="status", label="Status", min_width=10),
            ],
            [{"name": "run-a", "status": "running"}],
            console_width=25,
        )
        is False
    )


def test_rich_list_preserves_explicit_non_primary_wrapping(monkeypatch) -> None:
    from rich.table import Table as RichTable

    added_columns = []

    class RecordingTable(RichTable):
        def add_column(self, *args, **kwargs):
            added_columns.append((args, kwargs))
            return super().add_column(*args, **kwargs)

    monkeypatch.setattr("rich.table.Table", RecordingTable)
    result = ListResult(
        title="Training Runs",
        items=[{"name": "short-name", "status": "running status"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="name", label="Name", ratio=4, overflow="fold"),
            ListColumn(
                key="status",
                label="Status",
                no_wrap=True,
                overflow="fold",
                ratio=1,
            ),
        ],
    )

    _render_at_width(result, monkeypatch, 80)

    assert added_columns[1][1]["no_wrap"] is True
    assert added_columns[1][1]["overflow"] == "fold"


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


def test_rich_sectioned_list_renders_heading_and_table_per_section() -> None:
    result = SectionedListResult(
        sections=[
            ListSection(
                key="base_models",
                title="Base Models",
                items=[{"name": "Qwen/Qwen3", "creator": "brian"}],
                total_count=1,
                has_more=False,
                next_offset=None,
                columns=[
                    ListColumn(key="name", label="Name"),
                    ListColumn(key="creator", label="Created By"),
                ],
            ),
            ListSection(
                key="lora_models",
                title="LoRA Models",
                items=[{"name": "run-step-1", "step": "1"}],
                total_count=1,
                has_more=False,
                next_offset=None,
                columns=[
                    ListColumn(key="name", label="Name"),
                    ListColumn(key="step", label="Step"),
                ],
            ),
        ],
        display_hints=["Deploy a LoRA model with: osmosis model deploy <name>"],
    )
    stdout, _ = _render(result)
    assert "Base Models" in stdout
    assert "LoRA Models" in stdout
    assert "Created By" in stdout
    assert "Step" in stdout
    assert "Qwen/Qwen3" in stdout
    assert "run-step-1" in stdout
    assert "Deploy a LoRA model with: osmosis model deploy <name>" in stdout


def test_rich_sectioned_list_prints_per_section_truncation_hint() -> None:
    result = SectionedListResult(
        sections=[
            ListSection(
                key="base_models",
                title="Base Models",
                items=[{"name": "m-1"}, {"name": "m-2"}],
                total_count=10,
                has_more=True,
                next_offset=2,
                columns=[ListColumn(key="name", label="Name")],
            ),
            ListSection(
                key="lora_models",
                title="LoRA Models",
                items=[{"name": "l-1"}],
                total_count=1,
                has_more=False,
                next_offset=None,
                columns=[ListColumn(key="name", label="Name")],
            ),
        ],
    )
    stdout, _ = _render(result)
    assert "Showing 2 of 10." in stdout
    assert stdout.count("Showing") == 1


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
        display_hints=["Deploy: osmosis model deploy ckpt-a"],
    )
    stdout, _ = _render(result)
    assert "Name" in stdout
    assert "ckpt-a" in stdout
    assert "Deploy: osmosis model deploy ckpt-a" in stdout


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

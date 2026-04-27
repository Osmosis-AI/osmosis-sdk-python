"""Tests for the CommandResult shapes."""

from __future__ import annotations

from osmosis_ai.cli.output.result import (
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    MessageResult,
    OperationResult,
)


def test_detail_result_basic() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1", "file_name": "train.jsonl"},
        fields=[
            DetailField(label="ID", value="ds_1"),
            DetailField(label="File", value="train.jsonl"),
        ],
    )
    assert result.title == "Dataset"
    assert result.data["id"] == "ds_1"
    assert len(result.fields) == 2


def test_list_result_carries_pagination() -> None:
    result = ListResult(
        title="Datasets",
        items=[{"id": "ds_1"}, {"id": "ds_2"}],
        total_count=42,
        has_more=True,
        next_offset=2,
        columns=[
            ListColumn(key="id", label="ID"),
            ListColumn(key="status", label="Status"),
        ],
    )
    assert result.total_count == 42
    assert result.has_more is True
    assert result.next_offset == 2
    assert [column.key for column in result.columns] == ["id", "status"]


def test_list_result_next_offset_required_even_when_unsupported() -> None:
    result = ListResult(
        title="Caches",
        items=[],
        total_count=0,
        has_more=False,
        next_offset=None,
        columns=[],
    )
    assert result.next_offset is None


def test_operation_result_minimal() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        resource={"id": "ckpt_1", "checkpoint_name": "run-step-40"},
    )
    assert result.operation == "deploy"
    assert result.status == "success"
    assert result.resource is not None


def test_operation_result_with_structured_next_steps() -> None:
    result = OperationResult(
        operation="init",
        status="success",
        message="Workspace created.",
        next_steps_structured=[
            {"action": "open", "target": "https://app.osmosis.ai/ws"},
        ],
        display_next_steps=["Open the dashboard."],
    )
    assert result.next_steps_structured[0]["action"] == "open"
    assert result.display_next_steps == ["Open the dashboard."]


def test_message_result() -> None:
    result = MessageResult(message="Logged out successfully.")
    assert result.message == "Logged out successfully."

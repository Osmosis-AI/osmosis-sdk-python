"""JSON renderer envelope and parseability tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
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

GOLDEN = Path(__file__).resolve().parents[3] / "golden" / "cli_output"


def _render_to_json(result: Any) -> tuple[dict[str, Any], str]:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            render(result, ctx)
    return json.loads(out.getvalue()), err.getvalue()


def test_list_envelope_required_keys() -> None:
    result = ListResult(
        title="Datasets",
        items=[{"id": "ds_1"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[ListColumn(key="id", label="ID")],
    )
    payload, stderr = _render_to_json(result)
    expected_keys = json.loads(
        (GOLDEN / "list_envelope.json").read_text(encoding="utf-8")
    )["keys"]
    assert sorted(payload.keys()) == sorted(expected_keys)
    assert payload["schema_version"] == 1
    assert payload["items"] == [{"id": "ds_1"}]
    assert payload["next_offset"] is None
    assert stderr == ""


def test_list_envelope_supports_extra_keys() -> None:
    result = ListResult(
        title="Eval caches",
        items=[],
        total_count=0,
        has_more=False,
        next_offset=None,
        columns=[],
        extra={"workspace": "ws-a"},
    )
    payload, _ = _render_to_json(result)
    assert payload["workspace"] == "ws-a"


def test_list_envelope_extra_cannot_override_reserved_keys() -> None:
    result = ListResult(
        title="Datasets",
        items=[{"id": "ds_1"}],
        total_count=1,
        has_more=False,
        next_offset=None,
        columns=[],
        extra={
            "schema_version": 999,
            "items": [],
            "has_more": True,
            "workspace": "ws-a",
        },
    )
    payload, _ = _render_to_json(result)
    assert payload["schema_version"] == 1
    assert payload["items"] == [{"id": "ds_1"}]
    assert payload["has_more"] is False
    assert payload["workspace"] == "ws-a"


def test_detail_envelope_required_keys() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1"},
        fields=[DetailField(label="ID", value="ds_1")],
    )
    payload, _ = _render_to_json(result)
    expected_keys = json.loads(
        (GOLDEN / "detail_envelope.json").read_text(encoding="utf-8")
    )["keys"]
    assert sorted(payload.keys()) == sorted(expected_keys)
    assert payload["data"] == {"id": "ds_1"}


def test_operation_envelope_required_keys() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        resource={"id": "dep_1", "checkpoint_name": "run-step-40", "status": "active"},
    )
    payload, _ = _render_to_json(result)
    expected_keys = json.loads(
        (GOLDEN / "operation_envelope.json").read_text(encoding="utf-8")
    )["keys"]
    assert set(expected_keys["required"]).issubset(payload.keys())
    assert payload["status"] == "success"
    assert payload["operation"] == "deploy"


def test_operation_envelope_includes_structured_next_steps_only() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        next_steps_structured=[
            {"action": "deployment_info", "checkpoint_name": "run-step-40"}
        ],
        display_next_steps=["Inspect with: osmosis deployment info run-step-40"],
    )
    payload, _ = _render_to_json(result)
    assert payload["next_steps_structured"] == [
        {"action": "deployment_info", "checkpoint_name": "run-step-40"}
    ]
    assert "display_next_steps" not in payload


def test_operation_envelope_extra_cannot_override_reserved_keys() -> None:
    result = OperationResult(
        operation="deploy",
        status="success",
        extra={
            "status": "failed",
            "operation": "delete",
            "message": "from extra",
            "workspace": "ws-a",
        },
    )
    payload, _ = _render_to_json(result)
    assert payload["status"] == "success"
    assert payload["operation"] == "deploy"
    assert "message" not in payload
    assert payload["workspace"] == "ws-a"


def test_message_envelope_includes_schema_version() -> None:
    result = MessageResult(message="Logged out.")
    payload, _ = _render_to_json(result)
    assert payload == {"schema_version": 1, "message": "Logged out."}


def test_message_envelope_extra_cannot_override_reserved_keys() -> None:
    result = MessageResult(
        message="Logged out.",
        extra={"schema_version": 999, "message": "from extra", "workspace": "ws-a"},
    )
    payload, _ = _render_to_json(result)
    assert payload == {
        "schema_version": 1,
        "message": "Logged out.",
        "workspace": "ws-a",
    }


def test_no_ansi_or_rich_box_on_json_stdout() -> None:
    result = DetailResult(
        title="Dataset",
        data={"id": "ds_1"},
        fields=[DetailField(label="ID", value="ds_1")],
    )
    out = io.StringIO()
    with override_output_context(format=OutputFormat.json) as ctx:
        with redirect_stdout(out):
            render(result, ctx)
    raw = out.getvalue()
    assert "\x1b[" not in raw
    assert "\u2500" not in raw
    json.loads(raw)


def test_render_marks_output_emitted() -> None:
    result = MessageResult(message="ok")
    out = io.StringIO()
    with override_output_context(format=OutputFormat.json) as ctx:
        with redirect_stdout(out):
            render(result, ctx)
        assert ctx.output_emitted is True

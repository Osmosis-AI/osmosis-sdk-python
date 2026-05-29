"""Tests for render_command_result and verify_output_emitted safety hooks."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import click
import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.renderer import (
    render_command_result,
    verify_output_emitted,
)
from osmosis_ai.cli.output.result import MessageResult


def test_render_command_result_renders_command_result_in_json() -> None:
    out = io.StringIO()
    with override_output_context(format=OutputFormat.json) as ctx:
        with redirect_stdout(out):
            render_command_result(MessageResult(message="ok"))
        assert ctx.output_emitted is True
    payload = json.loads(out.getvalue())
    assert payload == {"schema_version": 1, "message": "ok"}


def test_render_command_result_passes_none_through_in_rich_mode() -> None:
    with override_output_context(format=OutputFormat.rich):
        render_command_result(None)


def test_render_command_result_rejects_none_in_json_mode() -> None:
    with override_output_context(format=OutputFormat.json):
        with pytest.raises(CLIError) as exc:
            render_command_result(None)
        assert exc.value.code == "INTERNAL"
        assert "structured output" in exc.value.message


def test_render_command_result_rejects_unsupported_type() -> None:
    with override_output_context(format=OutputFormat.json):
        with pytest.raises(CLIError) as exc:
            render_command_result(42)
        assert exc.value.code == "INTERNAL"


def test_verify_output_emitted_passes_when_emitted() -> None:
    with override_output_context(format=OutputFormat.json) as ctx:
        ctx.output_emitted = True
        verify_output_emitted()


def test_verify_output_emitted_passes_in_rich_mode() -> None:
    with override_output_context(format=OutputFormat.rich):
        verify_output_emitted()


def test_verify_output_emitted_emits_internal_error_when_silent_in_json() -> None:
    with override_output_context(format=OutputFormat.json) as ctx:
        assert ctx.output_emitted is False
        err = io.StringIO()
        with pytest.raises(click.exceptions.Exit) as exit_info:
            with redirect_stderr(err):
                verify_output_emitted()
        assert exit_info.value.exit_code == 1
        envelope = json.loads(err.getvalue())
        assert envelope["error"]["code"] == "INTERNAL"
        assert "structured output" in envelope["error"]["message"]


def test_verify_output_emitted_no_op_when_exception_in_flight() -> None:
    err = io.StringIO()
    try:
        raise RuntimeError("simulated")
    except RuntimeError:
        with override_output_context(format=OutputFormat.json):
            with redirect_stderr(err):
                verify_output_emitted()
    assert err.getvalue() == ""

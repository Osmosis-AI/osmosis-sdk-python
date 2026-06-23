"""`osmosis dev serve` ticket emission across output formats."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from osmosis_ai.cli.commands.dev import _emit_ticket
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.renderer import render_command_result


def _emit(fmt: OutputFormat, *, interactive: bool = False) -> tuple[str, str, bool]:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=fmt, interactive=interactive) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            _emit_ticket(8000, "TICKET123")
        emitted = ctx.output_emitted
    return out.getvalue(), err.getvalue(), emitted


def test_json_emits_parseable_envelope_on_stdout() -> None:
    out, err, emitted = _emit(OutputFormat.json)
    payload = json.loads(out)
    assert payload == {
        "schema_version": 1,
        "local_rollout_address": "TICKET123",
        "port": 8000,
    }
    assert err == ""
    assert emitted is True


def test_plain_emits_raw_ticket_on_stdout_hints_on_stderr() -> None:
    out, err, emitted = _emit(OutputFormat.plain)
    assert out == "TICKET123\n"
    assert "TICKET123" in err  # config snippet on stderr
    assert emitted is True


def test_none_return_is_clean_when_output_already_emitted() -> None:
    for fmt in (OutputFormat.json, OutputFormat.plain):
        with override_output_context(format=fmt, interactive=False) as ctx:
            ctx.output_emitted = True
            render_command_result(None)  # must not raise

        with override_output_context(format=fmt, interactive=False):
            with pytest.raises(CLIError):
                render_command_result(None)

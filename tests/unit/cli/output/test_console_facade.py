"""Tests for the Console facade lazy binding."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    override_output_context,
)


def test_console_print_is_silent_in_json_mode() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stdout(out), redirect_stderr(err):
            console = Console()
            console.print("This must not pollute JSON stdout.")
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_console_print_routes_to_stdout_in_rich_mode() -> None:
    out = io.StringIO()
    with override_output_context(format=OutputFormat.rich):
        with redirect_stdout(out):
            console = Console(file=out, force_terminal=False)
            console.print("Visible in rich mode.")
    assert "Visible in rich mode." in out.getvalue()


def test_console_print_error_always_writes_to_stderr() -> None:
    err = io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stderr(err):
            Console().print_error("Something went wrong.")
    assert "Something went wrong." in err.getvalue()


def test_console_spinner_is_silent_in_json_mode() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stdout(out), redirect_stderr(err):
            with Console().spinner("Loading workspaces..."):
                pass
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_console_spinner_writes_to_stderr_in_plain_mode() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.plain):
        with redirect_stdout(out), redirect_stderr(err):
            with Console().spinner("Loading..."):
                pass
    assert out.getvalue() == ""
    assert "Loading..." in err.getvalue()


def test_output_context_status_no_op_in_json() -> None:
    out, err = io.StringIO(), io.StringIO()
    output = OutputContext(format=OutputFormat.json, interactive=False)
    with redirect_stdout(out), redirect_stderr(err):
        with output.status("fetching..."):
            pass
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_console_table_is_silent_in_json_mode() -> None:
    out = io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stdout(out):
            Console().table([("ID", "ds_1")])
    assert out.getvalue() == ""


def test_console_print_warning_is_structured_json_on_stderr() -> None:
    # JSON mode emits a one-line warning envelope to stderr (not free text), so
    # stdout stays clean and stderr remains parseable as JSON Lines, while the
    # deprecation signal is still visible to machine consumers.
    import json

    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stdout(out), redirect_stderr(err):
            Console().print_warning("Deprecated CLI version.", code="DEPRECATION")

    assert out.getvalue() == ""
    payload = json.loads(err.getvalue())
    assert payload["schema_version"] == 1
    assert "cli_version" in payload
    assert payload["warning"] == {
        "code": "DEPRECATION",
        "message": "Deprecated CLI version.",
    }
    # Distinguishable from the error envelope.
    assert "error" not in payload


def test_console_print_warning_plain_mode_is_unstyled() -> None:
    err = io.StringIO()
    with override_output_context(format=OutputFormat.plain):
        with redirect_stderr(err):
            Console().print_warning("Deprecated CLI version.")
    value = err.getvalue()
    assert "warning: Deprecated CLI version." in value
    assert "⚠" not in value


def test_console_print_warning_writes_glyph_in_rich_mode() -> None:
    err = io.StringIO()
    with override_output_context(format=OutputFormat.rich):
        with redirect_stderr(err):
            Console().print_warning("Deprecated CLI version.")
    value = err.getvalue()
    assert "Deprecated CLI version." in value
    assert "⚠" in value

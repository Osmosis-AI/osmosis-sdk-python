"""Tests for OutputFormat, OutputContext, and context resolution."""

from __future__ import annotations

from io import StringIO

import click
import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    _argv_format_prescan,
    _output_context_var,
    default_output_context,
    get_output_context,
    install_output_context,
    override_output_context,
    resolve_format_selectors,
)


def test_output_format_values() -> None:
    assert OutputFormat.rich.value == "rich"
    assert OutputFormat.json.value == "json"
    assert OutputFormat.plain.value == "plain"


def test_default_output_context_is_rich_and_uses_stdin_isatty(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", StringIO())
    output = default_output_context()
    assert output.format is OutputFormat.rich
    assert output.interactive is False
    assert output.schema_version == 1
    assert output.output_emitted is False


def test_get_output_context_layer_1_uses_active_click_context() -> None:
    output = OutputContext(format=OutputFormat.plain, interactive=False)
    cmd = click.Command("dummy", callback=lambda: None)
    with click.Context(cmd) as ctx:
        ctx.obj = output
        assert get_output_context() is output


def test_get_output_context_layer_3_falls_back_to_contextvar() -> None:
    forced = OutputContext(format=OutputFormat.json, interactive=False)
    token = _output_context_var.set(forced)
    try:
        assert get_output_context() is forced
    finally:
        _output_context_var.reset(token)


def test_get_output_context_layer_4_uses_argv_prescan(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "dataset", "list"])
    output = get_output_context()
    assert output.format is OutputFormat.json
    assert output.interactive is False


def test_get_output_context_layer_5_default_when_unset(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis"])
    output = get_output_context()
    assert output.format is OutputFormat.rich


def test_install_output_context_sets_obj_and_contextvar() -> None:
    output = OutputContext(format=OutputFormat.plain, interactive=False)
    cmd = click.Command("dummy", callback=lambda: None)
    with click.Context(cmd) as ctx:
        install_output_context(ctx, output)
        assert ctx.obj is output
        assert _output_context_var.get() is output
    assert _output_context_var.get() is None


def test_argv_prescan_recognises_output_flags() -> None:
    assert _argv_format_prescan(["--json", "dataset", "list"]) is OutputFormat.json
    assert _argv_format_prescan(["--plain", "dataset", "list"]) is OutputFormat.plain


def test_argv_prescan_ignores_command_local_output() -> None:
    assert (
        _argv_format_prescan(["dataset", "download", "ds1", "--output", "./file"])
        is None
    )


def test_argv_prescan_ignores_unknown_flags() -> None:
    assert _argv_format_prescan(["--bogus", "dataset", "list"]) is None


def test_get_output_context_layer_2_uses_usage_error_ctx() -> None:
    output = OutputContext(format=OutputFormat.json, interactive=False)
    root_cmd = click.Command("root", callback=lambda: None)
    with click.Context(root_cmd) as ctx:
        ctx.obj = output
        exc = click.UsageError("Bad subcommand", ctx=ctx)
        assert exc.ctx is not None
        assert exc.ctx.find_root().obj is output


def test_resolve_format_selectors_default_is_rich() -> None:
    assert (
        resolve_format_selectors(json_alias=False, plain_alias=False)
        is OutputFormat.rich
    )


def test_resolve_format_selectors_json_alias() -> None:
    assert (
        resolve_format_selectors(json_alias=True, plain_alias=False)
        is OutputFormat.json
    )


def test_resolve_format_selectors_plain_alias() -> None:
    assert (
        resolve_format_selectors(json_alias=False, plain_alias=True)
        is OutputFormat.plain
    )


def test_resolve_format_selectors_json_and_plain_conflict() -> None:
    with pytest.raises(CLIError) as exc:
        resolve_format_selectors(json_alias=True, plain_alias=True)
    assert exc.value.code == "VALIDATION"


def test_override_output_context_helper_round_trip() -> None:
    assert _output_context_var.get() is None
    with override_output_context(format=OutputFormat.json) as output:
        assert output.format is OutputFormat.json
        assert _output_context_var.get() is output
    assert _output_context_var.get() is None


def test_rich_status_registers_active_spinner_on_shared_console() -> None:
    """The dominant `output.status(...)` path must register on the console.

    A warning emitted mid-spin (the upgrade nudge) pauses/resumes whatever
    `console._active_status` points at; if this spinner path failed to register,
    the spinner line would glue onto the warning and linger on screen — the
    exact regression this wiring prevents. Covers train/eval/secret/etc.
    """
    from osmosis_ai.cli.console import console

    ctx = OutputContext(format=OutputFormat.rich, interactive=True)
    assert console._active_status is None
    with ctx.status("Fetching training runs..."):
        assert console._active_status is not None
    assert console._active_status is None


def test_rich_status_skips_registration_in_non_interactive_mode() -> None:
    """Non-interactive rich falls back to a plain line, so nothing to pause."""
    from osmosis_ai.cli.console import console

    ctx = OutputContext(format=OutputFormat.rich, interactive=False)
    with ctx.status("Fetching training runs..."):
        assert console._active_status is None

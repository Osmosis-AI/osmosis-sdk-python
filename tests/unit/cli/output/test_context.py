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


def test_argv_prescan_recognises_format_flags() -> None:
    assert _argv_format_prescan(["--json", "dataset", "list"]) is OutputFormat.json
    assert _argv_format_prescan(["--plain", "dataset", "list"]) is OutputFormat.plain
    assert (
        _argv_format_prescan(["--format", "json", "dataset", "list"])
        is OutputFormat.json
    )
    assert (
        _argv_format_prescan(["--format=plain", "dataset", "list"])
        is OutputFormat.plain
    )


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
        resolve_format_selectors(None, json_alias=False, plain_alias=False)
        is OutputFormat.rich
    )


def test_resolve_format_selectors_json_alias() -> None:
    assert (
        resolve_format_selectors(None, json_alias=True, plain_alias=False)
        is OutputFormat.json
    )


def test_resolve_format_selectors_plain_alias() -> None:
    assert (
        resolve_format_selectors(None, json_alias=False, plain_alias=True)
        is OutputFormat.plain
    )


def test_resolve_format_selectors_explicit_value_wins_over_unset_aliases() -> None:
    assert (
        resolve_format_selectors(OutputFormat.json, json_alias=False, plain_alias=False)
        is OutputFormat.json
    )


def test_resolve_format_selectors_consistent_alias_and_value_is_ok() -> None:
    assert (
        resolve_format_selectors(OutputFormat.json, json_alias=True, plain_alias=False)
        is OutputFormat.json
    )


def test_resolve_format_selectors_json_and_plain_conflict() -> None:
    with pytest.raises(CLIError) as exc:
        resolve_format_selectors(None, json_alias=True, plain_alias=True)
    assert exc.value.code == "VALIDATION"


def test_resolve_format_selectors_format_and_alias_conflict() -> None:
    with pytest.raises(CLIError) as exc:
        resolve_format_selectors(OutputFormat.plain, json_alias=True, plain_alias=False)
    assert exc.value.code == "VALIDATION"


def test_override_output_context_helper_round_trip() -> None:
    assert _output_context_var.get() is None
    with override_output_context(format=OutputFormat.json) as output:
        assert output.format is OutputFormat.json
        assert _output_context_var.get() is output
    assert _output_context_var.get() is None

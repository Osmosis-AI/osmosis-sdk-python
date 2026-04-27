"""Root callback tests for --format/--json/--plain parsing."""

from __future__ import annotations

import json

import pytest

from osmosis_ai.cli import main as cli


def test_help_is_plain_text_even_with_json(capsys) -> None:
    exit_code = cli.main(["--json", "--help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Osmosis" in captured.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out)


def test_version_is_plain_text_even_with_json(capsys) -> None:
    exit_code = cli.main(["--json", "--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().startswith("osmosis-ai")


def test_conflicting_selectors_emit_validation_error(capsys) -> None:
    exit_code = cli.main(["--json", "--plain", "dataset", "list"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "conflict" in captured.err.lower() or "format" in captured.err.lower()


def test_json_mode_contextvar_resets_after_eager_help(capsys) -> None:
    from osmosis_ai.cli.output.context import _output_context_var

    cli.main(["--json", "--help"])
    capsys.readouterr()
    assert _output_context_var.get() is None

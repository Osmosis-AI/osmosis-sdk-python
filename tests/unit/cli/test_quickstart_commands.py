"""Tests for osmosis_ai.cli.commands.quickstart."""

from __future__ import annotations

import json
import re

import pytest
import typer

from osmosis_ai.cli import main as cli
from osmosis_ai.cli.main import _register_commands, app
from osmosis_ai.cli.output import OperationResult

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

HELP_TEXT = (
    "Interactive setup: sign in, clone your workspace, and get your first agent prompt."
)
MANUAL_SETUP_URL = "https://docs.osmosis.ai/platform/onboarding#manual-setup"


def _quickstart_command():
    _register_commands()
    return typer.main.get_command(app).commands["quickstart"]


def _flatten(text: str) -> str:
    return " ".join(ANSI_ESCAPE.sub("", text).split())


def test_quickstart_is_a_root_command() -> None:
    assert _quickstart_command().name == "quickstart"


def test_quickstart_help_text_is_frozen() -> None:
    assert _flatten(_quickstart_command().help or "") == HELP_TEXT


def test_root_help_lists_quickstart(capfd) -> None:
    exit_code = cli.main(["--plain", "--help"])
    out = _flatten(capfd.readouterr().out)

    assert exit_code == 0
    assert "quickstart" in out


def test_quickstart_help_exits_zero(capfd) -> None:
    exit_code = cli.main(["quickstart", "--help"])
    out = _flatten(capfd.readouterr().out)

    assert exit_code == 0
    assert HELP_TEXT in out


def test_quickstart_delegates_to_the_wizard_handler(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_quickstart(**kwargs: object) -> OperationResult:
        calls.append(kwargs)
        return OperationResult(
            operation="quickstart",
            status="success",
            resource={"intent": "train"},
        )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.quickstart.run_quickstart", fake_run_quickstart
    )

    exit_code = cli.main(["--json", "quickstart"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [{}]
    payload = json.loads(captured.out)
    assert payload["operation"] == "quickstart"
    assert payload["status"] == "success"
    assert payload["resource"]["intent"] == "train"


def test_quickstart_requires_an_interactive_terminal(capsys) -> None:
    exit_code = cli.main(["quickstart"])
    err = _flatten(capsys.readouterr().err)

    assert exit_code == 1
    assert "needs an interactive terminal" in err
    assert MANUAL_SETUP_URL in err


def test_quickstart_json_reports_the_interactive_requirement(capsys) -> None:
    exit_code = cli.main(["--json", "quickstart"])
    envelope = json.loads(capsys.readouterr().err)

    assert exit_code == 1
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert "cannot run with --json or --plain" in envelope["error"]["message"]
    assert MANUAL_SETUP_URL in envelope["error"]["message"]

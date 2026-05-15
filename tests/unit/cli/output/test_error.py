"""Structured error envelope and classification tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import click
import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.main import _handle_cli_error, main
from osmosis_ai.cli.output.error import (
    classify_error,
    command_path_for_error,
    emit_structured_error_to_stderr,
)
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)

GOLDEN = Path(__file__).resolve().parents[3] / "golden" / "cli_output"


def _capture_envelope(err: CLIError) -> dict[str, Any]:
    buf = io.StringIO()
    with redirect_stderr(buf):
        emit_structured_error_to_stderr(err, command="dataset list")
    return json.loads(buf.getvalue())


def _capture_json_usage_error_for_argv(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    rc = _handle_cli_error(
        click.UsageError("No such command."),
        argv=argv,
        exit_code=2,
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    return json.loads(captured.err)


def test_envelope_keys_match_golden() -> None:
    envelope = _capture_envelope(CLIError("Bad input.", code="VALIDATION"))
    expected = json.loads((GOLDEN / "error_envelope.json").read_text(encoding="utf-8"))
    assert sorted(envelope.keys()) == sorted(expected["keys"])
    assert envelope["schema_version"] == 1
    assert envelope["command"] == "dataset list"
    assert envelope["cli_version"]
    assert envelope["error"]["code"] == "VALIDATION"
    assert envelope["error"]["details"] == {}
    assert envelope["error"]["request_id"] is None


def test_envelope_includes_platform_details() -> None:
    err = PlatformAPIError(
        "Validation failed.",
        status_code=400,
        error_code="invalid_dataset_name",
        details={"field": "name"},
    )
    envelope = _capture_envelope(classify_error(err))
    assert envelope["error"]["code"] == "VALIDATION"
    assert envelope["error"]["details"]["platform_code"] == "invalid_dataset_name"
    assert envelope["error"]["details"]["field"] == "name"


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, "VALIDATION"),
        (401, "AUTH_REQUIRED"),
        (403, "PLATFORM_ERROR"),
        (404, "NOT_FOUND"),
        (409, "CONFLICT"),
        (429, "RATE_LIMITED"),
        (500, "PLATFORM_ERROR"),
        (502, "PLATFORM_ERROR"),
    ],
)
def test_platform_error_status_mapping(status: int, expected_code: str) -> None:
    cli_err = classify_error(PlatformAPIError("x", status_code=status))
    assert cli_err.code == expected_code


def test_authentication_expired_error_maps_to_auth_required() -> None:
    cli_err = classify_error(AuthenticationExpiredError("expired"))
    assert cli_err.code == "AUTH_REQUIRED"


def test_unknown_exception_maps_to_internal_with_safe_details() -> None:
    cli_err = classify_error(RuntimeError("traceback contains secrets"))
    assert cli_err.code == "INTERNAL"
    assert cli_err.details == {"exception_type": "RuntimeError"}
    assert "secrets" not in cli_err.message


def test_cli_error_is_returned_unchanged() -> None:
    original = CLIError("Bad", code="NOT_FOUND")
    assert classify_error(original) is original


def test_command_path_falls_back_to_argv_when_no_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["osmosis", "--json", "dataset", "list", "--limit", "5"]
    )
    assert command_path_for_error(None) == "dataset list"


def test_command_path_fallback_excludes_top_level_argument(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "deploy", "ckpt-name"])
    assert command_path_for_error(None) == "deploy"


def test_command_path_fallback_keeps_eval_cache_subcommand(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["osmosis", "--json", "eval", "cache", "rm", "task-1"]
    )
    assert command_path_for_error(None) == "eval cache rm"


def test_command_path_uses_click_context_when_available() -> None:
    import click

    parent = click.Context(click.Command("osmosis"))
    parent.info_name = "osmosis"
    middle = click.Context(click.Command("dataset"), parent=parent)
    middle.info_name = "dataset"
    nested = click.Context(click.Command("list"), parent=middle)
    nested.info_name = "list"
    assert command_path_for_error(nested) == "dataset list"


def test_command_path_root_when_argv_empty(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis"])
    assert command_path_for_error(None) == "<root>"


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    [
        (["--json", "doctor"], "doctor"),
        (["--json", "workspace"], "workspace"),
        (["--json", "workspace", "list"], "workspace"),
        (["--json", "workspace", "create", "team"], "workspace"),
        (["--json", "workspace", "delete", "team"], "workspace"),
        (["--json", "workspace", "switch", "team"], "workspace"),
        (["--json", "login"], "login"),
        (["--json", "logout"], "logout"),
        (["--json", "whoami"], "whoami"),
        (["--json", "init"], "init"),
        (["--json", "link"], "link"),
        (["--json", "unlink"], "unlink"),
        (["--json", "project", "init"], "project"),
        (["--json", "project", "info"], "project"),
        (["--json", "project", "list"], "project"),
        (["--json", "dataset", "delete", "name"], "dataset delete"),
        (["--json", "train", "delete", "run"], "train delete"),
        (["--json", "train", "info", "run"], "train info"),
        (["--json", "train", "traces"], "train traces"),
        (["--json", "model", "delete", "name"], "model delete"),
        (["--json", "deployment", "rename", "old", "new"], "deployment rename"),
        (["--json", "deployment", "delete", "checkpoint"], "deployment delete"),
        (["--json", "rollout", "validate", "config.toml"], "rollout validate"),
    ],
)
def test_json_unknown_removed_command_uses_explicit_main_argv(
    argv: list[str],
    expected_command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "dataset", "list"])

    envelope = _capture_json_usage_error_for_argv(argv, capsys)

    assert envelope["command"] == expected_command
    assert envelope["error"]["code"] == "VALIDATION"


def test_json_unknown_command_from_main_uses_explicit_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "dataset", "list"])

    rc = main(["--json", "definitely-unknown", "extra"])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["command"] == "definitely-unknown"
    assert envelope["error"]["code"] == "VALIDATION"


def test_json_unknown_command_from_main_without_argv_uses_process_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "workspace", "list"])

    rc = main()

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["command"] == "workspace"
    assert envelope["error"]["code"] == "VALIDATION"

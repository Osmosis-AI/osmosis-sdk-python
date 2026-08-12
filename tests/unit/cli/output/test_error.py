"""Structured error envelope and classification tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import pytest
import typer
import typer.core

import osmosis_ai.cli.main as cli_main
from osmosis_ai.cli._click_compat import Context
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.main import main
from osmosis_ai.cli.output.error import (
    classify_error,
    command_path_for_error,
    emit_internal_debug,
    emit_structured_error_to_stderr,
)
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    SubscriptionRequiredError,
)

GOLDEN = Path(__file__).resolve().parents[3] / "golden" / "cli_output"


def _capture_envelope(err: CLIError) -> dict[str, Any]:
    buf = io.StringIO()
    with redirect_stderr(buf):
        emit_structured_error_to_stderr(err, command="dataset list")
    return json.loads(buf.getvalue())


def test_envelope_keys_match_golden() -> None:
    envelope = _capture_envelope(CLIError("Bad input.", code="VALIDATION"))
    expected = json.loads((GOLDEN / "error_envelope.json").read_text(encoding="utf-8"))
    assert sorted(envelope.keys()) == sorted(expected["keys"])
    assert envelope["schema_version"] == 1
    assert envelope["command"] == "dataset list"
    assert envelope["cli_version"]
    assert envelope["error"]["code"] == "VALIDATION"
    assert envelope["error"]["details"] == {}
    assert sorted(envelope["error"].keys()) == sorted(expected["error_keys"])
    assert "request_id" not in envelope["error"]


def test_emit_structured_error_omits_unserializable_details() -> None:
    err = CLIError(
        "boom",
        code="INTERNAL",
        details={"value": float("nan")},
    )

    envelope = _capture_envelope(err)

    assert envelope["error"]["code"] == "INTERNAL"
    assert envelope["error"]["message"] == "boom"
    assert envelope["error"]["details"] == {"details_omitted": "CLIError"}


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
        (426, "UPGRADE_REQUIRED"),
        (429, "RATE_LIMITED"),
        (500, "PLATFORM_ERROR"),
        (502, "PLATFORM_ERROR"),
    ],
)
def test_platform_error_status_mapping(status: int, expected_code: str) -> None:
    cli_err = classify_error(PlatformAPIError("x", status_code=status))
    assert cli_err.code == expected_code


def test_subscription_required_maps_to_dedicated_code() -> None:
    cli_err = classify_error(
        SubscriptionRequiredError("need sub", error_code="SUBSCRIPTION_REQUIRED")
    )
    assert cli_err.code == "SUBSCRIPTION_REQUIRED"
    assert cli_err.details["platform_code"] == "SUBSCRIPTION_REQUIRED"
    assert cli_err.details["status_code"] == 403


def test_billing_required_maps_to_billing_code() -> None:
    cli_err = classify_error(
        SubscriptionRequiredError("need billing", error_code="BILLING_REQUIRED")
    )
    assert cli_err.code == "BILLING_REQUIRED"


def test_generic_403_stays_platform_error() -> None:
    cli_err = classify_error(PlatformAPIError("forbidden", status_code=403))
    assert cli_err.code == "PLATFORM_ERROR"


def test_authentication_expired_error_maps_to_auth_required() -> None:
    cli_err = classify_error(AuthenticationExpiredError("expired"))
    assert cli_err.code == "AUTH_REQUIRED"


def test_upgrade_required_error_maps_to_upgrade_required() -> None:
    # UpgradeRequiredError is a PlatformAPIError subclass exported from the
    # package's public surface (parity with SubscriptionRequiredError).
    from osmosis_ai.platform.auth import UpgradeRequiredError

    cli_err = classify_error(UpgradeRequiredError("upgrade", status_code=426))
    assert cli_err.code == "UPGRADE_REQUIRED"


def test_unknown_exception_maps_to_internal_with_safe_details() -> None:
    cli_err = classify_error(RuntimeError("traceback contains secrets"))
    assert cli_err.code == "INTERNAL"
    assert cli_err.details == {"exception_type": "RuntimeError"}
    assert "secrets" not in cli_err.message


def test_osmos_debug_appends_internal_traceback_after_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_DEBUG", "1")

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("traceback contains secrets")

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", boom)

    rc = main(["--json", "dataset", "list"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    lines = captured.err.splitlines()
    envelope = json.loads(lines[0])
    assert envelope["error"]["code"] == "INTERNAL"
    assert envelope["error"]["message"] == "An unexpected internal error occurred."
    assert "secrets" not in envelope["error"]["message"]
    rest = "\n".join(lines[1:])
    assert "RuntimeError: traceback contains secrets" in rest
    assert "Traceback (most recent call last)" in rest


@pytest.mark.parametrize("explicit_cause", [True, False], ids=["cause", "context"])
def test_osmos_debug_dumps_nested_internal_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    explicit_cause: bool,
) -> None:
    monkeypatch.setenv("OSMOSIS_DEBUG", "1")

    def boom(*_: Any, **__: Any) -> None:
        nested: ValueError
        try:
            raise ValueError("nested serialization failure")
        except ValueError as exc:
            nested = exc
        error = CLIError("Renderer failed.", code="INTERNAL")
        if explicit_cause:
            raise error from nested
        error.__context__ = nested
        raise error

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", boom)

    rc = main(["--json", "dataset", "list"])

    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert rc == 1
    assert json.loads(lines[0])["error"]["code"] == "INTERNAL"
    assert "ValueError: nested serialization failure" in "\n".join(lines[1:])


def test_internal_json_error_without_debug_omits_original_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OSMOSIS_DEBUG", raising=False)

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("traceback contains secrets")

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", boom)

    rc = main(["--json", "dataset", "list"])

    captured = capsys.readouterr()
    assert rc == 1
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "INTERNAL"
    assert envelope["error"]["message"] == "An unexpected internal error occurred."
    assert "secrets" not in captured.err
    assert "Traceback" not in captured.err


def test_osmos_debug_rich_omits_traceback_for_platform_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_DEBUG", "1")

    def boom(*_: Any, **__: Any) -> None:
        raise PlatformAPIError("dataset missing", status_code=404)

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", boom)

    rc = main(["dataset", "list"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Error: dataset missing" in captured.err
    assert "Traceback" not in captured.err


def test_emit_internal_debug_classifies_lazily_when_unclassified(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_DEBUG", "1")

    emit_internal_debug(PlatformAPIError("dataset missing", status_code=404))

    assert capsys.readouterr().err == ""


def test_emit_internal_debug_still_dumps_unclassified_internal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_DEBUG", "1")

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        emit_internal_debug(exc)

    err = capsys.readouterr().err
    assert "RuntimeError: boom" in err
    assert "Traceback (most recent call last)" in err


def test_rich_internal_error_prints_original_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("human-readable root cause")

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", boom)

    rc = main(["dataset", "list"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "Error: human-readable root cause" in captured.err
    assert "An unexpected internal error occurred." not in captured.err


def test_cli_error_is_returned_unchanged() -> None:
    original = CLIError("Bad", code="NOT_FOUND")
    assert classify_error(original) is original


def test_command_path_falls_back_to_argv_when_no_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["osmosis", "--json", "dataset", "list", "--limit", "5"]
    )
    assert command_path_for_error(None) == "dataset list"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["osmosis", "--json", "benchmark", "info", "HLE"],
            "benchmark info",
        ),
        (
            ["osmosis", "--json", "benchmark", "runs", "download", "hle-smoke"],
            "benchmark runs download",
        ),
    ],
)
def test_benchmark_command_path_falls_back_to_full_command(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: str,
) -> None:
    monkeypatch.setattr("sys.argv", argv)
    assert command_path_for_error(None) == expected


def test_command_path_fallback_excludes_top_level_argument(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "deploy", "ckpt-name"])
    assert command_path_for_error(None) == "deploy"


def test_command_path_prefers_click_context_over_argv() -> None:
    parent = Context(typer.core.TyperCommand(name="osmosis"))
    parent.info_name = "osmosis"
    middle = Context(typer.core.TyperCommand(name="dataset"), parent=parent)
    middle.info_name = "dataset"
    nested = Context(typer.core.TyperCommand(name="list"), parent=middle)
    nested.info_name = "list"
    assert command_path_for_error(nested, argv=["train", "list"]) == "dataset list"


def test_dev_server_up_command_path_from_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "dev", "server", "up"])
    assert command_path_for_error(None) == "dev server up"


def test_eval_cache_is_not_a_three_token_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis", "--json", "eval", "cache", "clear"])
    assert command_path_for_error(None) == "eval cache"


def test_command_registry_matches_registered_app() -> None:
    from osmosis_ai.cli.command_registry import COMMAND_GROUPS, STANDALONE_COMMANDS

    cli_main._register_commands()
    group_names = {info.name for info in cli_main.app.registered_groups}
    command_names = {info.name for info in cli_main.app.registered_commands}
    assert group_names == COMMAND_GROUPS
    assert command_names == STANDALONE_COMMANDS


def test_command_path_uses_click_context_when_available() -> None:
    parent = Context(typer.core.TyperCommand(name="osmosis"))
    parent.info_name = "osmosis"
    middle = Context(typer.core.TyperCommand(name="dataset"), parent=parent)
    middle.info_name = "dataset"
    nested = Context(typer.core.TyperCommand(name="list"), parent=middle)
    nested.info_name = "list"
    assert command_path_for_error(nested) == "dataset list"


def test_command_path_root_when_argv_empty(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["osmosis"])
    assert command_path_for_error(None) == "<root>"


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


def test_main_maps_click_abort_to_interrupt_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_abort(*_: Any, **__: Any) -> None:
        raise typer.Abort()

    monkeypatch.setattr(cli_main, "_register_commands", lambda: None)
    monkeypatch.setattr(cli_main, "app", raise_abort)

    assert main(["secret", "add", "OPENAI_API_KEY"]) == 130

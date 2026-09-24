"""Contract tests for applications that extend the public CLI."""

from __future__ import annotations

import json
from typing import Any

import pytest
import typer
from typer.main import get_command

import osmosis_ai.cli.main as cli
from osmosis_ai.cli._click_compat import Context
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.result import DetailResult


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_find_env_file", lambda: None)
    for name in (
        "OSMOSIS_PLATFORM_URL",
        "OSMOSIS_TOKEN",
        "OSMOSIS_TOKEN_PLATFORM_URL",
        "OSMOSIS_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def _extended_app() -> typer.Typer:
    target = cli.create_app(name="osmo")
    tools = typer.Typer()
    sessions = typer.Typer()

    @sessions.command()
    def show(identifier: str = "example") -> DetailResult:
        return DetailResult(title="Session", data={"id": identifier})

    @sessions.command()
    def fail() -> None:
        raise CLIError("Session not found.", code="NOT_FOUND")

    tools.add_typer(sessions, name="session")
    target.add_typer(tools, name="tools")
    return target


def test_factory_does_not_mutate_the_default_or_another_application() -> None:
    cli._register_commands()
    original = get_command(cli.app)
    first = _extended_app()
    second = cli.create_app()
    first_command = get_command(first)
    second_command = get_command(second)
    assert first is not cli.app and second is not first
    assert first_command.get_command(Context(first_command), "tools") is not None
    assert second_command.get_command(Context(second_command), "tools") is None
    assert original.get_command(Context(original), "tools") is None
    assert set(second_command.commands) == set(original.commands)


def test_nested_registrations_are_independent_but_handlers_are_shared() -> None:
    from osmosis_ai.cli.commands.dataset import app as dataset_app

    first = cli.create_app()
    second = cli.create_app()
    first_dataset = next(
        info.typer_instance
        for info in first.registered_groups
        if info.name == "dataset"
    )
    second_dataset = next(
        info.typer_instance
        for info in second.registered_groups
        if info.name == "dataset"
    )
    assert first_dataset is not None and second_dataset is not None
    assert first_dataset is not second_dataset and first_dataset is not dataset_app
    assert (
        first_dataset.registered_commands[0].callback
        is second_dataset.registered_commands[0].callback
        is dataset_app.registered_commands[0].callback
    )

    @first_dataset.command("extension")
    def extension() -> None:
        pass

    assert "extension" not in {info.name for info in second_dataset.registered_commands}
    assert "extension" not in {info.name for info in dataset_app.registered_commands}


def test_version_and_upgrade_are_explicit_customization_points(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def version() -> None:
        typer.echo("osmo 1.2.3 (osmosis-ai 4.5.6)")

    def upgrade(version: str | None = typer.Option(None, "--version")) -> DetailResult:
        return DetailResult(title="Upgrade", data={"version": version})

    target = cli.create_app(
        name="osmo", version_callback=version, upgrade_command=upgrade
    )
    assert cli.run_cli(target, ["--version"]) == 0
    assert capsys.readouterr().out == "osmo 1.2.3 (osmosis-ai 4.5.6)\n"
    assert cli.run_cli(target, ["upgrade", "--version", "1.2.2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == {"version": "1.2.2"}


@pytest.mark.parametrize("first_kind", ["command", "group"])
@pytest.mark.parametrize("second_kind", ["command", "group"])
def test_duplicate_root_names_are_rejected_before_running(
    first_kind: str, second_kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    target = cli.create_app()
    called = False

    def command() -> None:
        nonlocal called
        called = True

    for kind in (first_kind, second_kind):
        if kind == "command":
            target.command("extension")(command)
        else:
            group = typer.Typer()
            group.command("show")(command)
            target.add_typer(group, name="extension")

    assert cli.run_cli(target, ["extension"]) == 1
    assert "Duplicate CLI command: extension" in capsys.readouterr().err
    assert not called


def test_duplicate_nested_and_flattened_names_are_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def show() -> None:
        raise AssertionError("Must not run an ambiguous command.")

    for name in ("tools", None):
        target = cli.create_app()
        group = typer.Typer()
        group.command("show")(show)
        group.command("show")(show)
        target.add_typer(group, name=name)
        assert cli.run_cli(target, ["--help"]) == 1
        expected = "tools show" if name else "show"
        assert f"Duplicate CLI command: {expected}" in capsys.readouterr().err

    target = cli.create_app()
    unnamed = typer.Typer()
    unnamed.command("upgrade")(show)
    target.add_typer(unnamed)
    assert cli.run_cli(target, ["--help"]) == 1
    assert "Duplicate CLI command: upgrade" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "command", "exit_code", "code"),
    [
        (["tools", "session", "fail"], "tools session fail", 1, "NOT_FOUND"),
        (["tools", "session", "unknown"], "tools session unknown", 2, "VALIDATION"),
        (["tools", "unknown"], "tools unknown", 2, "VALIDATION"),
        (["unknown", "extra"], "unknown", 2, "VALIDATION"),
        (
            ["--workspace", "example", "tools", "session", "fail"],
            "tools session fail",
            1,
            "NOT_FOUND",
        ),
        (
            ["--workspace=example", "tools", "session", "fail"],
            "tools session fail",
            1,
            "NOT_FOUND",
        ),
        (
            ["--workspace", " ", "tools", "session", "show"],
            "tools session show",
            1,
            "VALIDATION",
        ),
        (
            ["--env-file", "/nonexistent/osmo.env", "tools", "session", "show"],
            "tools session show",
            2,
            "VALIDATION",
        ),
        (
            ["--env-file=/nonexistent/osmo.env", "tools", "session", "show"],
            "tools session show",
            2,
            "VALIDATION",
        ),
    ],
)
def test_errors_use_the_composed_tree_and_distribution_version(
    arguments: list[str],
    command: str,
    exit_code: int,
    code: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli.run_cli(_extended_app(), [*arguments, "--json"], cli_version="1.2.3")
        == exit_code
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["command"] == command
    assert envelope["cli_version"] == "1.2.3"
    assert envelope["error"]["code"] == code


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_uses_the_composed_program_name(
    shell: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("_OSMO_COMPLETE", f"complete_{shell}")
    monkeypatch.setenv("_TYPER_COMPLETE_ARGS", "osmo tools session ")
    monkeypatch.setenv("_TYPER_COMPLETE_FISH_ACTION", "get-args")
    monkeypatch.setenv("COMP_WORDS", "osmo tools session ")
    monkeypatch.setenv("COMP_CWORD", "3")
    monkeypatch.setattr("sys.argv", ["pytest"])
    assert cli.run_cli(_extended_app(), []) == 0
    captured = capsys.readouterr()
    assert "show" in captured.out and "fail" in captured.out
    assert captured.err == ""


def test_help_nudge_uses_the_composed_program_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.run_cli(_extended_app(), ["help"]) == 2
    assert "osmo --help" in capsys.readouterr().err


@pytest.mark.parametrize("alias", ["osmosis-ai", "osmosis_ai"])
def test_public_console_aliases_keep_their_completion_environment(
    alias: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", [f"/some/bin/{alias}"])
    monkeypatch.setenv(f"_{alias.replace('-', '_').upper()}_COMPLETE", "complete_zsh")
    monkeypatch.setenv("_TYPER_COMPLETE_ARGS", f"{alias} template ")
    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "apply" in captured.out and "list" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("format_flag", ["--json", "--plain"])
def test_root_scope_and_output_are_shared(
    format_flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from osmosis_ai.platform.workspace_scope import get_workspace_name

    target = cli.create_app(name="osmo")
    seen: list[dict[str, Any]] = []

    @target.command()
    def inspect() -> DetailResult:
        seen.append({"workspace": get_workspace_name()})
        return DetailResult(title="Scope", data=seen[-1])

    assert cli.run_cli(target, ["--workspace", "example", "inspect", format_flag]) == 0
    assert seen == [{"workspace": "example"}]
    assert get_workspace_name() is None
    captured = capsys.readouterr()
    assert captured.err == ""
    if format_flag == "--json":
        assert json.loads(captured.out)["data"] == seen[0]

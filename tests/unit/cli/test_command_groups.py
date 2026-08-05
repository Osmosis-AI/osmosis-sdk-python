"""Smoke tests for supported and hard-removed CLI commands."""

from __future__ import annotations

import re

import pytest
import typer

from osmosis_ai.cli.main import _register_commands, app, main

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

PRESERVED_ROOT_COMMANDS = [
    "auth",
    "doctor",
    "dataset",
    "train",
    "model",
    "benchmark",
    "rollout",
    "template",
    "eval",
    "upgrade",
]

# Commands registered but intentionally hidden from --help output.
HIDDEN_ROOT_COMMANDS = [
    "dev",
]


PRESERVED_HELP_COMMANDS = [
    [],
    ["--help"],
    ["--version"],
    ["auth", "--help"],
    ["doctor", "--help"],
    ["dataset", "--help"],
    ["dataset", "logs", "--help"],
    ["train", "--help"],
    ["train", "logs", "--help"],
    ["model", "--help"],
    ["model", "list", "--help"],
    ["model", "info", "--help"],
    ["model", "deploy", "--help"],
    ["model", "undeploy", "--help"],
    ["benchmark", "--help"],
    ["benchmark", "list", "--help"],
    ["benchmark", "info", "--help"],
    ["benchmark", "submit", "--help"],
    ["benchmark", "runs", "--help"],
    ["benchmark", "runs", "list", "--help"],
    ["benchmark", "runs", "info", "--help"],
    ["benchmark", "runs", "logs", "--help"],
    ["benchmark", "runs", "stop", "--help"],
    ["benchmark", "runs", "download", "--help"],
    ["rollout", "--help"],
    ["template", "--help"],
    ["eval", "--help"],
    ["eval", "logs", "--help"],
    ["upgrade", "--help"],
]


def _root_command_names() -> set[str]:
    _register_commands()
    click_command = typer.main.get_command(app)
    return set(click_command.commands)


def _root_help_command_names(output: str) -> set[str]:
    expected_command_names = set(PRESERVED_ROOT_COMMANDS) | set(HIDDEN_ROOT_COMMANDS)
    command_names = set()
    for line in output.splitlines():
        cleaned = ANSI_ESCAPE.sub("", line).strip()
        cleaned = cleaned.strip(" │┃║")
        if not cleaned:
            continue

        name = cleaned.split(maxsplit=1)[0]
        if name in expected_command_names:
            command_names.add(name)
    return command_names


def _flatten_help(output: str) -> str:
    """Normalize Rich help output for stable contract assertions."""
    return " ".join(ANSI_ESCAPE.sub("", output).split())


@pytest.mark.parametrize("args", PRESERVED_HELP_COMMANDS)
def test_preserved_help_commands_exit_zero(args, capfd):
    rc = main(args)
    assert rc == 0


def test_root_command_registry_includes_supported_groups():
    root_commands = _root_command_names()

    for command in PRESERVED_ROOT_COMMANDS:
        assert command in root_commands

    for command in HIDDEN_ROOT_COMMANDS:
        assert command in root_commands


def test_root_help_surface_lists_supported_groups(capfd):
    rc = main(["--plain", "--help"])
    captured = capfd.readouterr()
    root_help_commands = _root_help_command_names(captured.out)

    assert rc == 0
    for command in PRESERVED_ROOT_COMMANDS:
        assert command in root_help_commands

    for command in HIDDEN_ROOT_COMMANDS:
        assert command not in root_help_commands


def test_typer_027_required_argument_metavar_contract(capfd):
    rc = main(["secret", "set", "--help"])
    output = _flatten_help(capfd.readouterr().out)

    assert rc == 0
    # Typer 0.27 wraps required argument references in braces, preserves an
    # explicitly declared metavar's casing, and renders the Python type in <>.
    assert "secret set [OPTIONS] {NAME}" in output
    assert "NAME <str> Secret name. [required]" in output
    assert "--scope <str>" in output


def test_typer_027_declared_argument_name_contract(capfd):
    rc = main(["dataset", "upload", "--help"])
    output = _flatten_help(capfd.readouterr().out)

    assert rc == 0
    # Argument names retain their declared casing and are no longer rendered
    # in brackets in the Arguments panel.
    assert "dataset upload [OPTIONS] {file}" in output
    assert "file <str> Path to the file to upload. [required]" in output
    assert "[file]" not in output


def test_typer_027_numeric_option_metavar_contract(capfd):
    rc = main(["model", "list", "--help"])
    output = _flatten_help(capfd.readouterr().out)

    assert rc == 0
    assert "--limit <int range> [1<=x<=50]" in output


def test_help_command_nudges_to_help_flag(capfd):
    rc = main(["help"])
    captured = capfd.readouterr()

    assert rc != 0
    assert "Use 'osmosis --help'" in captured.err
    assert "Did you mean" not in captured.err

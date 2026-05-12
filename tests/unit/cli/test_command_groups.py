"""Smoke tests for supported and hard-removed CLI commands."""

from __future__ import annotations

import re

import pytest
import typer

from osmosis_ai.cli.main import _register_commands, app, main

REMOVED_COMMAND_PROBE = "--__removed-command-probe"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

REMOVED_ROOT_COMMANDS = [
    "workspace",
    "init",
    "link",
    "unlink",
    "login",
    "logout",
    "whoami",
]

PRESERVED_ROOT_COMMANDS = [
    "auth",
    "project",
    "dataset",
    "train",
    "model",
    "deployment",
    "rollout",
    "template",
    "eval",
    "deploy",
    "undeploy",
    "upgrade",
]


PRESERVED_HELP_COMMANDS = [
    [],
    ["--help"],
    ["--version"],
    ["auth", "--help"],
    ["project", "--help"],
    ["dataset", "--help"],
    ["train", "--help"],
    ["model", "--help"],
    ["deployment", "--help"],
    ["deploy", "--help"],
    ["undeploy", "--help"],
    ["rollout", "--help"],
    ["template", "--help"],
    ["eval", "--help"],
    ["upgrade", "--help"],
]


REMOVED_COMMANDS = [
    ["workspace", REMOVED_COMMAND_PROBE],
    ["workspace", "list", REMOVED_COMMAND_PROBE],
    ["workspace", "create", REMOVED_COMMAND_PROBE, "team"],
    ["workspace", "delete", REMOVED_COMMAND_PROBE, "team"],
    ["workspace", "switch", REMOVED_COMMAND_PROBE, "team"],
    ["dataset", "delete", REMOVED_COMMAND_PROBE, "data"],
    ["train", "delete", REMOVED_COMMAND_PROBE, "run"],
    ["train", "info", REMOVED_COMMAND_PROBE, "run"],
    ["train", "traces", REMOVED_COMMAND_PROBE],
    ["model", "delete", REMOVED_COMMAND_PROBE, "model"],
    ["init", REMOVED_COMMAND_PROBE],
    ["link", REMOVED_COMMAND_PROBE],
    ["unlink", REMOVED_COMMAND_PROBE],
    ["login", REMOVED_COMMAND_PROBE],
    ["logout", REMOVED_COMMAND_PROBE],
    ["whoami", REMOVED_COMMAND_PROBE],
    ["project", "init", REMOVED_COMMAND_PROBE, "demo"],
    ["project", "info", REMOVED_COMMAND_PROBE],
    ["project", "list", REMOVED_COMMAND_PROBE],
    ["deployment", "rename", REMOVED_COMMAND_PROBE, "old", "new"],
    ["deployment", "delete", REMOVED_COMMAND_PROBE, "checkpoint"],
    ["rollout", "validate", REMOVED_COMMAND_PROBE, "configs/eval/demo.toml"],
    ["eval", "cache", "dir", REMOVED_COMMAND_PROBE],
]


def _root_command_names() -> set[str]:
    _register_commands()
    click_command = typer.main.get_command(app)
    return set(click_command.commands)


def _root_help_command_names(output: str) -> set[str]:
    expected_command_names = set(REMOVED_ROOT_COMMANDS) | set(PRESERVED_ROOT_COMMANDS)
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


@pytest.mark.parametrize("args", PRESERVED_HELP_COMMANDS)
def test_preserved_help_commands_exit_zero(args, capfd):
    rc = main(args)
    assert rc == 0


@pytest.mark.parametrize("args", REMOVED_COMMANDS)
def test_removed_commands_are_unknown(args, capfd):
    rc = main(args)
    captured = capfd.readouterr()

    assert rc != 0
    assert "No such command" in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["workspace", "--help"],
        ["workspace", "list", "--help"],
        ["workspace", "create", "--help"],
        ["workspace", "delete", "--help"],
        ["workspace", "switch", "--help"],
        ["dataset", "delete", "--help"],
        ["train", "delete", "--help"],
        ["train", "info", "--help"],
        ["train", "traces", "--help"],
        ["model", "delete", "--help"],
        ["init", "--help"],
        ["link", "--help"],
        ["unlink", "--help"],
        ["login", "--help"],
        ["logout", "--help"],
        ["whoami", "--help"],
        ["project", "init", "--help"],
        ["project", "info", "--help"],
        ["project", "list", "--help"],
        ["deployment", "rename", "--help"],
        ["deployment", "delete", "--help"],
        ["rollout", "validate", "--help"],
        ["eval", "cache", "dir", "--help"],
    ],
)
def test_removed_help_paths_are_unknown(args, capfd):
    rc = main(args)
    captured = capfd.readouterr()

    assert rc != 0
    assert "No such command" in captured.err


def test_root_command_registry_does_not_include_removed_groups_or_aliases():
    root_commands = _root_command_names()

    for command in REMOVED_ROOT_COMMANDS:
        assert command not in root_commands

    for command in PRESERVED_ROOT_COMMANDS:
        assert command in root_commands


def test_root_help_surface_does_not_list_removed_groups_or_aliases(capfd):
    rc = main(["--plain", "--help"])
    captured = capfd.readouterr()
    root_help_commands = _root_help_command_names(captured.out)

    assert rc == 0
    for command in REMOVED_ROOT_COMMANDS:
        assert command not in root_help_commands

    for command in PRESERVED_ROOT_COMMANDS:
        assert command in root_help_commands


@pytest.mark.parametrize(
    ("args", "not_expected"),
    [
        (["loginn"], "Did you mean 'login'?"),
        (["workspac"], "Did you mean 'workspace'?"),
        (["train", "tracess"], "Did you mean 'traces'?"),
        (["deployment", "renam"], "Did you mean 'rename'?"),
    ],
)
def test_fuzzy_suggestions_do_not_offer_removed_commands(args, not_expected, capfd):
    rc = main(args)
    captured = capfd.readouterr()

    assert rc != 0
    assert not_expected not in captured.err

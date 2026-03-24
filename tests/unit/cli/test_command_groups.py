"""Smoke tests for two-level CLI command structure."""

from __future__ import annotations

import pytest

from osmosis_ai.cli.main import main


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--help"],
        ["--version"],
        ["auth", "--help"],
        ["workspace", "--help"],
        ["dataset", "--help"],
        ["train", "--help"],
        ["model", "--help"],
        ["environment", "--help"],
        ["eval", "--help"],
        ["upgrade", "--help"],
    ],
)
def test_help_exits_zero(args, capfd):
    """All --help commands should exit cleanly."""
    rc = main(args)
    assert rc == 0


@pytest.mark.parametrize(
    "args",
    [
        ["model", "deploy"],
        ["model", "export"],
        ["model", "build"],
        ["train", "submit"],
        ["train", "metrics"],
        ["train", "traces"],
        ["environment", "list"],
    ],
)
def test_placeholder_commands_exit_one(args):
    """Placeholder commands should exit with code 1."""
    rc = main(args)
    assert rc == 1


def test_deprecated_login_alias_works(capfd):
    """'osmosis login --help' should still work as a deprecated alias."""
    rc = main(["login", "--help"])
    assert rc == 0


def test_all_groups_in_help_output(capfd):
    """Root --help should list all command groups."""
    main(["--help"])
    captured = capfd.readouterr()
    output = captured.out.lower()
    for group in [
        "auth",
        "workspace",
        "dataset",
        "train",
        "model",
        "environment",
        "eval",
    ]:
        assert group in output, f"'{group}' not found in --help output"

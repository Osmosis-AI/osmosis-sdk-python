"""Tests for upgrade command helpers."""

from __future__ import annotations

import pytest

from osmosis_ai.cli.upgrade import (
    _detect_install_method,
    _get_upgrade_commands,
    _parse_version,
)

# ── _parse_version ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.2.3", (1, 2, 3)),
        ("0.1.0", (0, 1, 0)),
        ("10.20.30", (10, 20, 30)),
        ("bad", (0,)),
        ("", (0,)),
    ],
)
def test_parse_version(version: str, expected: tuple[int, ...]) -> None:
    assert _parse_version(version) == expected


def test_parse_version_comparison() -> None:
    assert _parse_version("1.2.3") < _parse_version("1.2.4")
    assert _parse_version("2.0.0") > _parse_version("1.9.9")
    assert _parse_version("1.0.0") == _parse_version("1.0.0")


# ── _detect_install_method ───────────────────────────────────────────


def test_detect_install_method_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.executable", "/home/user/.local/share/uv/tools/osmosis/bin/python"
    )
    assert _detect_install_method() == "uv_tool"


def test_detect_install_method_pipx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.executable", "/home/user/.local/pipx/venvs/osmosis/bin/python"
    )
    assert _detect_install_method() == "pipx"


def test_detect_install_method_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.executable", "/usr/bin/python3")
    assert _detect_install_method() == "pip"


# ── _get_upgrade_commands ────────────────────────────────────────────


def test_get_upgrade_commands_uv_tool() -> None:
    cmds = _get_upgrade_commands("uv_tool")
    assert len(cmds) == 1
    assert "uv" in cmds[0]


def test_get_upgrade_commands_pipx() -> None:
    cmds = _get_upgrade_commands("pipx")
    assert len(cmds) == 1
    assert "pipx" in cmds[0]


def test_get_upgrade_commands_pip() -> None:
    cmds = _get_upgrade_commands("pip")
    assert len(cmds) == 2


def test_get_upgrade_commands_unknown_defaults_to_pip() -> None:
    cmds = _get_upgrade_commands("unknown")
    assert len(cmds) == 2

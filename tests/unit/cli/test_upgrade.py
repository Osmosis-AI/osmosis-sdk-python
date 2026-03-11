"""Tests for upgrade command helpers."""

from __future__ import annotations

import pytest

from osmosis_ai.cli.upgrade import (
    _detect_install_method,
    _get_upgrade_commands,
    _is_up_to_date,
)

# ── _is_up_to_date ──────────────────────────────────────────────────


def test_is_up_to_date_simple_versions() -> None:
    assert _is_up_to_date("1.2.4", "1.2.3")
    assert _is_up_to_date("1.0.0", "1.0.0")
    assert not _is_up_to_date("1.2.3", "1.2.4")
    assert _is_up_to_date("2.0.0", "1.9.9")
    assert not _is_up_to_date("1.9.9", "2.0.0")


def test_is_up_to_date_prerelease_versions() -> None:
    # Pre-release is less than the release
    assert not _is_up_to_date("2.0.0a1", "2.0.0")
    assert not _is_up_to_date("2.0.0rc1", "2.0.0")
    assert not _is_up_to_date("2.0.0.dev1", "2.0.0")
    # Pre-release of a higher version is still greater than older release
    assert _is_up_to_date("2.0.0a1", "1.0.0")


def test_is_up_to_date_unparseable_assumes_upgrade_needed() -> None:
    # When versions can't be parsed at all, assume upgrade is needed
    assert not _is_up_to_date("bad", "1.0.0")


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

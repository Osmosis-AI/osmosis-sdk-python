"""Negative CLI contracts for removed local bootstrap commands."""

from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli import main as cli


def test_top_level_init_json_is_unknown_and_creates_no_workspace_directory(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["--json", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert not (tmp_path / "demo").exists()
    assert not (tmp_path / ".osmosis" / "project.toml").exists()


def test_project_init_json_is_unknown_and_creates_no_workspace_directory(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["--json", "project", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert not (tmp_path / "demo").exists()
    assert not (tmp_path / ".osmosis" / "project.toml").exists()

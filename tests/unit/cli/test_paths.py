"""Tests for shared CLI path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from osmosis_ai.cli.paths import parse_cli_path


def test_parse_cli_path_preserves_trailing_separator(tmp_path: Path) -> None:
    raw_path = f"{tmp_path / 'missing'}{os.sep}"

    parsed = parse_cli_path(raw_path)

    assert parsed.path == tmp_path / "missing"
    assert parsed.has_trailing_separator is True


def test_parse_cli_path_expands_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    parsed = parse_cli_path("~/out.json", expand_user=True)

    assert parsed.path == tmp_path / "out.json"
    assert parsed.has_trailing_separator is False

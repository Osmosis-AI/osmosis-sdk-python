"""Tests for shared CLI path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from osmosis_ai.cli.paths import display_path, parse_cli_path


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


def test_display_path_is_relative_inside_the_base(tmp_path: Path) -> None:
    assert (
        display_path(tmp_path / ".osmosis" / "evals" / "run-1", base=tmp_path)
        == ".osmosis/evals/run-1"
    )


def test_display_path_keeps_an_outside_path_absolute(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside" / "run-1"
    assert display_path(outside, base=tmp_path) == str(outside.resolve())

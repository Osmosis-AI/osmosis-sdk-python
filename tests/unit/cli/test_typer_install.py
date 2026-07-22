"""Regression test for the Typer packaging conflict."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_runtime_dependencies_do_not_upgrade_legacy_typer_slim() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    typer_requirements = [
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("typer")
    ]

    assert typer_requirements == ["typer>=0.26,<0.27"]

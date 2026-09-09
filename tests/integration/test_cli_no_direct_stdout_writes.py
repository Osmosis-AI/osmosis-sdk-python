"""Lint guardrail: converted CLI paths must not write directly to stdout."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN_PATHS = [
    ROOT / "osmosis_ai" / "cli" / "commands" / "eval.py",
    ROOT / "osmosis_ai" / "eval" / "rubric" / "cli.py",
    ROOT / "osmosis_ai" / "cli" / "upgrade.py",
]


@pytest.mark.parametrize("path", SCAN_PATHS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_direct_stdout_writes(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {
            "print",
            "sys.stdout.write",
            "typer.echo",
        }:
            line = getattr(node, "lineno", "?")
            pytest.fail(
                f"Direct stdout write at {path.relative_to(ROOT)}:{line}. "
                "Return CommandResult or use stderr for progress."
            )

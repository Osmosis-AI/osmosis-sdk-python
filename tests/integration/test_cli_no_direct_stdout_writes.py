"""Lint guardrail: converted CLI paths must not write directly to stdout."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN_PATHS = [
    ROOT / "osmosis_ai" / "cli" / "commands" / "eval.py",
    ROOT / "osmosis_ai" / "eval" / "rubric" / "cli.py",
    ROOT / "osmosis_ai" / "eval" / "evaluation" / "cli.py",
    ROOT / "osmosis_ai" / "cli" / "upgrade.py",
]


def _is_sys_stdout_write(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "stdout"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
    )


def _is_typer_echo(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "echo"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
    )


def _is_builtin_print(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    )


@pytest.mark.parametrize("path", SCAN_PATHS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_direct_stdout_writes(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            _is_builtin_print(node)
            or _is_sys_stdout_write(node)
            or _is_typer_echo(node)
        ):
            line = getattr(node, "lineno", "?")
            pytest.fail(
                f"Direct stdout write at {path.relative_to(ROOT)}:{line}. "
                "Return CommandResult or use stderr for progress."
            )

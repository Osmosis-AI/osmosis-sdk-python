from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from osmosis_ai.platform.cli.rollout_entrypoint import load_rollout_entrypoint


def test_load_rollout_entrypoint_supports_absolute_sibling_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_dir = tmp_path / "rollout"
    sibling_package = rollout_dir / "sibling_package"
    sibling_package.mkdir(parents=True)
    (sibling_package / "__init__.py").write_text("", encoding="utf-8")
    (sibling_package / "value.py").write_text("VALUE = 42\n", encoding="utf-8")
    (rollout_dir / "main.py").write_text(
        "from sibling_package.value import VALUE\n",
        encoding="utf-8",
    )

    rollout_dir_str = str(rollout_dir.resolve())
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != rollout_dir_str])
    try:
        module = load_rollout_entrypoint(rollout_dir, "main.py")
        assert module.VALUE == 42
    finally:
        sys.modules.pop("sibling_package", None)
        sys.modules.pop("sibling_package.value", None)


def test_load_rollout_entrypoint_supports_nested_relative_imports(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "rollout"
    package_dir = rollout_dir / "package"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "value.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package_dir / "main.py").write_text(
        textwrap.dedent(
            """\
            from .value import VALUE
            """
        ),
        encoding="utf-8",
    )

    module = load_rollout_entrypoint(rollout_dir, "package/main.py")

    assert module.VALUE == 42

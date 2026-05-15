"""Unit tests for the SDK-owned template registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    iter_template_files,
    list_templates,
)


def _write_workspace_template(root: Path) -> Path:
    """Create a minimal workspace-template checkout for catalog tests."""
    (root / "rollouts" / "multiply-local-strands").mkdir(parents=True)
    (root / "rollouts" / "multiply-local-strands" / "main.py").write_text(
        "# rollout\n", encoding="utf-8"
    )
    (root / "rollouts" / "multiply-local-strands" / "pyproject.toml").write_text(
        "[project]\nname = 'multiply-local-strands'\n", encoding="utf-8"
    )
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs" / "eval" / "multiply-local-strands.toml").write_text(
        "[eval]\n", encoding="utf-8"
    )
    (root / "configs" / "training").mkdir(parents=True)
    (root / "configs" / "training" / "multiply-local-strands.toml").write_text(
        "[experiment]\n", encoding="utf-8"
    )
    (root / "data").mkdir()
    (root / "data" / "multiply.jsonl").write_text("{}\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _write_workspace_template(tmp_path / "workspace-template")
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(root))
    return root


def test_list_templates_includes_sdk_catalog_recipes(workspace_template: Path) -> None:
    names = list_templates()
    assert "multiply-local-strands" in names
    assert "multiply-local-openai" in names
    assert "multiply-harbor-strands" in names
    assert names == sorted(names), "list_templates must return a sorted list"


def test_list_templates_excludes_hidden_entries(workspace_template: Path) -> None:
    names = list_templates()
    assert not any(name.startswith((".", "_")) for name in names)


@pytest.mark.parametrize(
    "bad_name",
    ["", ".secret", "_internal", "../etc", "nested/path", "back\\slash"],
)
def test_iter_template_files_rejects_invalid_names(
    workspace_template: Path, bad_name: str
) -> None:
    with pytest.raises(TemplateNotFoundError):
        iter_template_files(bad_name)


def test_iter_template_files_unknown_name(workspace_template: Path) -> None:
    with pytest.raises(TemplateNotFoundError):
        iter_template_files("definitely-does-not-exist")


def test_iter_template_files_returns_catalog_relative_paths(
    workspace_template: Path,
) -> None:
    files = iter_template_files("multiply-local-strands")
    rel_strs = {p.as_posix() for p in files}

    assert "rollouts/multiply-local-strands/main.py" in rel_strs
    assert "rollouts/multiply-local-strands/pyproject.toml" in rel_strs
    assert "configs/training/multiply-local-strands.toml" in rel_strs
    assert "configs/eval/multiply-local-strands.toml" in rel_strs
    assert "data/multiply.jsonl" in rel_strs

    for rel in files:
        assert isinstance(rel, Path)
        assert not rel.is_absolute()


def test_iter_template_files_missing_catalog_file_uses_user_facing_template_terms(
    workspace_template: Path,
) -> None:
    (workspace_template / "configs" / "eval" / "multiply-local-strands.toml").unlink()

    with pytest.raises(CLIError) as exc_info:
        iter_template_files("multiply-local-strands")

    message = str(exc_info.value).lower()
    assert "template file does not exist" in message
    assert "workspace template" not in message

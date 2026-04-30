"""Unit tests for the template cookbook registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    cookbook_root,
    iter_template_files,
    list_templates,
    template_path,
)


def test_list_templates_includes_multiply() -> None:
    names = list_templates()
    assert "multiply" in names
    assert names == sorted(names), "list_templates must return a sorted list"


def test_list_templates_excludes_hidden_entries() -> None:
    names = list_templates()
    assert not any(name.startswith((".", "_")) for name in names)


def test_template_path_returns_existing_directory() -> None:
    path = template_path("multiply")
    assert path.is_dir()


@pytest.mark.parametrize(
    "bad_name",
    ["", ".secret", "_internal", "../etc", "nested/path", "back\\slash"],
)
def test_template_path_rejects_invalid_names(bad_name: str) -> None:
    with pytest.raises(TemplateNotFoundError):
        template_path(bad_name)


def test_template_path_unknown_name() -> None:
    with pytest.raises(TemplateNotFoundError):
        template_path("definitely-does-not-exist")


def test_iter_template_files_returns_relative_paths() -> None:
    files = iter_template_files("multiply")
    rel_strs = {p.as_posix() for p in files}

    assert "rollouts/multiply/main.py" in rel_strs
    assert "rollouts/multiply/pyproject.toml" in rel_strs
    assert "rollouts/multiply/README.md" in rel_strs
    assert "rollouts/multiply/multiply_rollout/__init__.py" in rel_strs
    assert "rollouts/multiply/multiply_rollout/workflow.py" in rel_strs
    assert "rollouts/multiply/multiply_rollout/grader.py" in rel_strs
    assert "rollouts/multiply/multiply_rollout/tools.py" in rel_strs
    assert "rollouts/multiply/multiply_rollout/utils.py" in rel_strs
    assert "configs/training/multiply.toml" in rel_strs
    assert "configs/eval/multiply.toml" in rel_strs

    for rel in files:
        assert isinstance(rel, Path)
        assert not rel.is_absolute()


def test_cookbook_root_is_resource_traversable() -> None:
    root = cookbook_root()
    assert root.is_dir()
    children = {entry.name for entry in root.iterdir() if entry.is_dir()}
    assert "multiply" in children

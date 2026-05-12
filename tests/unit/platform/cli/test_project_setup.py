"""Tests for template-backed scaffold repair primitives."""

from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.platform.cli.scaffold import write_scaffold


def _make_existing_project(root: Path) -> Path:
    root.mkdir()
    return root


def test_write_scaffold_creates_repair_paths(tmp_path: Path) -> None:
    target = _make_existing_project(tmp_path / "project")

    write_scaffold(target, "project")

    assert not (target / ".osmosis" / "project.toml").exists()
    assert (target / "research" / "program.md").is_file()
    assert (target / ".osmosis" / "cache" / ".gitkeep").is_file()
    assert (target / "rollouts" / ".gitkeep").is_file()
    assert (target / "configs" / "eval" / ".gitkeep").is_file()
    assert (target / "configs" / "training" / ".gitkeep").is_file()
    assert (target / "configs" / "training" / "default.toml").is_file()
    assert (target / "data" / ".gitkeep").is_file()
    assert (target / "AGENTS.md").is_file()
    assert (target / "CLAUDE.md").is_file()
    assert (target / "configs" / "AGENTS.md").is_file()
    assert (target / ".claude" / "settings.json").is_file()
    assert not (target / ".git").exists()


def test_write_scaffold_renders_project_files(tmp_path: Path) -> None:
    from osmosis_ai.consts import PACKAGE_VERSION

    target = _make_existing_project(tmp_path / "cool-project")

    write_scaffold(target, "cool-project")

    pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
    readme = (target / "README.md").read_text(encoding="utf-8")

    assert not (target / ".osmosis" / "project.toml").exists()
    assert 'name = "cool-project"' in pyproject
    assert f'"osmosis-ai>={PACKAGE_VERSION}"' in pyproject
    assert "cool-project" in readme


def test_write_scaffold_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")

    existing_files = {
        "pyproject.toml": "custom pyproject",
        ".gitignore": "custom gitignore",
        "README.md": "custom readme",
        "research/program.md": "custom research",
        "rollouts/.gitkeep": "custom marker",
        "AGENTS.md": "custom agents",
    }
    for rel_path, content in existing_files.items():
        (target / rel_path).write_text(content, encoding="utf-8")

    write_scaffold(target, "project")

    for rel_path, content in existing_files.items():
        assert (target / rel_path).read_text(encoding="utf-8") == content


def test_write_scaffold_update_refreshes_agent_files_only(tmp_path: Path) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")

    refreshed_files = {
        "AGENTS.md": "custom agents",
        "CLAUDE.md": "custom claude",
        "configs/AGENTS.md": "custom config agents",
        ".claude/settings.json": "{}",
    }
    preserved_files = {
        "README.md": "custom readme",
        "research/program.md": "custom research",
        "configs/training/default.toml": "custom training",
    }
    for rel_path, content in {**refreshed_files, **preserved_files}.items():
        (target / rel_path).write_text(content, encoding="utf-8")

    write_scaffold(target, "project", update=True)

    for rel_path, content in refreshed_files.items():
        assert (target / rel_path).read_text(encoding="utf-8") != content
    for rel_path, content in preserved_files.items():
        assert (target / rel_path).read_text(encoding="utf-8") == content


def test_write_scaffold_respects_plugin_env_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OSMOSIS_PLUGIN_REPO", "my-org/my-plugins")
    monkeypatch.setenv("OSMOSIS_PLUGIN_MARKETPLACE", "my-marketplace")
    target = _make_existing_project(tmp_path / "project")

    write_scaffold(target, "project")

    agents = (target / "AGENTS.md").read_text(encoding="utf-8")
    settings = json.loads(
        (target / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert "codex plugin marketplace add my-org/my-plugins" in agents
    assert "codex plugin install my-marketplace" in agents
    assert settings["extraKnownMarketplaces"]["my-marketplace"]["source"]["repo"] == (
        "my-org/my-plugins"
    )
    assert settings["enabledPlugins"]["osmosis@my-marketplace"] is True

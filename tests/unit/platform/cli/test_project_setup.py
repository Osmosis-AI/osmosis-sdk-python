"""Tests for SDK-backed scaffold repair primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.scaffold import (
    official_scaffold_updates,
    refresh_agent_scaffold,
    write_scaffold,
)


def _write_workspace_template(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "AGENTS.md").write_text(
        "template agents\n"
        "codex plugin marketplace add Osmosis-AI/osmosis-plugins\n"
        "codex plugin install osmosis\n",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text("template claude\n", encoding="utf-8")
    (root / "configs" / "AGENTS.md").write_text(
        "template config agents\n", encoding="utf-8"
    )
    (root / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "extraKnownMarketplaces": {
                    "osmosis": {
                        "source": {
                            "source": "github",
                            "repo": "Osmosis-AI/osmosis-plugins",
                        }
                    }
                },
                "enabledPlugins": {"osmosis@osmosis": True},
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(autouse=True)
def workspace_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _write_workspace_template(tmp_path / "workspace-template")
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(root))
    return root


def _make_existing_project(root: Path) -> Path:
    root.mkdir()
    (root / ".osmosis").mkdir()
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    return root


def test_write_scaffold_creates_repair_paths(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")

    write_scaffold(target, "project")

    assert (target / ".osmosis" / "project.toml").read_text(
        encoding="utf-8"
    ) == "[project]\n"
    assert not (target / ".osmosis" / "research" / "program.md").exists()
    assert (target / ".osmosis" / "cache" / ".gitkeep").is_file()
    assert (target / "rollouts" / ".gitkeep").is_file()
    assert (target / "configs" / "eval" / ".gitkeep").is_file()
    assert (target / "configs" / "training" / ".gitkeep").is_file()
    assert not (target / "configs" / "training" / "default.toml").exists()
    assert (target / "data" / ".gitkeep").is_file()
    assert (target / "AGENTS.md").is_file()
    assert (target / "CLAUDE.md").is_file()
    assert (target / "configs" / "AGENTS.md").is_file()
    assert (target / ".claude" / "settings.json").is_file()
    assert (
        (target / "AGENTS.md")
        .read_text(encoding="utf-8")
        .startswith("template agents\n")
    )
    assert not (target / ".git").exists()


def test_write_scaffold_does_not_overwrite_existing_files(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")

    existing_files = {
        ".osmosis/project.toml": "custom project",
        "pyproject.toml": "custom pyproject",
        ".gitignore": "custom gitignore",
        "README.md": "custom readme",
        "rollouts/.gitkeep": "custom marker",
        "AGENTS.md": "custom agents",
    }
    for rel_path, content in existing_files.items():
        (target / rel_path).write_text(content, encoding="utf-8")

    write_scaffold(target, "project")

    for rel_path, content in existing_files.items():
        assert (target / rel_path).read_text(encoding="utf-8") == content


def test_write_scaffold_rejects_broken_symlinked_official_file(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    outside_file = tmp_path / "missing-agents.md"
    (target / "AGENTS.md").symlink_to(outside_file)

    with pytest.raises(CLIError) as exc_info:
        write_scaffold(target, "project")

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)
    assert not outside_file.exists()


def test_write_scaffold_rejects_symlinked_official_parent_directory(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    outside_dir = tmp_path / "outside-claude"
    outside_dir.mkdir()
    (target / ".claude").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(CLIError) as exc_info:
        write_scaffold(target, "project")

    assert exc_info.value.code == "CONFLICT"
    assert ".claude" in str(exc_info.value)
    assert not (outside_dir / "settings.json").exists()


def test_write_scaffold_update_does_not_overwrite_agent_files(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")

    refreshed_files = {
        "AGENTS.md": "custom agents",
        "CLAUDE.md": "custom claude",
        "configs/AGENTS.md": "custom config agents",
        ".claude/settings.json": "{}",
    }
    preserved_files = {
        ".osmosis/project.toml": "custom project",
        "README.md": "custom readme",
    }
    for rel_path, content in {**refreshed_files, **preserved_files}.items():
        (target / rel_path).write_text(content, encoding="utf-8")

    write_scaffold(target, "project", update=True)

    for rel_path, content in refreshed_files.items():
        assert (target / rel_path).read_text(encoding="utf-8") == content
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


def test_write_scaffold_missing_official_file_uses_user_facing_template_terms(
    tmp_path: Path, workspace_template: Path
) -> None:
    (workspace_template / "AGENTS.md").unlink()
    target = _make_existing_project(tmp_path / "project")

    with pytest.raises(CLIError) as exc_info:
        write_scaffold(target, "project")

    message = str(exc_info.value).lower()
    assert "template source is missing an official agent scaffold file" in message
    assert "workspace template" not in message


def test_official_scaffold_updates_reports_local_edits(tmp_path: Path) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    (target / "AGENTS.md").write_text("custom agents", encoding="utf-8")

    assert official_scaffold_updates(target) == ["AGENTS.md"]


def test_official_scaffold_updates_rejects_symlinked_official_file(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    outside_file = tmp_path / "outside-agents.md"
    outside_file.write_text("outside agents", encoding="utf-8")
    (target / "AGENTS.md").unlink()
    (target / "AGENTS.md").symlink_to(outside_file)

    with pytest.raises(CLIError) as exc_info:
        official_scaffold_updates(target)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)


def test_official_scaffold_updates_rejects_directory_at_official_file_path(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    (target / "AGENTS.md").unlink()
    (target / "AGENTS.md").mkdir()

    with pytest.raises(CLIError) as exc_info:
        official_scaffold_updates(target)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)


def test_refresh_agent_scaffold_refuses_local_edits_without_force(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    (target / "AGENTS.md").write_text("custom agents", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "custom agents"


def test_refresh_agent_scaffold_rejects_directory_at_scaffold_file_path(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    (target / "AGENTS.md").mkdir()

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)
    assert (target / "AGENTS.md").is_dir()


def test_refresh_agent_scaffold_does_not_add_missing_files_when_conflicts_exist(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    (target / "AGENTS.md").unlink()
    (target / "CLAUDE.md").write_text("custom claude", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target)

    assert exc_info.value.code == "CONFLICT"
    assert "CLAUDE.md" in str(exc_info.value)
    assert not (target / "AGENTS.md").exists()
    assert (target / "CLAUDE.md").read_text(encoding="utf-8") == "custom claude"


def test_refresh_agent_scaffold_force_overwrites_local_edits(tmp_path: Path) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    (target / "AGENTS.md").write_text("custom agents", encoding="utf-8")

    result = refresh_agent_scaffold(target, force=True)

    assert result["refreshed"] == ["AGENTS.md"]
    agents = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "custom agents" not in agents
    assert agents.startswith("template agents\n")


def test_refresh_agent_scaffold_rejects_symlinked_official_file(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    outside_file = tmp_path / "outside-agents.md"
    outside_file.write_text("outside agents", encoding="utf-8")
    (target / "AGENTS.md").unlink()
    (target / "AGENTS.md").symlink_to(outside_file)

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target, force=True)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)
    assert outside_file.read_text(encoding="utf-8") == "outside agents"


def test_refresh_agent_scaffold_rejects_broken_symlinked_official_file(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    outside_file = tmp_path / "missing-agents.md"
    (target / "AGENTS.md").unlink()
    (target / "AGENTS.md").symlink_to(outside_file)

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target, force=True)

    assert exc_info.value.code == "CONFLICT"
    assert "AGENTS.md" in str(exc_info.value)
    assert not outside_file.exists()


def test_refresh_agent_scaffold_rejects_symlinked_official_parent_directory(
    tmp_path: Path,
) -> None:
    target = _make_existing_project(tmp_path / "project")
    write_scaffold(target, "project")
    outside_dir = tmp_path / "outside-claude"
    outside_dir.mkdir()
    (target / ".claude" / "settings.json").unlink()
    (target / ".claude").rmdir()
    (target / ".claude").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(CLIError) as exc_info:
        refresh_agent_scaffold(target, force=True)

    assert exc_info.value.code == "CONFLICT"
    assert ".claude" in str(exc_info.value)
    assert not (outside_dir / "settings.json").exists()

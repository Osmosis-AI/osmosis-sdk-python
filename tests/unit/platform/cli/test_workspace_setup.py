"""Tests for workspace init (osmosis init <name>) core flow and error handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.errors import CLIError

# ── Error case: git not found ────────────────────────────────────


def test_init_raises_when_git_not_found(monkeypatch) -> None:
    """init raises CLIError when git is not on PATH."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: None)

    with pytest.raises(CLIError, match=r"[Gg]it"):
        init_module.init(name="test-ws")


# ── Error case: directory exists without workspace.toml ─────────────────


def test_init_raises_when_directory_exists_without_workspace_toml(
    monkeypatch, tmp_path: Path
) -> None:
    """init raises CLIError if target dir exists but has no .osmosis/workspace.toml."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    target = tmp_path / "my-workspace"
    target.mkdir()

    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="already exists"):
        init_module.init(name="my-workspace")


# ── Re-entry: directory exists with .osmosis/workspace.toml ─────────────


def _stub_scaffold_fns(monkeypatch, module) -> None:
    """Stub all scaffold/git/print functions to be no-ops."""
    for fn_name in (
        "_write_scaffold",
        "_git_init",
        "_git_initial_commit",
        "_update_workspace_metadata",
        "_print_next_steps",
    ):
        monkeypatch.setattr(module, fn_name, lambda *a, **kw: None)


def test_init_rerun_on_existing_workspace(monkeypatch, tmp_path: Path) -> None:
    """init enters update mode when target dir has .osmosis/workspace.toml."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    _stub_scaffold_fns(monkeypatch, init_module)

    target = tmp_path / "my-workspace"
    target.mkdir()
    osmosis_dir = target / ".osmosis"
    osmosis_dir.mkdir()
    (osmosis_dir / "workspace.toml").write_text('[workspace]\nname = "my-workspace"\n')

    monkeypatch.chdir(tmp_path)

    # Should NOT raise — enters update mode
    init_module.init(name="my-workspace")


# ── Cleanup on scaffold failure ──────────────────────────────────


def test_init_cleanup_on_scaffold_failure(monkeypatch, tmp_path: Path) -> None:
    """If scaffold fails, the created directory is cleaned up."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    def _fail_scaffold(*a, **kw):
        raise CLIError("scaffold failed")

    monkeypatch.setattr(init_module, "_write_scaffold", _fail_scaffold)

    monkeypatch.chdir(tmp_path)

    target = tmp_path / "fail-ws"
    assert not target.exists()

    with pytest.raises(CLIError, match="scaffold failed"):
        init_module.init(name="fail-ws")

    # Directory should have been cleaned up
    assert not target.exists()


# ── --here: raises when directory not empty ───────────────────────


def test_init_here_raises_when_not_empty(monkeypatch, tmp_path: Path) -> None:
    """init(here=True) raises CLIError if cwd is not empty."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    (tmp_path / "some_file.txt").write_text("content")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="not empty"):
        init_module.init(name="test-ws", here=True)


# ── --here: allows directory with only .git/ ──────────────────────


def test_init_here_allows_git_dir(monkeypatch, tmp_path: Path) -> None:
    """init(here=True) allows a directory containing only .git/."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    _stub_scaffold_fns(monkeypatch, init_module)

    (tmp_path / ".git").mkdir()

    monkeypatch.chdir(tmp_path)

    # Should NOT raise
    init_module.init(name="test-ws", here=True)


def test_init_raises_when_selected_workspace_has_connected_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """Fresh init should stop before creating a directory when the active workspace already has a repo."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(
        init_module,
        "_selected_workspace_git_context",
        lambda: {
            "workspace_name": "team-alpha",
            "git_sync_url": "https://platform.osmosis.ai/team-alpha/integrations/git",
            "has_github_app_installation": True,
            "connected_repo_url": "https://github.com/acme/rollouts",
        },
    )

    monkeypatch.chdir(tmp_path)

    target = tmp_path / "test-ws"
    with pytest.raises(CLIError, match="already connected"):
        init_module.init(name="test-ws")

    assert not target.exists()


def test_init_here_raises_when_selected_workspace_has_connected_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """init(here=True) should stop when the active workspace already has a connected repo."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(
        init_module,
        "_selected_workspace_git_context",
        lambda: {
            "workspace_name": "team-alpha",
            "git_sync_url": "https://platform.osmosis.ai/team-alpha/integrations/git",
            "has_github_app_installation": True,
            "connected_repo_url": "https://github.com/acme/rollouts",
        },
    )

    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match=r"git clone https://github\.com/acme/rollouts"):
        init_module.init(name="test-ws", here=True)


def test_init_ignores_auth_expiry_in_best_effort_workspace_lookup(
    monkeypatch, tmp_path: Path
) -> None:
    """init should continue when best-effort workspace Git metadata lookup hits 401."""
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.cli.init as init_module

    class _Creds:
        def is_expired(self) -> bool:
            return False

    def fake_platform_request(endpoint, **kwargs):
        assert endpoint == "/api/cli/workspaces"
        assert kwargs.get("cleanup_on_401") is False
        raise auth_module.AuthenticationExpiredError("session expired")

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    for fn_name in (
        "_write_scaffold",
        "_git_init",
        "_git_initial_commit",
        "_update_workspace_metadata",
    ):
        monkeypatch.setattr(init_module, fn_name, lambda *a, **kw: None)
    monkeypatch.setattr(init_module, "get_active_workspace_name", lambda: "team-alpha")
    monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
    monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)

    monkeypatch.chdir(tmp_path)

    init_module.init(name="test-ws")

    assert (tmp_path / "test-ws").is_dir()


def test_init_auto_selects_single_workspace_for_git_context(
    monkeypatch, tmp_path: Path
) -> None:
    """init should auto-select the only workspace when none is saved locally.

    This also persists the selection to local config so subsequent CLI
    commands see the same active workspace.
    """
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.cli.init as init_module

    class _Creds:
        def is_expired(self) -> bool:
            return False

    calls: list[tuple[str, dict]] = []
    persisted: dict[str, str] = {}

    def fake_platform_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        assert endpoint == "/api/cli/workspaces"
        assert kwargs.get("cleanup_on_401") is False
        return {
            "workspaces": [
                {
                    "id": "ws_solo",
                    "name": "solo-team",
                    "has_github_app_installation": False,
                    "connected_repo": None,
                }
            ]
        }

    def fake_set_active_workspace(workspace_id, workspace_name):
        persisted["id"] = workspace_id
        persisted["name"] = workspace_name

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    for fn_name in (
        "_write_scaffold",
        "_git_init",
        "_git_initial_commit",
        "_update_workspace_metadata",
    ):
        monkeypatch.setattr(init_module, fn_name, lambda *a, **kw: None)
    monkeypatch.setattr(init_module, "get_active_workspace_name", lambda: None)
    monkeypatch.setattr(init_module, "get_active_workspace_id", lambda: None)
    monkeypatch.setattr(init_module, "set_active_workspace", fake_set_active_workspace)
    monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
    monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)

    monkeypatch.chdir(tmp_path)

    init_module.init(name="test-ws")

    assert (tmp_path / "test-ws").is_dir()
    # Only one API call — fetch + verify + extract metadata are unified.
    assert len(calls) == 1
    # Auto-selection should have been persisted.
    assert persisted == {"id": "ws_solo", "name": "solo-team"}


def test_init_blocks_when_auto_selected_workspace_has_connected_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """init should raise when the auto-selected single workspace already has a connected repo."""
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.cli.init as init_module

    class _Creds:
        def is_expired(self) -> bool:
            return False

    def fake_platform_request(endpoint, **kwargs):
        assert endpoint == "/api/cli/workspaces"
        return {
            "workspaces": [
                {
                    "id": "ws_solo",
                    "name": "solo-team",
                    "has_github_app_installation": True,
                    "connected_repo": {
                        "repo_url": "https://github.com/acme/rollouts",
                    },
                }
            ]
        }

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(init_module, "get_active_workspace_name", lambda: None)
    monkeypatch.setattr(init_module, "get_active_workspace_id", lambda: None)
    monkeypatch.setattr(init_module, "set_active_workspace", lambda *a, **kw: None)
    monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
    monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)

    monkeypatch.chdir(tmp_path)

    target = tmp_path / "test-ws"
    with pytest.raises(CLIError, match="already connected"):
        init_module.init(name="test-ws")

    assert not target.exists()


def test_init_does_not_trust_stale_local_workspace_id(
    monkeypatch, tmp_path: Path
) -> None:
    """init should ignore a locally cached workspace id that the server no longer returns.

    Regression guard: previously, a stale local workspace name would be
    trusted verbatim, producing broken Git Sync links and bypassing the
    connected-repo guardrail.
    """
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.cli.init as init_module

    class _Creds:
        def is_expired(self) -> bool:
            return False

    def fake_platform_request(endpoint, **kwargs):
        assert endpoint == "/api/cli/workspaces"
        # Server lists two *other* workspaces — the local id is gone.
        return {
            "workspaces": [
                {"id": "ws_a", "name": "team-a"},
                {"id": "ws_b", "name": "team-b"},
            ]
        }

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    for fn_name in (
        "_write_scaffold",
        "_git_init",
        "_git_initial_commit",
        "_update_workspace_metadata",
    ):
        monkeypatch.setattr(init_module, fn_name, lambda *a, **kw: None)
    # Local state points to a workspace the user no longer belongs to.
    monkeypatch.setattr(init_module, "get_active_workspace_name", lambda: "old-team")
    monkeypatch.setattr(init_module, "get_active_workspace_id", lambda: "ws_stale")
    monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
    monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)

    monkeypatch.chdir(tmp_path)

    # init should still succeed (empty Git context, no false connected-repo block).
    init_module.init(name="test-ws")

    context = init_module._selected_workspace_git_context()
    # Stale id wasn't trusted, and there's no unambiguous auto-selection.
    assert context["workspace_name"] is None
    assert context["git_sync_url"] is None
    assert context["connected_repo_url"] is None


# ── _render_template ─────────────────────────────────────────────


class TestRenderTemplate:
    def test_renders_variables(self) -> None:
        """_render_template substitutes {variable} placeholders."""
        from osmosis_ai.platform.cli.init import _render_template

        result = _render_template(
            "workspace.toml.tpl",
            {
                "sdk_version": "1.2.3",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        )
        assert 'sdk_version = "1.2.3"' in result
        assert 'created_at = "2026-01-01T00:00:00+00:00"' in result

    def test_static_template_unchanged(self) -> None:
        """Reading a static template with no variables returns it verbatim."""
        from osmosis_ai.platform.cli.init import _render_template

        result = _render_template("gitignore.tpl", {})
        assert "__pycache__/" in result
        assert ".venv/" in result


# ── _write_scaffold ───────────────────────────────────────────────


class TestWriteScaffold:
    def test_creates_full_directory_structure(self, tmp_path: Path) -> None:
        """_write_scaffold creates all expected directories, files, and .gitkeep markers."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "my-ws"
        target.mkdir()
        _write_scaffold(target, "my-ws")

        # Config files (rendered)
        assert (target / ".osmosis" / "workspace.toml").is_file()
        assert (target / "pyproject.toml").is_file()
        assert (target / ".gitignore").is_file()
        assert (target / "README.md").is_file()

        # .gitkeep markers
        assert (target / ".osmosis" / "research" / "experiments" / ".gitkeep").is_file()
        assert (target / "rollouts" / ".gitkeep").is_file()
        assert (target / "configs" / "eval" / ".gitkeep").is_file()
        assert (target / "data" / ".gitkeep").is_file()

        # Training config template (replaces configs/training/.gitkeep)
        assert (target / "configs" / "training" / "default.toml").is_file()
        assert (target / ".osmosis" / "research" / "program.md").is_file()

        # Static templates (formerly downloaded)
        assert (target / "AGENTS.md").is_file()
        assert (target / "CLAUDE.md").is_file()
        assert (target / "configs" / "AGENTS.md").is_file()

        # Claude Code plugin marketplace registration
        assert (target / ".claude" / "settings.json").is_file()

        # Skills now live in the `osmosis` plugin repo, not the workspace.
        assert not (target / ".osmosis" / "skills").exists()

        # Directories exist
        assert (target / ".osmosis").is_dir()
        assert (target / ".osmosis" / "research").is_dir()
        assert (target / ".osmosis" / "research" / "experiments").is_dir()
        assert (target / "rollouts").is_dir()
        assert (target / "configs" / "training").is_dir()
        assert (target / "configs" / "eval").is_dir()
        assert (target / "data").is_dir()

    def test_workspace_toml_contains_sdk_version_and_created_at(
        self, tmp_path: Path
    ) -> None:
        """workspace.toml contains sdk_version from PACKAGE_VERSION and a created_at timestamp."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        content = (target / ".osmosis" / "workspace.toml").read_text(encoding="utf-8")
        assert f'sdk_version = "{PACKAGE_VERSION}"' in content
        assert "created_at = " in content
        assert 'setup_source = "osmosis init"' in content

    def test_pyproject_toml_contains_workspace_name_dependency_and_python_requirement(
        self, tmp_path: Path
    ) -> None:
        """pyproject.toml contains the workspace name, dependency, and Python floor."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "cool-project"
        target.mkdir()
        _write_scaffold(target, "cool-project")

        content = (target / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "cool-project"' in content
        assert 'requires-python = ">=3.12"' in content
        assert f'"osmosis-ai>={PACKAGE_VERSION}"' in content

    def test_agents_md_respects_plugin_overrides(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """AGENTS.md renders plugin setup instructions from env overrides."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        monkeypatch.setenv("OSMOSIS_PLUGIN_REPO", "my-org/my-plugins")
        monkeypatch.setenv("OSMOSIS_PLUGIN_MARKETPLACE", "my-marketplace")

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        content = (target / "AGENTS.md").read_text(encoding="utf-8")
        assert "codex plugin marketplace add my-org/my-plugins" in content
        assert "codex plugin install my-marketplace" in content
        assert "codex plugin install osmosis" not in content

    def test_gitignore_has_python_sections(self, tmp_path: Path) -> None:
        """.gitignore includes Python-related patterns."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        content = (target / ".gitignore").read_text(encoding="utf-8")
        assert "__pycache__" in content
        assert ".venv" in content
        assert ".env" in content

    def test_readme_contains_workspace_name(self, tmp_path: Path) -> None:
        """README.md contains the workspace name."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "my-awesome-ws")

        content = (target / "README.md").read_text(encoding="utf-8")
        assert "my-awesome-ws" in content
        assert "osmosis workspace validate" in content
        assert ".osmosis/research/program.md" in content

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Running _write_scaffold twice does NOT overwrite pre-existing files."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()

        _write_scaffold(target, "ws")

        original_ws = "custom workspace content"
        (target / ".osmosis" / "workspace.toml").write_text(
            original_ws, encoding="utf-8"
        )
        original_pyproject = "custom pyproject content"
        (target / "pyproject.toml").write_text(original_pyproject, encoding="utf-8")
        original_gitignore = "custom gitignore"
        (target / ".gitignore").write_text(original_gitignore, encoding="utf-8")
        original_readme = "custom readme"
        (target / "README.md").write_text(original_readme, encoding="utf-8")
        original_gitkeep = "custom gitkeep"
        (target / "rollouts" / ".gitkeep").write_text(
            original_gitkeep, encoding="utf-8"
        )
        original_agents = "custom agents"
        (target / "AGENTS.md").write_text(original_agents, encoding="utf-8")

        _write_scaffold(target, "ws")

        assert (target / ".osmosis" / "workspace.toml").read_text(
            encoding="utf-8"
        ) == original_ws
        assert (target / "pyproject.toml").read_text(
            encoding="utf-8"
        ) == original_pyproject
        assert (target / ".gitignore").read_text(encoding="utf-8") == original_gitignore
        assert (target / "README.md").read_text(encoding="utf-8") == original_readme
        assert (target / "rollouts" / ".gitkeep").read_text(
            encoding="utf-8"
        ) == original_gitkeep
        assert (target / "AGENTS.md").read_text(encoding="utf-8") == original_agents


# ── _write_scaffold update mode ──────────────────────────────────


class TestWriteScaffoldUpdate:
    def test_update_overwrites_agents_and_plugin_settings(self, tmp_path: Path) -> None:
        """update=True overwrites AGENTS.md, CLAUDE.md, and .claude/settings.json."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        # Tamper with overwrite-on-update files
        (target / "AGENTS.md").write_text("custom agents", encoding="utf-8")
        (target / "CLAUDE.md").write_text("custom claude", encoding="utf-8")
        (target / "configs" / "AGENTS.md").write_text(
            "custom cfg agents", encoding="utf-8"
        )
        settings = target / ".claude" / "settings.json"
        settings.write_text("{}", encoding="utf-8")

        _write_scaffold(target, "ws", update=True)

        assert (target / "AGENTS.md").read_text(encoding="utf-8") != "custom agents"
        assert (target / "CLAUDE.md").read_text(encoding="utf-8") != "custom claude"
        assert (target / "configs" / "AGENTS.md").read_text(
            encoding="utf-8"
        ) != "custom cfg agents"
        assert settings.read_text(encoding="utf-8") != "{}"

    def test_update_preserves_configs(self, tmp_path: Path) -> None:
        """update=True does NOT overwrite pyproject.toml, .gitignore, README, or training config."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        custom = {
            "pyproject.toml": "custom pyproject",
            ".gitignore": "custom gitignore",
            "README.md": "custom readme",
        }
        for rel, content in custom.items():
            (target / rel).write_text(content, encoding="utf-8")
        (target / "configs" / "training" / "default.toml").write_text(
            "custom training", encoding="utf-8"
        )

        _write_scaffold(target, "ws", update=True)

        for rel, content in custom.items():
            assert (target / rel).read_text(encoding="utf-8") == content
        assert (target / "configs" / "training" / "default.toml").read_text(
            encoding="utf-8"
        ) == "custom training"

    def test_update_does_not_overwrite_workspace_toml(self, tmp_path: Path) -> None:
        """update=True leaves workspace.toml alone (handled by _update_workspace_metadata)."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        original = "custom workspace toml"
        (target / ".osmosis" / "workspace.toml").write_text(original, encoding="utf-8")

        _write_scaffold(target, "ws", update=True)

        assert (target / ".osmosis" / "workspace.toml").read_text(
            encoding="utf-8"
        ) == original


# ── Claude plugin marketplace settings ──────────────────────────


class TestClaudePluginSettings:
    def test_settings_json_uses_default_repo_and_marketplace(
        self, tmp_path: Path
    ) -> None:
        """Without env overrides, .claude/settings.json points at the default plugin repo."""
        import json

        from osmosis_ai.platform.cli.init import (
            _PLUGIN_MARKETPLACE_DEFAULT,
            _PLUGIN_REPO_DEFAULT,
            _write_scaffold,
        )

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        data = json.loads(
            (target / ".claude" / "settings.json").read_text(encoding="utf-8")
        )
        assert _PLUGIN_MARKETPLACE_DEFAULT in data["extraKnownMarketplaces"]
        assert (
            data["extraKnownMarketplaces"][_PLUGIN_MARKETPLACE_DEFAULT]["source"][
                "repo"
            ]
            == _PLUGIN_REPO_DEFAULT
        )
        assert data["enabledPlugins"][f"osmosis@{_PLUGIN_MARKETPLACE_DEFAULT}"] is True

    def test_settings_json_respects_env_overrides(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """OSMOSIS_PLUGIN_REPO / OSMOSIS_PLUGIN_MARKETPLACE override the defaults."""
        import json

        from osmosis_ai.platform.cli.init import _write_scaffold

        monkeypatch.setenv("OSMOSIS_PLUGIN_REPO", "my-org/my-plugins")
        monkeypatch.setenv("OSMOSIS_PLUGIN_MARKETPLACE", "my-marketplace")

        target = tmp_path / "ws"
        target.mkdir()
        _write_scaffold(target, "ws")

        data = json.loads(
            (target / ".claude" / "settings.json").read_text(encoding="utf-8")
        )
        assert "my-marketplace" in data["extraKnownMarketplaces"]
        assert (
            data["extraKnownMarketplaces"]["my-marketplace"]["source"]["repo"]
            == "my-org/my-plugins"
        )
        assert data["enabledPlugins"]["osmosis@my-marketplace"] is True


# ── _update_workspace_metadata ───────────────────────────────────


class TestUpdateWorkspaceMetadata:
    def test_preserves_created_at_and_adds_updated_at(self, tmp_path: Path) -> None:
        """_update_workspace_metadata keeps original created_at and adds updated_at."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _update_workspace_metadata

        ws_dir = tmp_path / ".osmosis"
        ws_dir.mkdir()
        (ws_dir / "workspace.toml").write_text(
            "[workspace]\n"
            'sdk_version = "0.0.1"\n'
            'created_at = "2025-01-01T00:00:00+00:00"\n'
            'setup_source = "osmosis init"\n',
            encoding="utf-8",
        )

        _update_workspace_metadata(tmp_path)

        content = (ws_dir / "workspace.toml").read_text(encoding="utf-8")
        assert 'created_at = "2025-01-01T00:00:00+00:00"' in content
        assert "updated_at = " in content
        assert f'sdk_version = "{PACKAGE_VERSION}"' in content

    def test_handles_missing_workspace_toml(self, tmp_path: Path) -> None:
        """_update_workspace_metadata works even if workspace.toml doesn't exist yet."""
        from osmosis_ai.platform.cli.init import _update_workspace_metadata

        ws_dir = tmp_path / ".osmosis"
        ws_dir.mkdir()

        _update_workspace_metadata(tmp_path)

        content = (ws_dir / "workspace.toml").read_text(encoding="utf-8")
        assert "created_at = " in content
        assert "updated_at = " in content


# ── _git_init ─────────────────────────────────────────────────────


class TestGitInit:
    def test_git_init_creates_repo(self, tmp_path: Path) -> None:
        """_git_init creates a new git repository in the target directory."""
        from osmosis_ai.platform.cli.init import _git_init

        _git_init(tmp_path)
        assert (tmp_path / ".git").is_dir()

    def test_git_init_skips_existing_repo(self, tmp_path: Path) -> None:
        """_git_init is a no-op when .git/ already exists."""
        from osmosis_ai.platform.cli.init import _git_init

        (tmp_path / ".git").mkdir()
        _git_init(tmp_path)  # Should not raise
        assert (tmp_path / ".git").is_dir()


# ── _git_initial_commit ───────────────────────────────────────────


class TestGitInitialCommit:
    def test_git_initial_commit(self, tmp_path: Path) -> None:
        """_git_initial_commit stages all files and creates the initial commit."""
        from osmosis_ai.platform.cli.init import _git_initial_commit

        subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        (tmp_path / "test.txt").write_text("hello")
        _git_initial_commit(tmp_path)
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Initial workspace setup" in result.stdout


# ── _print_next_steps ──────────────────────────────────────────────


class TestPrintNextSteps:
    def test_print_next_steps_default(self, monkeypatch) -> None:
        """_print_next_steps includes 'cd' and generic Git Sync CTA when no workspace is selected."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: None)
        monkeypatch.setattr(auth_module, "load_credentials", lambda: None)
        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert "cd my-workspace" in output
        assert "Git Sync" in output
        assert "integrations/git" not in output

    def test_print_next_steps_selected_workspace(self, monkeypatch) -> None:
        """_print_next_steps shows a workspace Git Sync URL when no repo is connected."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: "team-alpha")
        monkeypatch.setattr(auth_module, "load_credentials", lambda: None)
        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert "cd my-workspace" in output
        assert f"{mod.PLATFORM_URL}/team-alpha/integrations/git" in output

    def test_print_next_steps_connected_repo(self, monkeypatch) -> None:
        """_print_next_steps shows the connected repository URL when one exists."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        class _Creds:
            def is_expired(self) -> bool:
                return False

        def fake_platform_request(endpoint, **kwargs):
            assert endpoint == "/api/cli/workspaces"
            assert kwargs.get("cleanup_on_401") is False
            return {
                "workspaces": [
                    {
                        "id": "ws_alpha",
                        "name": "team-alpha",
                        "has_github_app_installation": True,
                        "connected_repo": {
                            "repo_url": "https://github.com/acme/rollouts",
                        },
                    }
                ]
            }

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: "team-alpha")
        monkeypatch.setattr(mod, "get_active_workspace_id", lambda: "ws_alpha")
        monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
        monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)
        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert "cd my-workspace" in output
        assert "Connected repo:" in output
        assert "https://github.com/acme/rollouts" in output

    def test_print_next_steps_choose_repo_when_app_installed(self, monkeypatch) -> None:
        """_print_next_steps asks the user to choose a repo when GitHub is connected but no repo is linked."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        class _Creds:
            def is_expired(self) -> bool:
                return False

        def fake_platform_request(endpoint, **kwargs):
            assert endpoint == "/api/cli/workspaces"
            return {
                "workspaces": [
                    {
                        "id": "ws_alpha",
                        "name": "team-alpha",
                        "has_github_app_installation": True,
                        "connected_repo": None,
                    }
                ]
            }

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: "team-alpha")
        monkeypatch.setattr(mod, "get_active_workspace_id", lambda: "ws_alpha")
        monkeypatch.setattr(auth_module, "load_credentials", lambda: _Creds())
        monkeypatch.setattr(auth_module, "platform_request", fake_platform_request)
        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert f"{mod.PLATFORM_URL}/team-alpha/integrations/git" in output
        assert "choose a repo" in output

    def test_print_next_steps_includes_plugin_install_hints(self, monkeypatch) -> None:
        """_print_next_steps advertises the osmosis plugin for Claude / Cursor / Codex."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        monkeypatch.setenv("OSMOSIS_PLUGIN_REPO", "my-org/my-plugins")
        monkeypatch.setenv("OSMOSIS_PLUGIN_MARKETPLACE", "my-marketplace")

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: None)
        monkeypatch.setattr(auth_module, "load_credentials", lambda: None)

        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert "osmosis agent plugin" in output
        assert "Claude Code" in output
        assert "Cursor" in output
        assert "Codex" in output
        assert "my-org/my-plugins" in output
        assert "codex plugin install my-marketplace" in output
        assert "codex plugin install osmosis" not in output

    def test_print_next_steps_here(self, monkeypatch) -> None:
        """_print_next_steps omits 'cd' when here=True but keeps platform URL."""
        import io

        import osmosis_ai.platform.auth as auth_module
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        monkeypatch.setattr(mod, "get_active_workspace_name", lambda: None)
        monkeypatch.setattr(auth_module, "load_credentials", lambda: None)
        _print_next_steps("my-workspace", here=True)
        output = buf.getvalue()
        assert "cd my-workspace" not in output
        assert "Git Sync" in output


# ── Integration tests — full init flow ────────────────────────────


def test_full_init_flow(monkeypatch, tmp_path: Path) -> None:
    """Full init path: scaffold from templates + git init + commit."""
    import io

    import osmosis_ai.platform.cli.init as init_module
    from osmosis_ai.cli.console import Console

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, no_color=True)
    monkeypatch.setattr(init_module, "console", test_console)

    monkeypatch.chdir(tmp_path)

    init_module.init(name="test-ws")

    target = tmp_path / "test-ws"
    assert target.is_dir()

    assert (target / ".osmosis" / "workspace.toml").is_file()
    assert (target / ".osmosis" / "research" / "program.md").is_file()
    assert (target / ".osmosis" / "research" / "experiments" / ".gitkeep").is_file()
    assert (target / "pyproject.toml").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "README.md").is_file()

    assert (target / "AGENTS.md").is_file()
    assert (target / "CLAUDE.md").is_file()

    assert (target / ".git").is_dir()

    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Initial workspace setup" in result.stdout


def test_full_init_update_flow(monkeypatch, tmp_path: Path) -> None:
    """Update mode: overwrites agent docs/plugin settings, updates metadata, does NOT commit."""
    import io

    import osmosis_ai.platform.cli.init as init_module
    from osmosis_ai.cli.console import Console

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, no_color=True)
    monkeypatch.setattr(init_module, "console", test_console)

    monkeypatch.chdir(tmp_path)

    # First run — fresh init
    init_module.init(name="test-ws")
    target = tmp_path / "test-ws"

    # Record initial state
    commit_count_before = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Tamper with AGENTS.md and a config
    (target / "AGENTS.md").write_text("user edits", encoding="utf-8")
    (target / "pyproject.toml").write_text("user pyproject", encoding="utf-8")

    # Second run — update mode
    init_module.init(name="test-ws")

    # AGENTS.md was overwritten
    assert (target / "AGENTS.md").read_text(encoding="utf-8") != "user edits"
    # pyproject.toml was preserved
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == "user pyproject"
    # workspace.toml has updated_at
    ws_content = (target / ".osmosis" / "workspace.toml").read_text(encoding="utf-8")
    assert "updated_at = " in ws_content

    # No new git commit was created
    commit_count_after = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit_count_before == commit_count_after


# ── CLI command registration ──────────────────────────────────────


def test_init_command_registered() -> None:
    """The 'init' command is registered as a top-level command."""
    from typer.testing import CliRunner

    from osmosis_ai.cli.main import _register_commands, app
    from tests.unit.platform.cli.conftest import strip_ansi

    _register_commands()
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "Initialize" in output
    assert "--here" in output
    assert "NAME" in output

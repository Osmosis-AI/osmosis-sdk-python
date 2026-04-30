"""Tests for project init (osmosis init <name>) core flow and error handling."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError

# ── Shared fixtures ──────────────────────────────────────────────


class _FakeCreds:
    """Minimal credentials stand-in for the auth fixture below."""

    def is_expired(self) -> bool:
        return False


_DEFAULT_WORKSPACE_NAME = "default-workspace"
_DEFAULT_WORKSPACE_ID = "ws_default"


def _default_git_context() -> dict[str, str | bool | None]:
    """Return a workspace context with no connected repo and no GitHub App."""
    return {
        "workspace_id": _DEFAULT_WORKSPACE_ID,
        "workspace_name": _DEFAULT_WORKSPACE_NAME,
        "git_sync_url": (
            f"https://platform.osmosis.ai/{_DEFAULT_WORKSPACE_NAME}/integrations/git"
        ),
        "has_github_app_installation": False,
        "connected_repo_url": None,
    }


@pytest.fixture(autouse=True)
def _mock_init_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default `osmosis init` to a healthy auth + workspace context.

    `init` now requires an active platform workspace upfront via
    ``_require_auth``. Most tests in this module exercise local
    behavior (scaffolding, directory checks, post-init CTA), so this
    autouse fixture stubs the auth + Git Sync lookup to a workspace
    with no connected repo. Individual tests override these patches to
    cover auth/workspace failure modes and the connected-repo guard.

    The autouse stubs leave the real ``_resolve_workspace_git_context``
    in place so its graceful-degradation behavior is exercised; tests
    that care about specific Git Sync metadata can either patch
    ``_resolve_workspace_git_context`` directly or replace
    ``OsmosisClient.refresh_workspace_info`` to drive the underlying
    API call.
    """
    import osmosis_ai.platform.api.client as client_module
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: (_DEFAULT_WORKSPACE_NAME, _FakeCreds()),
    )
    monkeypatch.setattr(
        init_module, "get_active_workspace_id", lambda: _DEFAULT_WORKSPACE_ID
    )

    def _default_refresh(self: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "found": True,
            "has_subscription": True,
            "has_github_app_installation": False,
            "connected_repo": None,
        }

    monkeypatch.setattr(
        client_module.OsmosisClient,
        "refresh_workspace_info",
        _default_refresh,
    )


# ── Error case: git not found ────────────────────────────────────


def test_init_raises_when_git_not_found(monkeypatch) -> None:
    """init raises CLIError when git is not on PATH."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: None)

    with pytest.raises(CLIError, match=r"[Gg]it"):
        init_module.init(name="test-project")


# ── Error case: no active workspace ──────────────────────────────


def test_init_raises_when_no_workspace_selected(monkeypatch, tmp_path: Path) -> None:
    """init refuses to scaffold when ``_require_auth`` has no active workspace."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    def _no_workspace() -> tuple[str, Any]:
        raise CLIError(
            "No workspace selected. Run 'osmosis workspace' to select a workspace."
        )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        _no_workspace,
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="No workspace selected"):
        init_module.init(name="test-project")

    assert not (tmp_path / "test-project").exists()


def test_init_raises_when_not_logged_in(monkeypatch, tmp_path: Path) -> None:
    """init refuses to scaffold when credentials are missing/expired."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    def _not_logged_in() -> tuple[str, Any]:
        raise CLIError("Not logged in. Run 'osmosis auth login' first.")

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        _not_logged_in,
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="Not logged in"):
        init_module.init(name="test-project")

    assert not (tmp_path / "test-project").exists()


# ── Error case: directory exists without project.toml ─────────────────


def test_init_raises_when_directory_exists_without_project_toml(
    monkeypatch, tmp_path: Path
) -> None:
    """init raises CLIError if target dir exists but has no .osmosis/project.toml."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    target = tmp_path / "my-project"
    target.mkdir()

    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="already exists"):
        init_module.init(name="my-project")


# ── Re-entry: directory exists with .osmosis/project.toml ─────────────


def _stub_scaffold_fns(monkeypatch, module) -> None:
    """Stub all scaffold/git/print functions to be no-ops."""
    for fn_name in (
        "_write_scaffold",
        "_git_init",
        "_git_initial_commit",
        "_update_project_metadata",
        "_print_next_steps",
    ):
        monkeypatch.setattr(module, fn_name, lambda *a, **kw: None)


def test_init_rerun_on_existing_project(monkeypatch, tmp_path: Path) -> None:
    """init enters update mode when target dir has .osmosis/project.toml."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    _stub_scaffold_fns(monkeypatch, init_module)

    target = tmp_path / "my-project"
    target.mkdir()
    osmosis_dir = target / ".osmosis"
    osmosis_dir.mkdir()
    (osmosis_dir / "project.toml").write_text('[project]\nname = "my-project"\n')

    monkeypatch.chdir(tmp_path)

    # Should NOT raise — enters update mode
    init_module.init(name="my-project")


# ── Cleanup on scaffold failure ──────────────────────────────────


def test_init_cleanup_on_scaffold_failure(monkeypatch, tmp_path: Path) -> None:
    """If scaffold fails, the created directory is cleaned up."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")

    def _fail_scaffold(*a, **kw):
        raise CLIError("scaffold failed")

    monkeypatch.setattr(init_module, "_write_scaffold", _fail_scaffold)

    monkeypatch.chdir(tmp_path)

    target = tmp_path / "fail-project"
    assert not target.exists()

    with pytest.raises(CLIError, match="scaffold failed"):
        init_module.init(name="fail-project")

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
        init_module.init(name="test-project", here=True)


# ── --here: allows directory with only .git/ ──────────────────────


def test_init_here_allows_git_dir(monkeypatch, tmp_path: Path) -> None:
    """init(here=True) allows a directory containing only .git/."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    _stub_scaffold_fns(monkeypatch, init_module)

    (tmp_path / ".git").mkdir()

    monkeypatch.chdir(tmp_path)

    # Should NOT raise
    init_module.init(name="test-project", here=True)


def test_init_raises_when_selected_workspace_has_connected_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """Fresh init should stop before creating a directory when the active workspace already has a repo."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(
        init_module,
        "_resolve_workspace_git_context",
        lambda **_kwargs: {
            "workspace_id": "ws_alpha",
            "workspace_name": "team-alpha",
            "git_sync_url": "https://platform.osmosis.ai/team-alpha/integrations/git",
            "has_github_app_installation": True,
            "connected_repo_url": "https://github.com/acme/rollouts",
        },
    )

    monkeypatch.chdir(tmp_path)

    target = tmp_path / "test-project"
    with pytest.raises(CLIError, match="already connected"):
        init_module.init(name="test-project")

    assert not target.exists()


def test_init_here_raises_when_selected_workspace_has_connected_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """init(here=True) should stop when the active workspace already has a connected repo."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(
        init_module,
        "_resolve_workspace_git_context",
        lambda **_kwargs: {
            "workspace_id": "ws_alpha",
            "workspace_name": "team-alpha",
            "git_sync_url": "https://platform.osmosis.ai/team-alpha/integrations/git",
            "has_github_app_installation": True,
            "connected_repo_url": "https://github.com/acme/rollouts",
        },
    )

    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match=r"git clone https://github\.com/acme/rollouts"):
        init_module.init(name="test-project", here=True)


def test_init_tolerates_auth_expiry_in_workspace_lookup(
    monkeypatch, tmp_path: Path
) -> None:
    """A 401 during the post-auth Git Sync refresh shouldn't block scaffolding.

    ``_require_auth`` already validated the session, so a transient
    auth/platform error during the connected-repo metadata refresh
    degrades gracefully to a context with no connected repo (never
    blocks init on a non-existent connected repo).
    """
    import osmosis_ai.platform.api.client as client_module
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    _stub_scaffold_fns(monkeypatch, init_module)

    def _fake_refresh(self: Any, **_kwargs: Any) -> dict[str, Any]:
        raise auth_module.AuthenticationExpiredError("session expired")

    monkeypatch.setattr(
        client_module.OsmosisClient, "refresh_workspace_info", _fake_refresh
    )

    monkeypatch.chdir(tmp_path)

    init_module.init(name="test-project")

    assert (tmp_path / "test-project").is_dir()


# ── _render_template ─────────────────────────────────────────────


class TestRenderTemplate:
    def test_renders_variables(self) -> None:
        """_render_template substitutes {variable} placeholders."""
        from osmosis_ai.platform.cli.init import _render_template

        result = _render_template(
            "project.toml.tpl",
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

        target = tmp_path / "my-project"
        target.mkdir()
        _write_scaffold(target, "my-project")

        # Config files (rendered)
        assert (target / ".osmosis" / "project.toml").is_file()
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

        # Skills now live in the `osmosis` plugin repo, not the project.
        assert not (target / ".osmosis" / "skills").exists()

        # Directories exist
        assert (target / ".osmosis").is_dir()
        assert (target / ".osmosis" / "research").is_dir()
        assert (target / ".osmosis" / "research" / "experiments").is_dir()
        assert (target / "rollouts").is_dir()
        assert (target / "configs" / "training").is_dir()
        assert (target / "configs" / "eval").is_dir()
        assert (target / "data").is_dir()

    def test_project_toml_contains_sdk_version_and_created_at(
        self, tmp_path: Path
    ) -> None:
        """project.toml contains sdk_version from PACKAGE_VERSION and a created_at timestamp."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        content = (target / ".osmosis" / "project.toml").read_text(encoding="utf-8")
        assert f'sdk_version = "{PACKAGE_VERSION}"' in content
        assert "created_at = " in content
        assert 'setup_source = "osmosis init"' in content

    def test_pyproject_toml_contains_project_name_dependency_and_python_requirement(
        self, tmp_path: Path
    ) -> None:
        """pyproject.toml contains the project name, dependency, and Python floor."""
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

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        content = (target / "AGENTS.md").read_text(encoding="utf-8")
        assert "codex plugin marketplace add my-org/my-plugins" in content
        assert "codex plugin install my-marketplace" in content
        assert "codex plugin install osmosis" not in content

    def test_gitignore_has_python_sections(self, tmp_path: Path) -> None:
        """.gitignore includes Python-related patterns."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        content = (target / ".gitignore").read_text(encoding="utf-8")
        assert "__pycache__" in content
        assert ".venv" in content
        assert ".env" in content

    def test_readme_contains_project_name(self, tmp_path: Path) -> None:
        """README.md contains the project name."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "my-awesome-project")

        content = (target / "README.md").read_text(encoding="utf-8")
        assert "my-awesome-project" in content
        assert "osmosis --json project validate" in content
        assert ".osmosis/research/program.md" in content

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Running _write_scaffold twice does NOT overwrite pre-existing files."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()

        _write_scaffold(target, "project")

        original_proj = "custom project content"
        (target / ".osmosis" / "project.toml").write_text(
            original_proj, encoding="utf-8"
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

        _write_scaffold(target, "project")

        assert (target / ".osmosis" / "project.toml").read_text(
            encoding="utf-8"
        ) == original_proj
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

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        # Tamper with overwrite-on-update files
        (target / "AGENTS.md").write_text("custom agents", encoding="utf-8")
        (target / "CLAUDE.md").write_text("custom claude", encoding="utf-8")
        (target / "configs" / "AGENTS.md").write_text(
            "custom cfg agents", encoding="utf-8"
        )
        settings = target / ".claude" / "settings.json"
        settings.write_text("{}", encoding="utf-8")

        _write_scaffold(target, "project", update=True)

        assert (target / "AGENTS.md").read_text(encoding="utf-8") != "custom agents"
        assert (target / "CLAUDE.md").read_text(encoding="utf-8") != "custom claude"
        assert (target / "configs" / "AGENTS.md").read_text(
            encoding="utf-8"
        ) != "custom cfg agents"
        assert settings.read_text(encoding="utf-8") != "{}"

    def test_update_preserves_configs(self, tmp_path: Path) -> None:
        """update=True does NOT overwrite pyproject.toml, .gitignore, README, or training config."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

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

        _write_scaffold(target, "project", update=True)

        for rel, content in custom.items():
            assert (target / rel).read_text(encoding="utf-8") == content
        assert (target / "configs" / "training" / "default.toml").read_text(
            encoding="utf-8"
        ) == "custom training"

    def test_update_does_not_overwrite_project_toml(self, tmp_path: Path) -> None:
        """update=True leaves project.toml alone (handled by _update_project_metadata)."""
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        original = "custom project toml"
        (target / ".osmosis" / "project.toml").write_text(original, encoding="utf-8")

        _write_scaffold(target, "project", update=True)

        assert (target / ".osmosis" / "project.toml").read_text(
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

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

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

        target = tmp_path / "project"
        target.mkdir()
        _write_scaffold(target, "project")

        data = json.loads(
            (target / ".claude" / "settings.json").read_text(encoding="utf-8")
        )
        assert "my-marketplace" in data["extraKnownMarketplaces"]
        assert (
            data["extraKnownMarketplaces"]["my-marketplace"]["source"]["repo"]
            == "my-org/my-plugins"
        )
        assert data["enabledPlugins"]["osmosis@my-marketplace"] is True


# ── _update_project_metadata ───────────────────────────────────


class TestUpdateProjectMetadata:
    def test_preserves_created_at_and_adds_updated_at(self, tmp_path: Path) -> None:
        """_update_project_metadata keeps original created_at and adds updated_at."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _update_project_metadata

        osmosis_dir = tmp_path / ".osmosis"
        osmosis_dir.mkdir()
        (osmosis_dir / "project.toml").write_text(
            "[project]\n"
            'sdk_version = "0.0.1"\n'
            'created_at = "2025-01-01T00:00:00+00:00"\n'
            'setup_source = "osmosis init"\n',
            encoding="utf-8",
        )

        _update_project_metadata(tmp_path)

        content = (osmosis_dir / "project.toml").read_text(encoding="utf-8")
        assert 'created_at = "2025-01-01T00:00:00+00:00"' in content
        assert "updated_at = " in content
        assert f'sdk_version = "{PACKAGE_VERSION}"' in content

    def test_handles_missing_project_toml(self, tmp_path: Path) -> None:
        """_update_project_metadata works even if project.toml doesn't exist yet."""
        from osmosis_ai.platform.cli.init import _update_project_metadata

        osmosis_dir = tmp_path / ".osmosis"
        osmosis_dir.mkdir()

        _update_project_metadata(tmp_path)

        content = (osmosis_dir / "project.toml").read_text(encoding="utf-8")
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
        assert "Initial project setup" in result.stdout


# ── _print_next_steps ──────────────────────────────────────────────


class TestPrintNextSteps:
    @staticmethod
    def _build_console(monkeypatch):
        """Swap the init module's console for a captured StringIO console."""
        import io

        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        return buf

    def test_print_next_steps_empty_context(self, monkeypatch) -> None:
        """_print_next_steps falls back to the generic Git Sync CTA with an empty context."""
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = self._build_console(monkeypatch)
        empty_context: dict[str, str | bool | None] = {
            "workspace_id": None,
            "workspace_name": None,
            "git_sync_url": None,
            "has_github_app_installation": False,
            "connected_repo_url": None,
        }
        _print_next_steps("my-project", here=False, git_context=empty_context)
        output = buf.getvalue()
        assert "cd my-project" in output
        assert "Git Sync" in output
        assert "integrations/git" not in output

    def test_print_next_steps_selected_workspace(self, monkeypatch) -> None:
        """_print_next_steps shows a workspace Git Sync URL when no repo is connected."""
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = self._build_console(monkeypatch)
        _print_next_steps(
            "my-project",
            here=False,
            git_context={
                "workspace_id": "ws_alpha",
                "workspace_name": "team-alpha",
                "git_sync_url": (f"{mod.PLATFORM_URL}/team-alpha/integrations/git"),
                "has_github_app_installation": False,
                "connected_repo_url": None,
            },
        )
        output = buf.getvalue()
        assert "cd my-project" in output
        assert f"{mod.PLATFORM_URL}/team-alpha/integrations/git" in output

    def test_print_next_steps_connected_repo(self, monkeypatch) -> None:
        """_print_next_steps shows the connected repository URL when one exists."""
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = self._build_console(monkeypatch)
        _print_next_steps(
            "my-project",
            here=False,
            git_context={
                "workspace_id": "ws_alpha",
                "workspace_name": "team-alpha",
                "git_sync_url": (f"{mod.PLATFORM_URL}/team-alpha/integrations/git"),
                "has_github_app_installation": True,
                "connected_repo_url": "https://github.com/acme/rollouts",
            },
        )
        output = buf.getvalue()
        assert "cd my-project" in output
        assert "Connected repo:" in output
        assert "https://github.com/acme/rollouts" in output

    def test_print_next_steps_choose_repo_when_app_installed(self, monkeypatch) -> None:
        """_print_next_steps asks the user to choose a repo when GitHub is connected but no repo is linked."""
        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = self._build_console(monkeypatch)
        _print_next_steps(
            "my-project",
            here=False,
            git_context={
                "workspace_id": "ws_alpha",
                "workspace_name": "team-alpha",
                "git_sync_url": (f"{mod.PLATFORM_URL}/team-alpha/integrations/git"),
                "has_github_app_installation": True,
                "connected_repo_url": None,
            },
        )
        output = buf.getvalue()
        assert f"{mod.PLATFORM_URL}/team-alpha/integrations/git" in output
        assert "choose a repo" in output

    def test_print_next_steps_includes_plugin_install_hints(self, monkeypatch) -> None:
        """_print_next_steps advertises the osmosis plugin for Claude / Cursor / Codex."""
        from osmosis_ai.platform.cli.init import _print_next_steps

        monkeypatch.setenv("OSMOSIS_PLUGIN_REPO", "my-org/my-plugins")
        monkeypatch.setenv("OSMOSIS_PLUGIN_MARKETPLACE", "my-marketplace")

        buf = self._build_console(monkeypatch)
        _print_next_steps(
            "my-project",
            here=False,
            git_context=_default_git_context(),
        )
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
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = self._build_console(monkeypatch)
        _print_next_steps(
            "my-project",
            here=True,
            git_context=_default_git_context(),
        )
        output = buf.getvalue()
        assert "cd my-project" not in output
        assert "Git Sync" in output or "integrations/git" in output


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

    init_module.init(name="test-project")

    target = tmp_path / "test-project"
    assert target.is_dir()

    assert (target / ".osmosis" / "project.toml").is_file()
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
    assert "Initial project setup" in result.stdout


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
    init_module.init(name="test-project")
    target = tmp_path / "test-project"

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
    init_module.init(name="test-project")

    # AGENTS.md was overwritten
    assert (target / "AGENTS.md").read_text(encoding="utf-8") != "user edits"
    # pyproject.toml was preserved
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == "user pyproject"
    # project.toml has updated_at
    project_content = (target / ".osmosis" / "project.toml").read_text(encoding="utf-8")
    assert "updated_at = " in project_content

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

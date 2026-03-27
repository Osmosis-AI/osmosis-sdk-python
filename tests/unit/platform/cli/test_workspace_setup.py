"""Tests for workspace init (osmosis init <name>) core flow and error handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError

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
        assert (target / "rollouts" / ".gitkeep").is_file()
        assert (target / "configs" / "eval" / ".gitkeep").is_file()
        assert (target / "data" / ".gitkeep").is_file()

        # Training config template (replaces configs/training/.gitkeep)
        assert (target / "configs" / "training" / "default.toml").is_file()

        # Static templates (formerly downloaded)
        assert (target / "AGENTS.md").is_file()
        assert (target / "CLAUDE.md").is_file()
        assert (target / "configs" / "AGENTS.md").is_file()
        assert (
            target / ".osmosis" / "skills" / "create-rollout" / "SKILL.md"
        ).is_file()
        assert (
            target / ".osmosis" / "skills" / "evaluate-rollout" / "SKILL.md"
        ).is_file()
        assert (
            target / ".osmosis" / "skills" / "submit-training" / "SKILL.md"
        ).is_file()

        # Directories exist
        assert (target / ".osmosis").is_dir()
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

    def test_pyproject_toml_contains_workspace_name_and_dependency(
        self, tmp_path: Path
    ) -> None:
        """pyproject.toml contains the workspace name and osmosis-ai dependency."""
        from osmosis_ai.consts import PACKAGE_VERSION
        from osmosis_ai.platform.cli.init import _write_scaffold

        target = tmp_path / "cool-project"
        target.mkdir()
        _write_scaffold(target, "cool-project")

        content = (target / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "cool-project"' in content
        assert f'"osmosis-ai>={PACKAGE_VERSION}"' in content

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

    def test_git_initial_commit_update_no_changes(self, tmp_path: Path) -> None:
        """update=True skips the commit when there are no staged changes."""
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
        # Create an initial commit so the repo is not empty.
        subprocess.run(
            ["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "seed"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )

        # Now call with update=True — nothing has changed, should NOT raise.
        _git_initial_commit(tmp_path, update=True)

        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        # Only the seed commit should exist; no extra commit was created.
        assert result.stdout.strip().count("\n") == 0
        assert "seed" in result.stdout

    def test_git_initial_commit_update_with_changes(self, tmp_path: Path) -> None:
        """update=True creates a commit when there are staged changes."""
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
        subprocess.run(
            ["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "seed"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )

        # Add a new file, then call with update=True.
        (tmp_path / "new.txt").write_text("new")
        _git_initial_commit(tmp_path, update=True)

        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Update workspace scaffold" in result.stdout


# ── _print_next_steps ──────────────────────────────────────────────


class TestPrintNextSteps:
    def test_print_next_steps_default(self, monkeypatch) -> None:
        """_print_next_steps includes 'cd' and platform URL when here=False."""
        import io

        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
        _print_next_steps("my-workspace", here=False)
        output = buf.getvalue()
        assert "cd my-workspace" in output
        assert "Git Sync" in output

    def test_print_next_steps_here(self, monkeypatch) -> None:
        """_print_next_steps omits 'cd' when here=True but keeps platform URL."""
        import io

        import osmosis_ai.platform.cli.init as mod
        from osmosis_ai.cli.console import Console
        from osmosis_ai.platform.cli.init import _print_next_steps

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)
        monkeypatch.setattr(mod, "console", test_console)
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


# ── CLI command registration ──────────────────────────────────────


def test_init_command_registered() -> None:
    """The 'init' command is registered as a top-level command."""
    from typer.testing import CliRunner

    from osmosis_ai.cli.main import _register_commands, app

    _register_commands()
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "Initialize" in result.output
    assert "--here" in result.output
    assert "NAME" in result.output

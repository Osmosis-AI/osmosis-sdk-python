"""Project init — scaffold a new local Osmosis project directory.

This module implements the core flow for ``osmosis init <name>``:
check prerequisites, create the directory, scaffold files from
bundled templates, initialise git, and print next steps.

A "project" is the local on-disk directory created by this command —
distinct from a platform workspace (the remote tenant managed via
the platform and linked to a local project with ``osmosis project link``).
"""

from __future__ import annotations

import os
import shutil
import subprocess as _subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.cli.constants import validate_name
from osmosis_ai.platform.cli.project_mapping import (
    CONFIG_FILE,
    ProjectLinkRecord,
    ProjectMappingStore,
    now_linked_at,
)

# ── Prerequisites ────────────────────────────────────────────────


def _check_git_installed() -> None:
    """Verify that ``git`` is available on PATH."""
    if shutil.which("git") is None:
        raise CLIError(
            "Git is not installed or not on PATH.\n"
            "  Install it from https://git-scm.com/ and try again."
        )


# ── Plugin marketplace configuration ────────────────────────────
#
# `osmosis init` scaffolds a `.claude/settings.json` that points Claude Code
# at the Osmosis plugin marketplace. The plugin hosts the agent skills
# (`plan-training`, `create-rollouts`, `evaluate-rollouts`, etc.), so updates ship
# via a `git push` to the plugins repo rather than a new SDK release.
#
# The env vars let us point scaffolded projects at a different repo or
# marketplace (e.g. a staging mirror) without shipping an SDK release.

_PLUGIN_REPO_DEFAULT = "Osmosis-AI/osmosis-plugins"
_PLUGIN_MARKETPLACE_DEFAULT = "osmosis"


def _plugin_repo() -> str:
    """GitHub repo (`owner/name`) hosting the Osmosis plugin marketplace."""
    return os.environ.get("OSMOSIS_PLUGIN_REPO") or _PLUGIN_REPO_DEFAULT


def _plugin_marketplace() -> str:
    """Marketplace name as declared in the plugin repo's `marketplace.json`."""
    return os.environ.get("OSMOSIS_PLUGIN_MARKETPLACE") or _PLUGIN_MARKETPLACE_DEFAULT


# ── Scaffold manifest ───────────────────────────────────────────

_TEMPLATES = files("osmosis_ai.platform.cli") / "templates"


@dataclass(frozen=True, slots=True)
class ScaffoldEntry:
    """A single file to create during project scaffolding.

    Attributes:
        template: Relative path inside the templates package directory.
            Empty string means write an empty file (used for ``.gitkeep``).
        dest: Relative path inside the target project directory.
        render: Whether to apply ``str.format_map()`` variable substitution.
        overwrite_on_update: If True, overwrite existing file during update mode.
    """

    template: str
    dest: str
    render: bool = False
    overwrite_on_update: bool = False


SCAFFOLD: list[ScaffoldEntry] = [
    # Directory placeholders
    ScaffoldEntry("", ".osmosis/cache/.gitkeep"),
    ScaffoldEntry("", "rollouts/.gitkeep"),
    ScaffoldEntry("", "configs/eval/.gitkeep"),
    ScaffoldEntry("", "configs/training/.gitkeep"),
    ScaffoldEntry("", "data/.gitkeep"),
    # Rendered templates (project.toml handled separately on update)
    ScaffoldEntry("project.toml.tpl", ".osmosis/project.toml", render=True),
    ScaffoldEntry("program.md.tpl", ".osmosis/program.md"),
    ScaffoldEntry("pyproject.toml.tpl", "pyproject.toml", render=True),
    ScaffoldEntry("README.md.tpl", "README.md", render=True),
    # Static files — configs (skip on update)
    ScaffoldEntry("gitignore.tpl", ".gitignore"),
    ScaffoldEntry("configs/training/default.toml.tpl", "configs/training/default.toml"),
    # Agent docs (overwrite on update)
    ScaffoldEntry("AGENTS.md.tpl", "AGENTS.md", render=True, overwrite_on_update=True),
    ScaffoldEntry("CLAUDE.md.tpl", "CLAUDE.md", overwrite_on_update=True),
    ScaffoldEntry(
        "configs/AGENTS.md.tpl", "configs/AGENTS.md", overwrite_on_update=True
    ),
    # Plugin marketplace registration for Claude Code (committed so the
    # team shares a single plugin source). Always refreshed on update so
    # SDK upgrades can bump the marketplace URL in-place.
    ScaffoldEntry(
        "claude/settings.json.tpl",
        ".claude/settings.json",
        render=True,
        overwrite_on_update=True,
    ),
]

AGENT_REFRESH_PATHS = {
    "AGENTS.md",
    "CLAUDE.md",
    "configs/AGENTS.md",
    ".claude/settings.json",
}


# ── Scaffold generation ─────────────────────────────────────────


def _render_template(name: str, variables: dict[str, str]) -> str:
    """Read a template file and apply variable substitution."""
    text = (_TEMPLATES / name).read_text(encoding="utf-8")
    return text.format_map(variables)


def _write_scaffold(target: Path, project_name: str, *, update: bool = False) -> None:
    """Write all scaffold files into *target* from the SCAFFOLD manifest.

    In normal mode, files that already exist are skipped (idempotent).
    In *update* mode, entries with ``overwrite_on_update=True`` are
    overwritten; all other existing files are left untouched.
    """
    import datetime

    from osmosis_ai.consts import PACKAGE_VERSION

    variables = {
        "name": project_name,
        "sdk_version": PACKAGE_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "plugin_repo": _plugin_repo(),
        "plugin_marketplace": _plugin_marketplace(),
    }

    for entry in SCAFFOLD:
        dest = target / entry.dest
        if dest.exists() and not (update and entry.overwrite_on_update):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not entry.template:
            dest.write_text("", encoding="utf-8")
        elif entry.render:
            dest.write_text(
                _render_template(entry.template, variables), encoding="utf-8"
            )
        else:
            dest.write_bytes((_TEMPLATES / entry.template).read_bytes())


# ── Project metadata update ─────────────────────────────────────

_CREATED_AT_RE = r'created_at\s*=\s*"([^"]*)"'


def _update_project_metadata(target: Path) -> None:
    """Update project.toml: refresh ``sdk_version``, add ``updated_at``, preserve ``created_at``."""
    import datetime
    import re

    from osmosis_ai.consts import PACKAGE_VERSION

    project_toml = target / ".osmosis" / "project.toml"
    original = (
        project_toml.read_text(encoding="utf-8") if project_toml.is_file() else ""
    )

    match = re.search(_CREATED_AT_RE, original)
    created_at = (
        match.group(1) if match else datetime.datetime.now(datetime.UTC).isoformat()
    )
    updated_at = datetime.datetime.now(datetime.UTC).isoformat()

    project_toml.write_text(
        "# Osmosis Project Configuration\n"
        "# Generated by `osmosis init`\n"
        "\n"
        "[project]\n"
        f'sdk_version = "{PACKAGE_VERSION}"\n'
        f'created_at = "{created_at}"\n'
        f'updated_at = "{updated_at}"\n'
        'setup_source = "osmosis init"\n',
        encoding="utf-8",
    )


# ── Git helpers ──────────────────────────────────────────────────


def _git_init(target: Path) -> None:
    """Initialise a git repository in the target directory.

    Skips if ``.git`` already exists as either a directory or a file. Git
    worktrees and submodules use a file that points at the real gitdir.
    """
    if (target / ".git").exists():
        return

    _subprocess.run(
        ["git", "init", "-b", "main", str(target)],
        capture_output=True,
        check=True,
    )


def _git_initial_commit(target: Path) -> None:
    """Stage all files and create the initial git commit.

    Only called for fresh project creation (empty directory).
    """
    _subprocess.run(
        ["git", "add", "-A"],
        cwd=target,
        capture_output=True,
        check=True,
    )

    # Set committer identity via env so the commit succeeds even when
    # the user has no global git config (e.g. CI, fresh containers).
    env = {
        **os.environ,
        "GIT_COMMITTER_NAME": "Osmosis",
        "GIT_COMMITTER_EMAIL": "noreply@osmosis.ai",
    }
    _subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Initial project setup",
            "--author",
            "Osmosis <noreply@osmosis.ai>",
        ],
        cwd=target,
        capture_output=True,
        check=True,
        env=env,
    )


def _resolve_workspace_git_context(
    *,
    workspace_name: str,
    workspace_id: str | None,
    credentials: Any,
) -> dict[str, str | bool | None]:
    """Fetch the target workspace's Git Sync metadata from the platform.

    The caller is responsible for resolving the workspace identity; this
    function only refreshes the connected-repo state and GitHub App
    installation flag used to drive the post-init CTA and the connected-repo
    guard.

    Network/auth errors during the metadata refresh degrade the
    returned context (no connected repo, no GitHub App), but the
    workspace identity and Git Sync URL are always preserved so the
    CTA still points at the right page. The connected-repo guard only
    blocks when the platform confirms a connected repo, so a transient
    outage doesn't spuriously block init.
    """
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.auth import (
        AuthenticationExpiredError,
        PlatformAPIError,
    )
    from osmosis_ai.platform.cli.utils import platform_call

    context: dict[str, str | bool | None] = {
        "workspace_id": workspace_id,
        "workspace_name": workspace_name,
        "git_sync_url": f"{PLATFORM_URL}/{workspace_name}/integrations/git",
        "has_github_app_installation": False,
        "connected_repo_url": None,
    }

    client = OsmosisClient()
    try:
        info = platform_call(
            "Loading workspace Git Sync status...",
            lambda: client.refresh_workspace_info(
                credentials=credentials,
                workspace_name=workspace_name,
                cleanup_on_401=False,
            ),
            output_console=console,
        )
    except (AuthenticationExpiredError, PlatformAPIError):
        return context

    context["has_github_app_installation"] = bool(
        info.get("has_github_app_installation")
    )
    connected_repo = info.get("connected_repo")
    if isinstance(connected_repo, dict):
        repo_url = connected_repo.get("repo_url")
        if isinstance(repo_url, str) and repo_url:
            context["connected_repo_url"] = repo_url
    return context


def _git_sync_cta_text(git_context: dict[str, str | bool | None]) -> Any:
    """Build the Git Sync CTA shown after project scaffolding."""
    from rich.text import Text

    connected_repo_url = git_context.get("connected_repo_url")
    if isinstance(connected_repo_url, str) and connected_repo_url:
        return Text.assemble(
            "Connected repo: ",
            console.format_url(connected_repo_url, style="cyan"),
        )

    git_sync_url = git_context.get("git_sync_url")
    if isinstance(git_sync_url, str) and git_sync_url:
        action = (
            "choose a repo"
            if git_context.get("has_github_app_installation")
            else "connect your repo"
        )
        return Text.assemble(
            f"{action}: ",
            console.format_url(git_sync_url, style="cyan"),
        )

    return Text.assemble(
        "connect your repo with Git Sync: ",
        console.format_url(PLATFORM_URL, style="cyan"),
    )


def _raise_if_selected_workspace_has_connected_repo(
    git_context: dict[str, str | bool | None],
) -> None:
    """Block subdirectory create when the selected workspace already has a repo."""
    workspace_name = git_context.get("workspace_name")
    connected_repo_url = git_context.get("connected_repo_url")

    if not (
        isinstance(workspace_name, str)
        and workspace_name
        and isinstance(connected_repo_url, str)
        and connected_repo_url
    ):
        return

    raise CLIError(
        f"Workspace '{workspace_name}' is already connected to:\n"
        f"  {connected_repo_url}\n"
        "\n"
        "Clone the connected repo instead:\n"
        f"  git clone {connected_repo_url}\n"
        "\n"
        "Or adopt an existing checkout from its Git worktree root:\n"
        f"  osmosis init {workspace_name} --here --workspace {workspace_name}"
    )


def _require_init_link_choice(workspace: str | None, no_link: bool) -> None:
    """Require an explicit link/no-link choice in non-interactive modes."""
    from osmosis_ai.cli.output import OutputFormat, get_output_context

    if workspace and no_link:
        raise CLIError("Use either --workspace or --no-link, not both.")

    if workspace or no_link:
        return

    output = get_output_context()
    if output.format is OutputFormat.rich and output.interactive:
        return

    raise CLIError(
        "Use --workspace to link this project, or --no-link to create it without linking.",
        code="INTERACTIVE_REQUIRED",
    )


def _require_link_credentials() -> Any:
    """Require login credentials for workspace-linked init."""
    from osmosis_ai.platform.cli.project import _require_link_credentials as require

    return require()


def _resolve_workspace_for_link(ref: str, credentials: Any) -> dict[str, Any]:
    """Resolve a workspace ID/name for init-time linking."""
    from osmosis_ai.platform.cli.workspace_context import (
        list_accessible_workspaces,
        resolve_workspace_ref,
    )

    return resolve_workspace_ref(
        ref, list_accessible_workspaces(credentials=credentials)
    )


def _target_for_init(name: str, *, here: bool) -> tuple[Path, bool]:
    """Return the target project path and whether init creates a new directory."""
    if here:
        return Path.cwd().resolve(), False
    return (Path.cwd() / name).resolve(), True


def _git_worktree_top_level(path: Path) -> Path | None:
    """Return the Git worktree root containing *path*, or ``None``."""
    if shutil.which("git") is None:
        return None
    result = _subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top).resolve() if top else None


def _is_empty_or_only_git_dir(path: Path) -> bool:
    """Return whether *path* is empty or contains only a ``.git`` entry."""
    return all(child.name == ".git" for child in path.iterdir())


def _snapshot_existing_paths(root: Path) -> set[Path]:
    """Capture paths that existed before an adopt-mode init attempt."""
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*")}


def _remove_paths_created_after(root: Path, before: set[Path]) -> None:
    """Remove files/directories created after *before* without touching existing paths."""
    if not root.exists():
        return
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if path.resolve() in before:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _repo_url_from_workspace(workspace: dict[str, Any]) -> str | None:
    from osmosis_ai.platform.cli.project import _repo_url_from_workspace as repo_url

    return repo_url(workspace)


def _raw_repo_url_from_workspace(workspace: dict[str, Any]) -> str | None:
    connected_repo = workspace.get("connected_repo")
    if isinstance(connected_repo, dict):
        repo_url = connected_repo.get("repo_url")
        if isinstance(repo_url, str) and repo_url:
            return repo_url
    repo_url = workspace.get("repo_url")
    if isinstance(repo_url, str) and repo_url:
        return repo_url
    return None


def _workspace_summary(workspace: dict[str, Any]) -> dict[str, str]:
    workspace_id = workspace.get("id")
    workspace_name = workspace.get("name")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise CLIError("Invalid workspace record from Osmosis platform: missing id.")
    if not isinstance(workspace_name, str) or not workspace_name:
        raise CLIError("Invalid workspace record from Osmosis platform: missing name.")
    return {"id": workspace_id, "name": workspace_name}


def _git_context_from_workspace(
    workspace: dict[str, Any],
) -> dict[str, str | bool | None]:
    summary = _workspace_summary(workspace)
    repo_url = _raw_repo_url_from_workspace(workspace)
    return {
        "workspace_id": summary["id"],
        "workspace_name": summary["name"],
        "git_sync_url": f"{PLATFORM_URL}/{summary['name']}/integrations/git",
        "has_github_app_installation": bool(
            workspace.get("has_github_app_installation")
        ),
        "connected_repo_url": repo_url,
    }


def _require_origin_matches_connected_repo(
    target: Path,
    git_context: dict[str, str | bool | None],
) -> None:
    """Require ``target`` to be a checkout of the workspace connected repo."""
    from osmosis_ai.platform.cli.workspace_repo import (
        get_local_git_remote_url,
        normalize_git_url,
    )

    connected_repo_url = git_context.get("connected_repo_url")
    if not isinstance(connected_repo_url, str) or not connected_repo_url:
        return

    expected = normalize_git_url(connected_repo_url)
    local_remote = get_local_git_remote_url(target)
    if expected is not None and normalize_git_url(local_remote) == expected:
        return

    workspace_name = git_context.get("workspace_name")
    raise CLIError(
        f"Workspace '{workspace_name}' is already connected to:\n"
        f"  {connected_repo_url}\n"
        "\n"
        "Run this command from a clone of the connected repo:\n"
        f"  git clone {connected_repo_url}\n"
        "\n"
        "Or choose a workspace without a connected repo."
    )


def _print_next_steps(
    project_name: str,
    *,
    here: bool = False,
    git_context: dict[str, str | bool | None],
) -> None:
    """Print post-setup CTA with Rich panels."""
    from rich import box as rich_box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    cmd_table = Table.grid(padding=(0, 1))
    cmd_table.add_column(no_wrap=True)
    cmd_table.add_column()
    if not here:
        cmd_table.add_row(
            "[bold green]$[/bold green]", console.format_text(f"cd {project_name}")
        )
    cmd_table.add_row(
        "[bold green]$[/bold green]",
        "gh repo create --private --source=. --push",
    )
    cmd_table.add_row(
        "[dim] [/dim]",
        "[dim]or[/dim]",
    )
    cmd_table.add_row(
        "[bold green]$[/bold green]",
        "git remote add origin <your-repo-url>",
    )
    cmd_table.add_row(
        "[bold green]$[/bold green]",
        "git push -u origin main",
    )
    cmd_table.add_row()
    cmd_table.add_row(
        "[bold green]>[/bold green]",
        _git_sync_cta_text(git_context),
    )

    prompt_body = (
        "I want to train a model for <my task domain>. "
        "Read .osmosis/program.md, create a baseline rollout in the "
        "canonical project structure, iterate locally with evals, and prepare "
        "a training config. Use `osmosis --json` for Osmosis CLI commands."
    )

    plugin_repo = _plugin_repo()
    plugin_marketplace = _plugin_marketplace()
    plugin_table = Table.grid(padding=(0, 1))
    plugin_table.add_column(no_wrap=True)
    plugin_table.add_column(no_wrap=True)
    plugin_table.add_row(
        "[bold cyan]Claude Code[/bold cyan]",
        "open this folder — it'll prompt to install [cyan]osmosis[/cyan]",
    )
    plugin_table.add_row(
        "[bold cyan]Cursor[/bold cyan]",
        Text.assemble(
            "Settings → Rules → Add Remote Rule → ",
            console.format_text(plugin_repo, style="cyan"),
        ),
    )
    plugin_table.add_row(
        "[bold cyan]Codex[/bold cyan]",
        console.format_text(
            f"codex plugin marketplace add {plugin_repo}", style="cyan"
        ),
    )
    plugin_table.add_row(
        "[dim] [/dim]",
        "[dim]then[/dim]",
    )
    plugin_table.add_row(
        "[bold cyan] [/bold cyan]",
        console.format_text(f"codex plugin install {plugin_marketplace}", style="cyan"),
    )

    content = Group(
        Panel(
            cmd_table,
            title="next steps",
            border_style="green",
            box=rich_box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ),
        Text(),
        Panel(
            plugin_table,
            title="install the osmosis agent plugin",
            border_style="cyan",
            box=rich_box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ),
        Text(),
        Panel(
            Text(prompt_body, style="italic"),
            title="ask your ai agent",
            border_style="magenta",
            box=rich_box.ROUNDED,
            padding=(1, 2),
            expand=False,
        ),
    )

    console.print()
    console.rich.print(
        Panel(
            content,
            title="[bold white]get started[/bold white]",
            border_style="bright_blue",
            box=rich_box.DOUBLE,
            padding=(1, 2),
            expand=False,
        )
    )


def _created_paths_for_result(target: Path) -> list[str]:
    """Return scaffold paths to expose in machine-readable init output."""
    paths: list[Path] = [target]
    paths.extend(
        target / entry.dest for entry in SCAFFOLD if (target / entry.dest).exists()
    )
    if (target / ".git").exists():
        paths.append(target / ".git")
    return [str(path.resolve()) for path in paths]


def _workspace_for_result(
    git_context: dict[str, str | bool | None],
) -> dict[str, str] | None:
    """Shape workspace metadata for the JSON/plain ``init`` result envelope.

    Linked init results should include ``workspace_name``; ``workspace_id``
    may be missing when the workspace was resolved from an older context.
    """
    workspace_id = git_context.get("workspace_id")
    workspace_name = git_context.get("workspace_name")
    if not (isinstance(workspace_name, str) and workspace_name):
        return None
    if isinstance(workspace_id, str) and workspace_id:
        return {"id": workspace_id, "name": workspace_name}
    return {"name": workspace_name}


def _init_next_steps_structured(
    project_name: str,
    *,
    here: bool,
    git_context: dict[str, str | bool | None],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if not here:
        steps.append(
            {"action": "cd", "path": project_name, "command": f"cd {project_name}"}
        )
    steps.extend(
        [
            {
                "action": "create_github_repo",
                "command": "gh repo create --private --source=. --push",
            },
            {
                "action": "add_git_remote",
                "command": "git remote add origin <your-repo-url>",
            },
            {"action": "push_git", "command": "git push -u origin main"},
        ]
    )

    connected_repo_url = git_context.get("connected_repo_url")
    if isinstance(connected_repo_url, str) and connected_repo_url:
        steps.append({"action": "clone_connected_repo", "url": connected_repo_url})
    else:
        git_sync_url = git_context.get("git_sync_url")
        if isinstance(git_sync_url, str) and git_sync_url:
            steps.append({"action": "configure_git_sync", "url": git_sync_url})

    plugin_repo = _plugin_repo()
    plugin_marketplace = _plugin_marketplace()
    steps.extend(
        [
            {
                "action": "install_cursor_rule",
                "source": plugin_repo,
            },
            {
                "action": "install_codex_plugin",
                "commands": [
                    f"codex plugin marketplace add {plugin_repo}",
                    f"codex plugin install {plugin_marketplace}",
                ],
            },
        ]
    )
    return steps


def _init_display_next_steps(project_name: str, *, here: bool) -> list[str]:
    steps: list[str] = []
    if not here:
        steps.append(f"cd {project_name}")
    steps.extend(
        [
            "gh repo create --private --source=. --push",
            "git remote add origin <your-repo-url>",
            "git push -u origin main",
        ]
    )
    return steps


# ── Main entry point ─────────────────────────────────────────────


def init(
    name: str,
    here: bool = False,
    workspace: str | None = None,
    no_link: bool = False,
) -> Any:
    """Initialise a new local Osmosis project directory.

    This is the main entry point for ``osmosis init <name>``.

    Non-interactive calls must choose either ``workspace`` (link after
    scaffolding) or ``no_link`` (create/adopt without platform auth).
    """
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context

    output = get_output_context()
    _check_git_installed()
    _require_init_link_choice(workspace, no_link)

    name_error = validate_name(name, label="Project name")
    if name_error:
        raise CLIError(name_error)

    target, creates_subdir = _target_for_init(name, here=here)
    mode = "adopt" if here else "create"
    created_dir = False
    before_paths: set[Path] = set()
    record: ProjectLinkRecord | None = None
    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    git_context: dict[str, str | bool | None] = {
        "workspace_id": None,
        "workspace_name": None,
        "git_sync_url": None,
        "has_github_app_installation": False,
        "connected_repo_url": None,
    }

    if here:
        git_top = _git_worktree_top_level(target)
        if git_top is not None and git_top != target:
            raise CLIError(
                "--here must be run from the Git worktree top-level, not a subdirectory."
            )
        if git_top is None and not _is_empty_or_only_git_dir(target):
            raise CLIError(
                "--here must be run from an empty directory or Git worktree top-level."
            )
    elif target.exists():
        if (target / ".osmosis" / "project.toml").is_file():
            raise CLIError(
                "Already an Osmosis project. Use 'osmosis project link' to change "
                "workspace links, or 'osmosis project doctor --fix' to repair "
                "project scaffolding."
            )
        raise CLIError(
            f"Directory ./{name} already exists. "
            "Use --here to initialize in the current directory."
        )

    if (target / ".osmosis" / "project.toml").is_file():
        raise CLIError(
            "Already an Osmosis project. Use 'osmosis project link' to change "
            "workspace links, or 'osmosis project doctor --fix' to repair "
            "project scaffolding."
        )

    if workspace:
        credentials = _require_link_credentials()
        selected_workspace = _resolve_workspace_for_link(workspace, credentials)
        git_context = _git_context_from_workspace(selected_workspace)
        if creates_subdir:
            _raise_if_selected_workspace_has_connected_repo(git_context)
        else:
            _require_origin_matches_connected_repo(target, git_context)
        summary = _workspace_summary(selected_workspace)
        record = ProjectLinkRecord(
            project_path=str(target.resolve()),
            workspace_id=summary["id"],
            workspace_name=summary["name"],
            repo_url=_repo_url_from_workspace(selected_workspace),
            linked_at=now_linked_at(),
        )
        store.check_link_allowed(record)

    try:
        if creates_subdir:
            target.mkdir()
            created_dir = True
        else:
            before_paths = _snapshot_existing_paths(target)

        _write_scaffold(target, name, update=False)
        if creates_subdir:
            _git_init(target)
            _git_initial_commit(target)
        else:
            _git_init(target)

        if record is not None:
            store.link(record)
    except Exception:
        if created_dir and target.exists():
            shutil.rmtree(target)
        elif not creates_subdir:
            _remove_paths_created_after(target, before_paths)
        raise

    if output.format is OutputFormat.rich:
        _print_next_steps(name, here=here, git_context=git_context)
        return None

    resource = {
        "path": str(target.resolve()),
        "created_paths": _created_paths_for_result(target),
        "workspace": _workspace_for_result(git_context) if record is not None else None,
        "linked": record is not None,
        "mode": mode,
    }
    return OperationResult(
        operation="init",
        status="success",
        resource=resource,
        message=f"Initialized project in {target.resolve()}.",
        display_next_steps=_init_display_next_steps(name, here=here),
        next_steps_structured=_init_next_steps_structured(
            name,
            here=here,
            git_context=git_context,
        ),
    )

"""Workspace init — scaffold a new local workspace directory.

This module implements the core flow for ``osmosis init <name>``:
check prerequisites, create the directory, scaffold files from
bundled templates, initialise git, and print next steps.
"""

from __future__ import annotations

import os
import shutil
import subprocess as _subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.cli.constants import validate_name

# ── Prerequisites ────────────────────────────────────────────────


def _check_git_installed() -> None:
    """Verify that ``git`` is available on PATH."""
    if shutil.which("git") is None:
        raise CLIError(
            "Git is not installed or not on PATH.\n"
            "  Install it from https://git-scm.com/ and try again."
        )


# ── Scaffold manifest ───────────────────────────────────────────

_TEMPLATES = files("osmosis_ai.platform.cli") / "templates"


@dataclass(frozen=True, slots=True)
class ScaffoldEntry:
    """A single file to create during workspace scaffolding.

    Attributes:
        template: Relative path inside the templates package directory.
            Empty string means write an empty file (used for ``.gitkeep``).
        dest: Relative path inside the target workspace directory.
        render: Whether to apply ``str.format_map()`` variable substitution.
        overwrite_on_update: If True, overwrite existing file during update mode.
    """

    template: str
    dest: str
    render: bool = False
    overwrite_on_update: bool = False


SCAFFOLD: list[ScaffoldEntry] = [
    # Directory placeholders
    ScaffoldEntry("", "rollouts/.gitkeep"),
    ScaffoldEntry("", "configs/eval/.gitkeep"),
    ScaffoldEntry("", "data/.gitkeep"),
    # Rendered templates (workspace.toml handled separately on update)
    ScaffoldEntry("workspace.toml.tpl", ".osmosis/workspace.toml", render=True),
    ScaffoldEntry("pyproject.toml.tpl", "pyproject.toml", render=True),
    ScaffoldEntry("README.md.tpl", "README.md", render=True),
    # Static files — configs (skip on update)
    ScaffoldEntry("gitignore.tpl", ".gitignore"),
    ScaffoldEntry("configs/training/default.toml.tpl", "configs/training/default.toml"),
    # Agent docs & skills (overwrite on update)
    ScaffoldEntry("AGENTS.md.tpl", "AGENTS.md", overwrite_on_update=True),
    ScaffoldEntry("CLAUDE.md.tpl", "CLAUDE.md", overwrite_on_update=True),
    ScaffoldEntry(
        "configs/AGENTS.md.tpl", "configs/AGENTS.md", overwrite_on_update=True
    ),
    ScaffoldEntry(
        "skills/create-rollout/SKILL.md.tpl",
        ".osmosis/skills/create-rollout/SKILL.md",
        overwrite_on_update=True,
    ),
    ScaffoldEntry(
        "skills/evaluate-rollout/SKILL.md.tpl",
        ".osmosis/skills/evaluate-rollout/SKILL.md",
        overwrite_on_update=True,
    ),
    ScaffoldEntry(
        "skills/submit-training/SKILL.md.tpl",
        ".osmosis/skills/submit-training/SKILL.md",
        overwrite_on_update=True,
    ),
]


# ── Scaffold generation ─────────────────────────────────────────


def _render_template(name: str, variables: dict[str, str]) -> str:
    """Read a template file and apply variable substitution."""
    text = (_TEMPLATES / name).read_text(encoding="utf-8")
    return text.format_map(variables)


def _write_scaffold(target: Path, ws_name: str, *, update: bool = False) -> None:
    """Write all scaffold files into *target* from the SCAFFOLD manifest.

    In normal mode, files that already exist are skipped (idempotent).
    In *update* mode, entries with ``overwrite_on_update=True`` are
    overwritten; all other existing files are left untouched.
    """
    import datetime

    from osmosis_ai.consts import PACKAGE_VERSION

    variables = {
        "name": ws_name,
        "sdk_version": PACKAGE_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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


# ── Workspace metadata update ───────────────────────────────────

_CREATED_AT_RE = r'created_at\s*=\s*"([^"]*)"'


def _update_workspace_metadata(target: Path) -> None:
    """Update workspace.toml: refresh ``sdk_version``, add ``updated_at``, preserve ``created_at``."""
    import datetime
    import re

    from osmosis_ai.consts import PACKAGE_VERSION

    ws_toml = target / ".osmosis" / "workspace.toml"
    original = ws_toml.read_text(encoding="utf-8") if ws_toml.is_file() else ""

    match = re.search(_CREATED_AT_RE, original)
    created_at = (
        match.group(1)
        if match
        else datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    ws_toml.write_text(
        "# Osmosis Workspace Configuration\n"
        "# Generated by `osmosis init`\n"
        "\n"
        "[workspace]\n"
        f'sdk_version = "{PACKAGE_VERSION}"\n'
        f'created_at = "{created_at}"\n'
        f'updated_at = "{updated_at}"\n'
        'setup_source = "osmosis init"\n',
        encoding="utf-8",
    )


# ── Git helpers ──────────────────────────────────────────────────


def _git_init(target: Path) -> None:
    """Initialise a git repository in the target directory.

    Skips if ``.git/`` already exists.
    """
    if (target / ".git").is_dir():
        return

    _subprocess.run(
        ["git", "init", "-b", "main", str(target)],
        capture_output=True,
        check=True,
    )


def _git_initial_commit(target: Path) -> None:
    """Stage all files and create the initial git commit.

    Only called for fresh workspace creation (empty directory).
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
            "Initial workspace setup",
            "--author",
            "Osmosis <noreply@osmosis.ai>",
        ],
        cwd=target,
        capture_output=True,
        check=True,
        env=env,
    )


def _print_next_steps(ws_name: str, *, here: bool = False) -> None:
    """Print post-setup CTA with Rich panels."""
    from rich import box as rich_box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    cmd_table = Table.grid(padding=(0, 1))
    if not here:
        cmd_table.add_row("[bold green]$[/bold green]", f"cd {ws_name}")
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
        f"Go to [cyan]{PLATFORM_URL}[/cyan] → Git Sync to connect your repo",
    )

    prompt_body = (
        "I want to train a model for <my task domain>. "
        "Create an initial rollout with tools and a grader, "
        "then run a quick eval baseline."
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


# ── Main entry point ─────────────────────────────────────────────


def init(name: str, here: bool = False) -> None:
    """Initialise a new local workspace directory.

    This is the main entry point for ``osmosis init <name>``.
    """
    _check_git_installed()

    name_error = validate_name(name, label="Workspace name")
    if name_error:
        raise CLIError(name_error)

    # Determine target directory
    created_dir = False
    if here:
        target = Path.cwd()
        if any(p.name != ".git" for p in target.iterdir()):
            raise CLIError(
                "Current directory is not empty. "
                "Use 'osmosis init <name>' (without --here) to create a new directory, "
                "or empty this directory first."
            )
    else:
        target = Path.cwd() / name
        if target.exists():
            if (target / ".osmosis" / "workspace.toml").is_file():
                # Re-entry: update mode
                console.print(
                    f"Updating existing workspace in [cyan]./{name}[/cyan]...",
                )
            else:
                raise CLIError(
                    f"Directory ./{name} already exists. "
                    "Use --here to initialize in the current directory."
                )
        else:
            target.mkdir()
            created_dir = True

    is_update = (target / ".osmosis" / "workspace.toml").is_file()

    try:
        _write_scaffold(target, name, update=is_update)
        if is_update:
            _update_workspace_metadata(target)
        else:
            _git_init(target)
            _git_initial_commit(target)
    except Exception:
        if created_dir and target.exists():
            shutil.rmtree(target)
        raise

    _print_next_steps(name, here=here)

"""Workspace init — scaffold a new local workspace directory.

This module implements the core flow for ``osmosis init <name>``:
check prerequisites, create the directory, scaffold files from
bundled templates, initialise git, and print next steps.
"""

from __future__ import annotations

import shutil
import subprocess as _subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import PLATFORM_URL

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
    """

    template: str
    dest: str
    render: bool = False


SCAFFOLD: list[ScaffoldEntry] = [
    # Directory placeholders
    ScaffoldEntry("", "rollouts/.gitkeep"),
    ScaffoldEntry("", "configs/eval/.gitkeep"),
    ScaffoldEntry("", "data/.gitkeep"),
    # Rendered templates
    ScaffoldEntry("workspace.toml.tpl", ".osmosis/workspace.toml", render=True),
    ScaffoldEntry("pyproject.toml.tpl", "pyproject.toml", render=True),
    ScaffoldEntry("README.md.tpl", "README.md", render=True),
    # Static files
    ScaffoldEntry("gitignore.tpl", ".gitignore"),
    ScaffoldEntry("AGENTS.md.tpl", "AGENTS.md"),
    ScaffoldEntry("CLAUDE.md.tpl", "CLAUDE.md"),
    ScaffoldEntry("configs/AGENTS.md.tpl", "configs/AGENTS.md"),
    ScaffoldEntry("configs/training/default.toml.tpl", "configs/training/default.toml"),
    ScaffoldEntry(
        "skills/create-rollout/SKILL.md.tpl",
        ".osmosis/skills/create-rollout/SKILL.md",
    ),
    ScaffoldEntry(
        "skills/evaluate-rollout/SKILL.md.tpl",
        ".osmosis/skills/evaluate-rollout/SKILL.md",
    ),
    ScaffoldEntry(
        "skills/submit-training/SKILL.md.tpl",
        ".osmosis/skills/submit-training/SKILL.md",
    ),
]


# ── Scaffold generation ─────────────────────────────────────────


def _render_template(name: str, variables: dict[str, str]) -> str:
    """Read a template file and apply variable substitution."""
    text = (_TEMPLATES / name).read_text(encoding="utf-8")
    return text.format_map(variables)


def _write_scaffold(target: Path, ws_name: str) -> None:
    """Write all scaffold files into *target* from the SCAFFOLD manifest.

    Files that already exist are **not** overwritten (idempotent).
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
        if dest.exists():
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


# ── Git helpers ──────────────────────────────────────────────────


def _git_init(target: Path) -> None:
    """Initialise a git repository in the target directory.

    Skips if ``.git/`` already exists.
    """
    if (target / ".git").is_dir():
        return

    _subprocess.run(
        ["git", "init", str(target)],
        capture_output=True,
        check=True,
    )


def _git_initial_commit(target: Path, *, update: bool = False) -> None:
    """Stage all files and create a git commit.

    In *update* mode the commit is skipped when there are no staged
    changes (all scaffold files already existed).
    """
    _subprocess.run(
        ["git", "add", "-A"],
        cwd=target,
        capture_output=True,
        check=True,
    )

    # When updating an existing workspace, nothing may have changed.
    if update:
        result = _subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=target,
            capture_output=True,
        )
        if result.returncode == 0:
            # Nothing staged — skip commit.
            return

    msg = "Update workspace scaffold" if update else "Initial workspace setup"
    _subprocess.run(
        ["git", "commit", "-m", msg, "--author", "Osmosis Init <noreply@osmosis.ai>"],
        cwd=target,
        capture_output=True,
        check=True,
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
        _write_scaffold(target, name)
        _git_init(target)
        _git_initial_commit(target, update=is_update)
    except Exception:
        if created_dir and target.exists():
            shutil.rmtree(target)
        raise

    _print_next_steps(name, here=here)

"""Handler for `osmosis upgrade` — self-upgrade the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.consts import PACKAGE_VERSION, package_name

PYPI_URL = f"https://pypi.org/pypi/{package_name}/json"


def _is_up_to_date(installed: str, latest: str) -> bool:
    """Return True if installed >= latest using PEP 440 version comparison."""
    try:
        from packaging.version import Version

        return Version(installed) >= Version(latest)
    except Exception:
        # If packaging is unavailable or versions are unparseable,
        # fall back to simple numeric tuple comparison.
        pass

    try:
        installed_t = tuple(int(x) for x in installed.split("."))
        latest_t = tuple(int(x) for x in latest.split("."))
        return installed_t >= latest_t
    except (ValueError, AttributeError):
        # Cannot reliably compare — assume an upgrade is needed.
        return False


def _fetch_latest_version() -> str | None:
    """Fetch the latest version from PyPI."""
    try:
        req = urllib.request.Request(PYPI_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _detect_install_method() -> str:
    """Detect how the package was installed.

    Returns one of: "uv_tool", "pipx", "pip".
    """
    exe_path = sys.executable
    if "uv/tools" in exe_path or "uv\\tools" in exe_path:
        return "uv_tool"
    if "pipx/venvs" in exe_path or "pipx\\venvs" in exe_path:
        return "pipx"
    return "pip"


def _get_upgrade_commands(method: str) -> list[list[str]]:
    """Return the shell commands to try for upgrading the package."""
    commands: dict[str, list[list[str]]] = {
        "uv_tool": [["uv", "tool", "upgrade", package_name]],
        "pipx": [["pipx", "upgrade", package_name]],
        "pip": [
            ["uv", "pip", "install", "--upgrade", package_name],
            [sys.executable, "-m", "pip", "install", "--upgrade", package_name],
        ],
    }
    return commands.get(method, commands["pip"])


def upgrade() -> None:
    """Upgrade the Osmosis CLI to the latest version."""
    installed = PACKAGE_VERSION

    console.print(f"Installed version: {installed}")
    console.print("Checking for updates...", style="dim")

    latest = _fetch_latest_version()
    if latest is None:
        console.print_error("Failed to check for updates from PyPI.")
        raise typer.Exit(1)

    console.print(f"Latest version:    {latest}")
    console.print()

    if _is_up_to_date(installed, latest):
        console.print("Already up to date!", style="green")
        return

    console.print(
        f"A newer version is available: {latest}",
        style="bold yellow",
    )
    console.print()

    method = _detect_install_method()
    cmds = _get_upgrade_commands(method)

    console.print(f"Detected install method: {method}")
    console.print()

    for cmd in cmds:
        if shutil.which(cmd[0]) is None:
            continue

        console.print(f"Running: {' '.join(cmd)}", style="dim")
        try:
            result = subprocess.run(cmd, timeout=120, stdin=subprocess.DEVNULL)
            if result.returncode == 0:
                console.print()
                console.print(f"Successfully upgraded to {latest}!", style="bold green")
                return
            console.print_error(f"Command failed (exit {result.returncode}).")
        except subprocess.TimeoutExpired:
            console.print_error("Upgrade command timed out.")
        except Exception as exc:
            console.print_error(f"Error running upgrade: {exc}")

    console.print()
    console.print_error("Upgrade failed. You can try manually:")
    console.print(f"  uv tool upgrade {package_name}", style="dim")
    console.print(f"  pipx upgrade {package_name}", style="dim")
    console.print(f"  pip install --upgrade {package_name}", style="dim")
    raise typer.Exit(1)

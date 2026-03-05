"""Handler for `osmosis upgrade` — self-upgrade the CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request

from osmosis_ai.cli.console import console
from osmosis_ai.consts import PACKAGE_VERSION, package_name

PYPI_URL = f"https://pypi.org/pypi/{package_name}/json"


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

    Returns one of: "uv_tool", "pipx", "uv", "pip".
    """
    # Check if installed via uv tool
    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "tool", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and package_name in result.stdout:
                return "uv_tool"
        except Exception:
            pass

    # Check if installed via pipx
    if shutil.which("pipx"):
        try:
            result = subprocess.run(
                ["pipx", "list", "--short"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and package_name in result.stdout:
                return "pipx"
        except Exception:
            pass

    # Check if uv is available (prefer uv pip over plain pip)
    if shutil.which("uv"):
        return "uv"

    return "pip"


def _get_upgrade_command(method: str) -> list[str]:
    """Return the shell command to upgrade the package."""
    commands = {
        "uv_tool": ["uv", "tool", "upgrade", package_name],
        "pipx": ["pipx", "upgrade", package_name],
        "uv": ["uv", "pip", "install", "--upgrade", package_name],
    }
    return commands.get(
        method, [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
    )


class UpgradeCommand:
    """Handler for `osmosis upgrade`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)

    def run(self, args: argparse.Namespace) -> int:
        installed = PACKAGE_VERSION

        console.print(f"Installed version: {installed}")
        console.print("Checking for updates...", style="dim")

        latest = _fetch_latest_version()
        if latest is None:
            console.print_error("Failed to check for updates from PyPI.")
            return 1

        console.print(f"Latest version:    {latest}")
        console.print()

        if installed == latest:
            console.print("Already up to date!", style="green")
            return 0

        console.print(
            f"A newer version is available: {latest}",
            style="bold yellow",
        )
        console.print()

        method = _detect_install_method()
        cmd = _get_upgrade_command(method)

        console.print(f"Detected install method: {method}")
        console.print(f"Running: {' '.join(cmd)}", style="dim")
        console.print()

        try:
            result = subprocess.run(cmd, timeout=120)
        except subprocess.TimeoutExpired:
            console.print_error("Upgrade timed out.")
            return 1
        except Exception as exc:
            console.print_error(f"Upgrade failed: {exc}")
            return 1

        if result.returncode != 0:
            console.print_error("Upgrade command failed.")
            return 1

        console.print()
        console.print(f"Successfully upgraded to {latest}!", style="bold green")
        return 0

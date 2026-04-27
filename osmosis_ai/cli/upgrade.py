"""Handler for `osmosis upgrade` — self-upgrade the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
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
        # Pad shorter tuple with zeros to ensure correct comparison
        # (e.g., "1.2" == "1.2.0")
        max_len = max(len(installed_t), len(latest_t))
        installed_t += (0,) * (max_len - len(installed_t))
        latest_t += (0,) * (max_len - len(latest_t))
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
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--upgrade",
                package_name,
            ],
            [sys.executable, "-m", "pip", "install", "--upgrade", package_name],
        ],
    }
    return commands.get(method, commands["pip"])


def _upgrade_resource(
    *,
    installed: str,
    latest: str | None,
    method: str | None = None,
    command: list[str] | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "installed_version": installed,
        "latest_version": latest,
    }
    if method is not None:
        resource["method"] = method
    if command is not None:
        resource["command"] = command
    if stdout:
        resource["stdout"] = stdout
    if stderr:
        resource["stderr"] = stderr
    return resource


def upgrade() -> Any:
    """Upgrade the Osmosis CLI to the latest version."""
    installed = PACKAGE_VERSION
    output = get_output_context()
    structured_output = output.format is not OutputFormat.rich

    console.print(f"Installed version: {installed}")
    console.print("Checking for updates...", style="dim")

    latest = _fetch_latest_version()
    if latest is None:
        if structured_output:
            raise CLIError(
                "Failed to check for updates from PyPI.", code="PLATFORM_ERROR"
            )
        console.print_error("Failed to check for updates from PyPI.")
        raise typer.Exit(1)

    console.print(f"Latest version:    {latest}")
    console.print()

    if _is_up_to_date(installed, latest):
        if structured_output:
            return OperationResult(
                operation="upgrade",
                status="no_update",
                resource=_upgrade_resource(installed=installed, latest=latest),
                message="Already up to date.",
            )
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

    last_failure: OperationResult | None = None

    for cmd in cmds:
        if shutil.which(cmd[0]) is None:
            continue

        console.print(f"Running: {' '.join(cmd)}", style="dim")
        try:
            run_kwargs: dict[str, Any] = {"timeout": 120}
            if structured_output:
                run_kwargs.update({"capture_output": True, "text": True})
            else:
                run_kwargs["stdout"] = subprocess.DEVNULL
            result = subprocess.run(cmd, **run_kwargs)
            if result.returncode == 0:
                if structured_output:
                    return OperationResult(
                        operation="upgrade",
                        status="success",
                        resource=_upgrade_resource(
                            installed=installed,
                            latest=latest,
                            method=method,
                            command=cmd,
                            stdout=getattr(result, "stdout", None),
                            stderr=getattr(result, "stderr", None),
                        ),
                        message=f"Successfully upgraded to {latest}.",
                    )
                console.print()
                console.print(f"Successfully upgraded to {latest}!", style="bold green")
                return
            if structured_output:
                last_failure = OperationResult(
                    operation="upgrade",
                    status="failed",
                    resource=_upgrade_resource(
                        installed=installed,
                        latest=latest,
                        method=method,
                        command=cmd,
                        stdout=getattr(result, "stdout", None),
                        stderr=getattr(result, "stderr", None),
                    ),
                    message="Upgrade failed.",
                    exit_code=1,
                )
                continue
            console.print()
        except subprocess.TimeoutExpired:
            if structured_output:
                last_failure = OperationResult(
                    operation="upgrade",
                    status="failed",
                    resource=_upgrade_resource(
                        installed=installed,
                        latest=latest,
                        method=method,
                        command=cmd,
                    ),
                    message="Upgrade command timed out.",
                    exit_code=1,
                )
                continue
            console.print_error("Upgrade command timed out.")
        except Exception as exc:
            if structured_output:
                last_failure = OperationResult(
                    operation="upgrade",
                    status="failed",
                    resource=_upgrade_resource(
                        installed=installed,
                        latest=latest,
                        method=method,
                        command=cmd,
                        stderr=str(exc),
                    ),
                    message="Error running upgrade.",
                    exit_code=1,
                )
                continue
            console.print_error(f"Error running upgrade: {exc}")

    if structured_output:
        if last_failure is not None:
            return last_failure
        return OperationResult(
            operation="upgrade",
            status="failed",
            resource=_upgrade_resource(
                installed=installed,
                latest=latest,
                method=method,
            ),
            message="Upgrade failed.",
            exit_code=1,
        )

    console.print()
    console.print_error("Upgrade failed. You can try manually:")
    console.print(f"  uv tool upgrade {package_name}", style="dim")
    console.print(f"  pipx upgrade {package_name}", style="dim")
    console.print(f"  pip install --upgrade {package_name}", style="dim")
    raise typer.Exit(1)

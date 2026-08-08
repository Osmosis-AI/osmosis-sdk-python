"""Resolve per-run secret values without ever placing one in argv."""

from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path

from osmosis_ai.cli.errors import CLIError


def _read_dotenv(source: str) -> dict[str, str]:
    """Read `NAME=value` pairs. ``-`` reads stdin, for piping from a manager."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text("utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if not separator:
            raise CLIError(f"Invalid line in secrets file: {stripped!r}")
        values[name.strip()] = value.strip()
    return values


def resolve_run_secrets(
    *,
    names: list[str],
    secrets_file: str | None,
    stored_names: set[str],
    is_tty: bool,
) -> dict[str, str]:
    """Values for ``names``, by first hit: secrets file, process environment,
    interactive prompt. A name already in the secret store is omitted so the
    platform resolves it. Outside a TTY every unresolved name is reported at
    once, so CI shows the whole gap rather than one name per retry.
    """
    from_file = _read_dotenv(secrets_file) if secrets_file else {}
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for name in names:
        if name in from_file:
            resolved[name] = from_file[name]
            continue
        env_value = os.environ.get(name)
        if env_value:
            resolved[name] = env_value
            continue
        if name in stored_names:
            continue
        if is_tty:
            resolved[name] = getpass(f"Value for {name}: ")
            continue
        missing.append(name)

    if missing:
        raise CLIError(
            "No value found for: "
            + ", ".join(missing)
            + ". Export them, pass --secrets-file, or save them with "
            "`osmosis secret set <NAME>`."
        )
    return resolved

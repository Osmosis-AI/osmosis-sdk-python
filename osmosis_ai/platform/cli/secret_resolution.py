"""Resolve per-run secret values without ever placing one in argv."""

from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import get_output_context


def _unquote(value: str) -> str:
    """Drop one matching pair of surrounding quotes, as dotenv files carry.

    Without this the quotes travel with the value and the run submits a secret
    that is wrong by two characters — which surfaces much later, as an opaque
    authentication failure inside a job that is already running.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _read_dotenv(source: str) -> dict[str, str]:
    """Read `NAME=value` pairs. ``-`` reads stdin, for piping from a manager.

    Errors report the location and never the line: every line in this file is
    a secret, and the CLI's JSON error envelope goes to stderr, which CI keeps.
    A `NAME=value` line whose name is not a plain identifier is rejected rather
    than stored under a name that could never be looked up.
    """
    text = sys.stdin.read() if source == "-" else Path(source).read_text("utf-8")
    label = "stdin" if source == "-" else source
    values: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not separator or not name.isidentifier():
            raise CLIError(
                f"Invalid line {lineno} in {label}: expected NAME=value. "
                "Values spanning multiple lines are not supported."
            )
        values[name] = _unquote(value.strip())
    return values


def resolve_run_secrets(
    *,
    names: list[str],
    secrets_file: str | None,
    stored_names: set[str],
) -> dict[str, str]:
    """Values for ``names``, by first hit: secrets file, process environment,
    interactive prompt. A name already in the secret store is omitted so the
    platform resolves it.

    Prompting is gated on ``get_output_context().interactive`` (rich + TTY),
    never on ``stdin.isatty()`` alone: ``--json`` / ``--plain`` on a developer
    terminal must not dead-end on ``getpass``. Outside an interactive session
    every unresolved name is reported at once as ``INTERACTIVE_REQUIRED``.
    Empty prompted values are rejected.
    """
    from_file = _read_dotenv(secrets_file) if secrets_file else {}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    interactive = get_output_context().interactive

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
        if interactive:
            value = getpass(f"Value for {name}: ")
            if not value:
                raise CLIError(
                    f"Secret value for {name} must not be empty.",
                    code="VALIDATION",
                )
            resolved[name] = value
            continue
        missing.append(name)

    if missing:
        raise CLIError(
            "No value found for: "
            + ", ".join(missing)
            + ". Export them, pass --secrets-file, or save them with "
            "`osmosis secret set <NAME>`.",
            code="INTERACTIVE_REQUIRED",
            details={"flags": ["--secrets-file"]},
        )
    return resolved

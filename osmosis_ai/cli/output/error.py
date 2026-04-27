"""Structured error envelope + PlatformAPIError to CLI code mapping."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.consts import PACKAGE_VERSION


def _classify_platform_status(status: int | None) -> str:
    if status == 401:
        return "AUTH_REQUIRED"
    if status == 404:
        return "NOT_FOUND"
    if status == 409:
        return "CONFLICT"
    if status == 429:
        return "RATE_LIMITED"
    if status == 400:
        return "VALIDATION"
    return "PLATFORM_ERROR"


def classify_error(exc: BaseException) -> CLIError:
    """Map any supported error type into a structured CLIError."""
    if isinstance(exc, CLIError):
        return exc

    from osmosis_ai.platform.auth.platform_client import (
        AuthenticationExpiredError,
        PlatformAPIError,
    )

    if isinstance(exc, AuthenticationExpiredError):
        return CLIError(str(exc) or "Session expired.", code="AUTH_REQUIRED")

    if isinstance(exc, PlatformAPIError):
        details: dict[str, Any] = {}
        if exc.error_code:
            details["platform_code"] = exc.error_code
        if exc.field:
            details["field"] = exc.field
        if exc.details:
            for key, value in exc.details.items():
                details.setdefault(key, value)
        if exc.status_code is not None:
            details["status_code"] = exc.status_code
        return CLIError(
            str(exc),
            code=_classify_platform_status(exc.status_code),
            details=details,
        )

    if isinstance(exc, click.UsageError):
        return CLIError(str(exc) or "Invalid usage.", code="VALIDATION")

    return CLIError(
        "An unexpected internal error occurred.",
        code="INTERNAL",
        details={"exception_type": type(exc).__name__},
    )


def _argv_command_path(argv: list[str]) -> str:
    skip_flags_with_value = {"--format"}
    skip_flags = {"--json", "--plain", "--version", "-V", "--help", "-h"}
    result: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in skip_flags_with_value:
            i += 2
            continue
        if any(token.startswith(f"{flag}=") for flag in skip_flags_with_value):
            i += 1
            continue
        if token in skip_flags:
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        result.append(token)
        if len(result) == 2:
            break
        i += 1
    return " ".join(result) if result else "<root>"


def command_path_for_error(ctx: click.Context | None) -> str:
    """Resolve the command path for the error envelope."""
    if ctx is not None:
        path = ctx.command_path
        parts = path.split(" ", 1)
        if len(parts) == 2:
            return parts[1]
        return path or "<root>"
    return _argv_command_path(sys.argv[1:] if sys.argv[1:] else [])


def emit_structured_error_to_stderr(
    err: CLIError,
    *,
    command: str | None = None,
    cli_version: str | None = None,
) -> None:
    """Write the JSON-mode error envelope to stderr."""
    if command is None:
        try:
            ctx = click.get_current_context(silent=True)
        except RuntimeError:
            ctx = None
        command = command_path_for_error(ctx)

    envelope: dict[str, Any] = {
        "schema_version": 1,
        "command": command,
        "cli_version": cli_version or PACKAGE_VERSION,
        "error": {
            "code": err.code,
            "message": err.message,
            "details": err.details,
            "request_id": err.request_id,
        },
    }
    sys.stderr.write(json.dumps(envelope, ensure_ascii=False))
    sys.stderr.write("\n")
    sys.stderr.flush()

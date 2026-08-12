"""Structured error envelope + PlatformAPIError to CLI code mapping."""

from __future__ import annotations

import sys
from typing import Any

from osmosis_ai.cli._click_compat import Context, UsageError, get_current_context
from osmosis_ai.cli.command_registry import (
    COMMAND_GROUPS,
    REMOVED_TOP_LEVEL_COMMANDS,
    REMOVED_TWO_TOKEN_COMMANDS,
    STANDALONE_COMMANDS,
    THREE_TOKEN_PREFIXES,
)
from osmosis_ai.cli.errors import CLIError, CLIErrorCode
from osmosis_ai.cli.output.jsonutil import dump_cli_json
from osmosis_ai.consts import PACKAGE_VERSION


def _classify_platform_status(status: int | None) -> CLIErrorCode:
    if status == 401:
        return CLIErrorCode.AUTH_REQUIRED
    if status == 404:
        return CLIErrorCode.NOT_FOUND
    if status == 409:
        return CLIErrorCode.CONFLICT
    if status == 426:
        return CLIErrorCode.UPGRADE_REQUIRED
    if status == 429:
        return CLIErrorCode.RATE_LIMITED
    if status == 400:
        return CLIErrorCode.VALIDATION
    return CLIErrorCode.PLATFORM_ERROR


def _platform_error_details(exc: Any) -> dict[str, Any]:
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
    return details


def classify_error(exc: BaseException) -> CLIError:
    """Map any supported error type into a structured CLIError."""
    if isinstance(exc, CLIError):
        return exc

    from osmosis_ai.platform.auth.platform_client import (
        AuthenticationExpiredError,
        PlatformAPIError,
        SubscriptionRequiredError,
    )

    if isinstance(exc, AuthenticationExpiredError):
        return CLIError(str(exc) or "Session expired.", code=CLIErrorCode.AUTH_REQUIRED)

    if isinstance(exc, SubscriptionRequiredError):
        code = (
            CLIErrorCode.BILLING_REQUIRED
            if exc.error_code == "BILLING_REQUIRED"
            else CLIErrorCode.SUBSCRIPTION_REQUIRED
        )
        return CLIError(str(exc), code=code, details=_platform_error_details(exc))

    if isinstance(exc, PlatformAPIError):
        return CLIError(
            str(exc),
            code=_classify_platform_status(exc.status_code),
            details=_platform_error_details(exc),
        )

    if isinstance(exc, UsageError):
        return CLIError(str(exc) or "Invalid usage.", code=CLIErrorCode.VALIDATION)

    return CLIError(
        "An unexpected internal error occurred.",
        code=CLIErrorCode.INTERNAL,
        details={"exception_type": type(exc).__name__},
    )


def emit_internal_debug(exc: BaseException, classified: CLIError | None = None) -> None:
    """Write original message + traceback for INTERNAL errors when debugging.

    ``OSMOSIS_DEBUG=1`` is the only switch (no CLI flag). The JSON error
    envelope is unchanged; this extra text is appended to stderr after it.
    Explicit ``CLIError(code="INTERNAL")`` values dump an attached cause or
    context, but never repeat the structured wrapper itself.
    """
    import os
    import traceback

    if os.environ.get("OSMOSIS_DEBUG") != "1":
        return
    debug_exc = exc
    if isinstance(exc, CLIError):
        if exc.code != CLIErrorCode.INTERNAL:
            return
        nested = exc.__cause__ or exc.__context__
        if nested is None:
            return
        debug_exc = nested
    else:
        if classified is not None and classified.code != CLIErrorCode.INTERNAL:
            return
        if classified is None and isinstance(exc, UsageError):
            return
    sys.stderr.write(f"{type(debug_exc).__name__}: {debug_exc}\n")
    traceback.print_exception(debug_exc, file=sys.stderr)
    sys.stderr.flush()


def _argv_command_path(argv: list[str]) -> str:
    skip_flags = {"--json", "--plain", "--version", "-V", "--help", "-h"}
    tokens: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in skip_flags:
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        tokens.append(token)
        i += 1
    if not tokens:
        return "<root>"

    command = tokens[0]
    if command in STANDALONE_COMMANDS or command in REMOVED_TOP_LEVEL_COMMANDS:
        return command
    if len(tokens) == 1:
        return command
    if (command, tokens[1]) in REMOVED_TWO_TOKEN_COMMANDS:
        return " ".join(tokens[:2])
    if len(tokens) >= 3 and (command, tokens[1]) in THREE_TOKEN_PREFIXES:
        return " ".join(tokens[:3])
    if command in COMMAND_GROUPS:
        return " ".join(tokens[:2])
    return command


def _click_subcommand_path(ctx: Context) -> str | None:
    path = ctx.command_path.strip()
    if not path:
        return None
    parts = path.split()
    if len(parts) >= 2:
        return " ".join(parts[1:])
    return None


def command_path_for_error(
    ctx: Context | None,
    *,
    argv: list[str] | None = None,
) -> str:
    """Resolve the command path for the error envelope.

    Prefer Click's ``command_path`` when the context is already inside a
    subcommand. Fall back to argv parsed against the same name catalog
    ``_register_commands`` uses.
    """
    if ctx is not None:
        click_path = _click_subcommand_path(ctx)
        if click_path is not None:
            return click_path
    if argv is not None:
        return _argv_command_path(argv)
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
            ctx = get_current_context(silent=True)
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
        },
    }
    try:
        serialized = dump_cli_json(envelope)
    except CLIError as exc:
        envelope["error"]["details"] = {"details_omitted": type(exc).__name__}
        serialized = dump_cli_json(envelope)
    sys.stderr.write(serialized)
    sys.stderr.write("\n")
    sys.stderr.flush()


def emit_structured_warning_to_stderr(
    message: str,
    *,
    code: str | None = None,
    cli_version: str | None = None,
) -> None:
    """Write a JSON-mode warning envelope (one line) to stderr.

    Non-fatal warnings share stderr with the error envelope but are
    distinguished by the top-level ``warning`` key (vs ``error``), so the stream
    stays parseable as JSON Lines. Unlike errors, warnings are not tied to a
    specific command (they originate from transport-level signals such as a
    deprecation response header), so no ``command`` field is emitted.
    """
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "cli_version": cli_version or PACKAGE_VERSION,
        "warning": {
            "code": code,
            "message": message,
        },
    }
    sys.stderr.write(dump_cli_json(envelope))
    sys.stderr.write("\n")
    sys.stderr.flush()

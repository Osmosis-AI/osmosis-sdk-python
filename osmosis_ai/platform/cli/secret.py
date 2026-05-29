"""Business logic for ``osmosis secret`` (list + add).

Security invariants for this module:
  * The secret *value* travels through this process only to be POSTed once.
    It is never logged, never written to disk, and never placed in a
    ``CommandResult``, an error message, or an error ``details`` payload.
  * The value is accepted from a hidden interactive prompt or a named
    environment variable — never from a plaintext command-line argument.
  * Local validation operates on the secret *name* only; the value is checked
    solely for emptiness so we never embed it in a validation message.
"""

from __future__ import annotations

import os
import re
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    ListColumn,
    ListResult,
    OperationResult,
    OutputFormat,
    get_output_context,
    serialize_environment_secret,
)
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.cli.utils import (
    fetch_all_pages,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

# Mirror the platform's ``environmentSecretNameSchema`` so we fail fast with a
# clear local message instead of round-tripping an obviously invalid name —
# and so the name (never the value) is the only thing a validation error
# can reference.
_SECRET_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SECRET_NAME_MAX = 255


def _validate_secret_name(name: str) -> None:
    if not name or len(name) > _SECRET_NAME_MAX:
        raise CLIError(
            "Secret name must be between 1 and 255 characters.",
            code="VALIDATION",
        )
    if not _SECRET_NAME_RE.fullmatch(name):
        raise CLIError(
            "Secret name may only contain letters, digits, '_' and '-'.",
            code="VALIDATION",
        )


def _resolve_secret_value(*, env: str | None, output: Any) -> str | None:
    """Resolve the secret value from ``--env`` or a hidden interactive prompt.

    Returns the value, or ``None`` if the user cancels the interactive prompt.
    Never echoes, logs, or embeds the value in an error. The value is read
    verbatim (no whitespace/newline trimming) so secrets that legitimately
    contain surrounding whitespace are preserved.
    """
    if env is not None:
        # Read from a named environment variable: only the variable *name*
        # ever appears on the command line, never the value.
        if not env.strip():
            raise CLIError(
                "--env requires an environment variable name.",
                code="VALIDATION",
            )
        value = os.environ.get(env)
        if value is None:
            raise CLIError(
                f"Environment variable {env} is not set.",
                code="VALIDATION",
            )
        if value == "":
            raise CLIError(
                f"Environment variable {env} is empty.",
                code="VALIDATION",
            )
        return value

    # No --env: a hidden interactive prompt is the only other accepted source.
    # In non-interactive modes (``--json`` / ``--plain`` or a non-TTY stdin)
    # there is no safe way to obtain the value, so require an explicit flag.
    if output.format is not OutputFormat.rich or not output.interactive:
        raise CLIError(
            "Secret value required. Run interactively to type it at a hidden "
            "prompt, or pass --env VARNAME to read it from an environment "
            "variable.",
            code="INTERACTIVE_REQUIRED",
            details={"flags": ["--env"]},
        )

    from osmosis_ai.cli.prompts import password

    value = password("Secret value (input hidden)")
    if value is None:
        return None  # user cancelled (Ctrl+C / ESC)
    if value == "":
        raise CLIError("Secret value must not be empty.", code="VALIDATION")
    return value


def _redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    if isinstance(data, str):
        return data.replace(secret_value, "[REDACTED]")
    if isinstance(data, dict):
        return {
            key: (
                "[REDACTED]"
                if str(key).lower() in {"value", "secret", "secret_value"}
                else _redact_secret_value(value, secret_value)
            )
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_redact_secret_value(value, secret_value) for value in data]
    return data


def _redact_secret_platform_error(exc: Any, secret_value: str) -> Any:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    return PlatformAPIError(
        _redact_secret_value(str(exc), secret_value),
        status_code=exc.status_code,
        error_code=exc.error_code,
        field=exc.field,
        details=_redact_secret_value(exc.details, secret_value),
    )


def list_secrets(*, limit: int, all_: bool) -> ListResult:
    """List workspace secrets (names + metadata only; values are never returned)."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()

    # ``fetch_all_pages`` discards the page object, so capture the page-level
    # ``platform_url`` from the first response via a closure.
    captured: dict[str, str | None] = {}

    def _fetch(lim: int, off: int) -> Any:
        page = client.list_environment_secrets(
            limit=lim,
            offset=off,
            credentials=credentials,
            git_identity=context.git_identity,
        )
        captured.setdefault("platform_url", page.platform_url)
        return page

    with output.status("Fetching secrets..."):
        if fetch_all:
            secrets, total_count = fetch_all_pages(
                _fetch, items_attr="environment_secrets"
            )
            has_more = False
            next_offset: int | None = None
        else:
            page = _fetch(effective_limit, 0)
            secrets = page.environment_secrets
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    platform_url = captured.get("platform_url")

    extra = git_result_context(context)
    display_hints: list[str] = []
    if platform_url:
        extra["platform_url"] = platform_url
        display_hints.append(f"View secrets: {platform_url}")

    return ListResult(
        title="Secrets",
        items=[serialize_environment_secret(s) for s in secrets],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=extra,
        columns=[
            ListColumn(key="name", label="Name", ratio=3, overflow="fold"),
            ListColumn(key="created_at", label="Added", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Added by", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_environment_secret(s),
                "created_at": format_local_date(s.created_at),
                "creator_name": s.creator_name or "—",
            }
            for s in secrets
        ],
        display_hints=display_hints,
    )


def add_secret(*, name: str, env: str | None) -> OperationResult:
    """Add a workspace secret.

    The value is read from ``--env VARNAME`` or a hidden interactive prompt,
    POSTed once, and immediately discarded. It never appears in the result.
    """
    _validate_secret_name(name)

    context = require_git_workspace_directory_context()
    output = get_output_context()

    value = _resolve_secret_value(env=env, output=output)
    if value is None:
        return OperationResult(
            operation="secret.add",
            status="cancelled",
            resource=git_result_context(context),
            message="Cancelled.",
        )

    client = OsmosisClient()
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    try:
        with output.status(f'Adding secret "{name}"...'):
            secret = client.create_environment_secret(
                name,
                value,
                credentials=context.credentials,
                git_identity=context.git_identity,
            )
    except PlatformAPIError as exc:
        raise _redact_secret_platform_error(exc, value) from exc
    finally:
        # Drop the only remaining reference to the plaintext value promptly,
        # including the failure path.
        del value

    resource = serialize_environment_secret(secret)
    resource.update(git_result_context(context))

    extra: dict[str, Any] = {}
    display_next_steps: list[str] = []
    next_steps_structured: list[dict[str, Any]] = []
    if secret.platform_url:
        extra["platform_url"] = secret.platform_url
        display_next_steps.append(f"View secrets: {secret.platform_url}")
        next_steps_structured.append(
            {"action": "view_secrets", "url": secret.platform_url}
        )

    return OperationResult(
        operation="secret.add",
        status="success",
        resource=resource,
        message=f'Secret "{name}" added.',
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
        extra=extra,
    )

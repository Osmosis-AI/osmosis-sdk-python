"""Business logic for ``osmosis secret`` (set, list, delete).

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
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import WIRE_SCOPE_PERSONAL
from osmosis_ai.platform.cli.shared_config import SECRET_NAME_RE
from osmosis_ai.platform.cli.utils import (
    fetch_all_pages,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

# Mirror the platform's ``environmentSecretNameSchema`` (^[A-Z][A-Z0-9_]*$):
# uppercase env-var style so a referencing config's [env]-style name and the
# secret record name are the same shape. Validating fails fast locally and,
# because only the name (never the value) is referenced, can never leak a value.
_SECRET_NAME_RE = SECRET_NAME_RE
_SECRET_NAME_MAX = 255

# User-facing scope vocabulary is "personal"/"workspace"; the platform wire
# value for a personal secret is "user", so translate at the API boundary.
_VALID_SCOPES = ("workspace", "personal")
_VALID_LIST_SCOPES = ("all", "workspace", "personal")
_SCOPE_TO_WIRE = {
    "all": "all",
    "workspace": "workspace",
    "personal": WIRE_SCOPE_PERSONAL,
}

# Human-facing labels for the scope column (the raw wire values stay in JSON output).
_SCOPE_DISPLAY = {"workspace": "Workspace", WIRE_SCOPE_PERSONAL: "Personal"}


def _validate_secret_name(name: str) -> None:
    if not name or len(name) > _SECRET_NAME_MAX:
        raise CLIError(
            "Secret name must be between 1 and 255 characters.",
            code="VALIDATION",
        )
    if not _SECRET_NAME_RE.fullmatch(name):
        raise CLIError(
            "Secret name must match ^[A-Z][A-Z0-9_]*$ "
            "(uppercase letters, digits and '_', starting with a letter).",
            code="VALIDATION",
        )


def _validate_scope(scope: str) -> None:
    if scope not in _VALID_SCOPES:
        raise CLIError(
            "Scope must be 'workspace' or 'personal'.",
            code="VALIDATION",
        )


def _require_value_source_available(env: str | None, output: Any) -> None:
    """Fail fast when no usable secret-value source exists.

    Without ``--env``, a hidden interactive prompt is the only accepted source.
    In non-interactive modes (``--json`` / ``--plain`` or a non-TTY stdin) there
    is no safe way to obtain the value, so require an explicit flag. Raising here
    lets ``set_secret`` reject before any network/workspace call (the message and
    ``details`` stay in sync with the equivalent guard in
    ``_resolve_secret_value``).
    """
    if env is None and (
        output.format is not OutputFormat.rich or not output.interactive
    ):
        raise CLIError(
            "Secret value required. Run interactively to type it at a hidden "
            "prompt, or pass --env VARNAME to read it from an environment "
            "variable.",
            code="INTERACTIVE_REQUIRED",
            details={"flags": ["--env"]},
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
    _require_value_source_available(env, output)

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


def _redact_secret_text(value: str | None, secret_value: str) -> str | None:
    if value is None:
        return None
    return value.replace(secret_value, "[REDACTED]")


def _redact_secret_platform_error(exc: Any, secret_value: str) -> Any:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    return PlatformAPIError(
        _redact_secret_text(str(exc), secret_value) or "",
        status_code=exc.status_code,
        error_code=_redact_secret_text(exc.error_code, secret_value),
        field=_redact_secret_text(exc.field, secret_value),
        details=_redact_secret_value(exc.details, secret_value),
    )


def list_secrets(*, limit: int, all_: bool, scope: str = "all") -> ListResult:
    """List secrets (names + metadata only; values never shown).

    ``scope`` is ``"all"`` (workspace + the caller's personal secrets),
    ``"workspace"``, or ``"personal"`` (personal only).
    """
    if scope not in _VALID_LIST_SCOPES:
        raise CLIError(
            "Scope must be 'all', 'workspace', or 'personal'.",
            code="VALIDATION",
        )
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials

    client = OsmosisClient()
    output = get_output_context()

    # ``paginated_fetch`` returns only items + cursor fields, discarding the page
    # object, so capture the page-level ``platform_url`` from the first response
    # via a closure (``_fetch`` is invoked on every branch of ``paginated_fetch``).
    captured: dict[str, str | None] = {}

    def _fetch(lim: int, off: int) -> Any:
        page = client.list_environment_secrets(
            limit=lim,
            offset=off,
            scope=_SCOPE_TO_WIRE[scope],
            credentials=credentials,
            git_identity=context.git_identity,
        )
        captured.setdefault("platform_url", page.platform_url)
        return page

    with output.status("Fetching secrets..."):
        secrets, total_count, has_more, next_offset = paginated_fetch(
            _fetch,
            items_attr="environment_secrets",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

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
            ListColumn(key="scope", label="Scope", no_wrap=True, ratio=1),
            ListColumn(key="updated_at", label="Updated", no_wrap=True, ratio=1),
            ListColumn(key="updater_name", label="Updated By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_environment_secret(s),
                "scope": _SCOPE_DISPLAY.get(s.scope, "—"),
                "updated_at": format_local_date(s.updated_at),
                "updater_name": s.updater_name or "—",
            }
            for s in secrets
        ],
        display_hints=display_hints,
    )


def set_secret(
    *, name: str, scope: str = "personal", env: str | None = None
) -> OperationResult:
    """Create or update (upsert) a secret.

    The value is read from ``--env VARNAME`` or a hidden interactive prompt,
    POSTed once, and immediately discarded. It never appears in the result.
    """
    _validate_secret_name(name)
    _validate_scope(scope)

    output = get_output_context()
    # Reject before any network/workspace call when no value source is usable.
    _require_value_source_available(env, output)

    context = require_git_workspace_directory_context()

    value = _resolve_secret_value(env=env, output=output)
    if value is None:
        return OperationResult(
            operation="secret.set",
            status="cancelled",
            resource=git_result_context(context),
            message="Cancelled.",
        )

    client = OsmosisClient()
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    try:
        with output.status(f'Saving secret "{name}"...'):
            secret = client.set_environment_secret(
                name,
                value,
                scope=_SCOPE_TO_WIRE[scope],
                credentials=context.credentials,
                git_identity=context.git_identity,
            )
    except PlatformAPIError as exc:
        raise _redact_secret_platform_error(exc, value) from None
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
        operation="secret.set",
        status="success",
        resource=resource,
        message=f'Secret "{name}" saved in {scope} scope.',
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
        extra=extra,
    )


def _existing_secret_names(
    client: OsmosisClient,
    *,
    scope: str,
    credentials: Any,
    git_identity: str,
) -> set[str] | None:
    """Names that exist in ``scope`` (a wire value). ``None`` if the lookup fails,
    so a transient error falls back to letting the platform validate."""
    try:

        def _fetch(limit: int, offset: int) -> Any:
            return client.list_environment_secrets(
                limit=limit,
                offset=offset,
                scope=scope,
                credentials=credentials,
                git_identity=git_identity,
            )

        secrets, _ = fetch_all_pages(_fetch, items_attr="environment_secrets")
        return {s.name for s in secrets}
    except Exception:
        return None


def delete_secret(
    *, name: str, scope: str = "personal", yes: bool = False
) -> OperationResult:
    """Delete a secret by name within ``scope``.

    ``scope`` is ``"workspace"`` or ``"personal"``. Requires confirmation
    (``--yes`` to skip). Workspace deletion is additionally role-gated
    server-side (admin/owner).
    """
    _validate_secret_name(name)
    _validate_scope(scope)

    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    wire_scope = _SCOPE_TO_WIRE[scope]

    # Fail before prompting if the secret doesn't exist in the requested scope,
    # so a typo or wrong --scope is reported immediately instead of after a Y/n.
    existing = _existing_secret_names(
        client,
        scope=wire_scope,
        credentials=context.credentials,
        git_identity=context.git_identity,
    )
    if existing is not None and name not in existing:
        other_scope = "personal" if scope == "workspace" else "workspace"
        other_wire = _SCOPE_TO_WIRE[other_scope]
        other_names = _existing_secret_names(
            client,
            scope=other_wire,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
        if other_names is not None and name in other_names:
            raise CLIError(
                f'Secret "{name}" not found in {scope} scope.'
                f"\nDid you mean: osmosis secret delete {name}"
                f" --scope {other_scope}",
                code="NOT_FOUND",
            )
        raise CLIError(
            f'Secret "{name}" not found.',
            code="NOT_FOUND",
        )

    require_confirmation(
        f'Delete {scope} secret "{name}"? This cannot be undone.',
        yes=yes,
        default=False,
        summary=[("Name", name), ("Scope", scope)],
    )

    with get_output_context().status(f'Deleting secret "{name}"...'):
        client.delete_environment_secret(
            name,
            scope=wire_scope,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    resource: dict[str, Any] = {"name": name, "scope": scope}
    resource.update(git_result_context(context))
    return OperationResult(
        operation="secret.delete",
        status="success",
        resource=resource,
        message=f'Secret "{name}" deleted from {scope} scope.',
    )

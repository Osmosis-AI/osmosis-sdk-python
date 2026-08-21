"""Authentication handlers: login, logout, whoami."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailField,
    DetailResult,
    OperationResult,
    OutputFormat,
    get_output_context,
)

_AUTH_LOGIN_ERROR_CODES = {
    "AUTH_HEADER_MISSING",
    "TOKEN_MISSING",
    "TOKEN_EXPIRED",
    "TOKEN_INVALID",
    "TOKEN_REVOKED",
    "UNKNOWN_AUTH_ERROR",
}
_RECOVERABLE_CREDENTIAL_METADATA_CODES = {
    "CREDENTIALS_PARSE_FAILED",
    "CREDENTIALS_VERSION_CHANGED",
}
_PLATFORM_CREATE_REPO_STEP = (
    "Create or open a workspace in the Osmosis Platform, then clone the repository "
    "created there."
)
_DOCTOR_CLONE_STEP = "From the workspace directory, run `osmosis doctor`."


def _login_operation_result(
    *,
    email: str,
    name: str | None,
    expires_at: Any,
    source: str,
    saved: bool,
    token_store: str | None = None,
) -> OperationResult:
    """Build the structured login result."""
    from osmosis_ai.platform.auth.config import get_platform_url

    resource: dict[str, Any] = {
        "email": email,
        "name": name,
        "expires_at": expires_at.isoformat(),
        "platform_url": get_platform_url(),
        "workspace": None,
        "source": source,
        "verified": True,
        "saved": saved,
    }
    if token_store is not None:
        resource["token_store"] = token_store

    next_steps: list[str] = []
    next_steps_structured: list[dict[str, Any]] = []
    if saved:
        next_steps.extend([_PLATFORM_CREATE_REPO_STEP, _DOCTOR_CLONE_STEP])
        next_steps_structured.extend(
            [
                {
                    "action": "platform.clone_repository",
                    "description": _PLATFORM_CREATE_REPO_STEP,
                },
                {
                    "action": "doctor",
                    "description": _DOCTOR_CLONE_STEP,
                },
            ]
        )

    return OperationResult(
        operation="auth.login",
        status="success",
        resource=resource,
        message=f"Logged in as {email}.",
        display_next_steps=next_steps,
        next_steps_structured=next_steps_structured,
    )


def _verify_env_token(env_token: str, *, git_identity: str | None = None) -> Any:
    """Verify OSMOSIS_TOKEN and replace generic 401s with actionable guidance."""
    from osmosis_ai.platform.auth import LoginError, verify_token
    from osmosis_ai.platform.constants import (
        MSG_ENV_TOKEN_EXPIRED,
        MSG_ENV_TOKEN_INVALID,
        MSG_ENV_TOKEN_REVOKED,
    )

    env_messages = {
        "AUTH_HEADER_MISSING": MSG_ENV_TOKEN_INVALID,
        "TOKEN_MISSING": MSG_ENV_TOKEN_INVALID,
        "TOKEN_EXPIRED": MSG_ENV_TOKEN_EXPIRED,
        "TOKEN_INVALID": MSG_ENV_TOKEN_INVALID,
        "TOKEN_REVOKED": MSG_ENV_TOKEN_REVOKED,
        "UNKNOWN_AUTH_ERROR": MSG_ENV_TOKEN_INVALID,
    }

    try:
        return verify_token(env_token, git_identity=git_identity)
    except LoginError as exc:
        code = exc.code
        message = env_messages.get(code) if code is not None else None
        if message is None and exc.status_code == 401:
            message = MSG_ENV_TOKEN_INVALID
        if message is not None:
            raise LoginError(message, code=code, status_code=exc.status_code) from exc
        raise


def _is_auth_login_error(exc: Any) -> bool:
    """True when a LoginError signals an invalid/expired/revoked session."""
    platform_code = getattr(exc, "code", None)
    return getattr(exc, "status_code", None) == 401 or (
        isinstance(platform_code, str) and platform_code in _AUTH_LOGIN_ERROR_CODES
    )


def _verify_with_optional_workspace(verify: Any, *, git_identity: str | None) -> Any:
    """Verify a session, enriching it with the linked workspace when possible.

    ``git_identity`` only adds the linked workspace (resolved via the
    X-Osmosis-Git header) to the result. A repo/workspace scope mismatch must
    not stop identity resolution, so any non-auth failure retries without the
    git identity. Genuine auth failures (revoked/expired/invalid) still
    propagate so callers can surface them.
    """
    from osmosis_ai.platform.auth import LoginError

    if git_identity is None:
        return verify(git_identity=None)
    try:
        return verify(git_identity=git_identity)
    except LoginError as exc:
        if _is_auth_login_error(exc):
            raise
        return verify(git_identity=None)


def _cli_error_from_login_error(exc: Any) -> CLIError:
    status_code = getattr(exc, "status_code", None)
    platform_code = getattr(exc, "code", None)
    if _is_auth_login_error(exc):
        return CLIError(str(exc), code="AUTH_REQUIRED")

    if status_code == 426:
        details: dict[str, Any] = {"status_code": 426}
        # Mirror the structured signal (status/message) a 426 carries on a
        # regular API call so the UPGRADE_REQUIRED envelope looks the same
        # whether it came from the login handshake or any other command.
        signal = getattr(exc, "details", None)
        if isinstance(signal, dict):
            for key, value in signal.items():
                details.setdefault(key, value)
        return CLIError(str(exc), code="UPGRADE_REQUIRED", details=details)

    details: dict[str, Any] = {}
    if isinstance(status_code, int):
        details["status_code"] = status_code
    if isinstance(platform_code, str):
        details["platform_code"] = platform_code
    return CLIError(str(exc), code="PLATFORM_ERROR", details=details)


def _finish_login(
    *,
    creds: Any,
    old_credentials: Any | None,
    result: Any,
    source: str,
    output: Any,
) -> OperationResult:
    """Replace the stored session with *creds* and report the outcome.

    Shared by the token and device-code paths, which differ only in how they
    obtain *creds*. Saves the replacement before revoking the old CLI token,
    so a failed local save leaves the previous login usable.

    Imports stay inside the body: callers patch the source modules, and
    hoisting them would also undo the deferred-import CLI startup work.
    """
    from osmosis_ai.platform.auth.credentials import _cleanup_replaced_credentials
    from osmosis_ai.platform.auth.flow import LoginError, verify_token
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    same_server_token = bool(
        old_credentials is not None
        and (
            old_credentials.access_token == creds.access_token
            or (
                old_credentials.token_id
                and creds.token_id
                and old_credentials.token_id == creds.token_id
            )
        )
    )
    old_credentials_to_revoke = (
        old_credentials
        if old_credentials is not None
        and not old_credentials.is_expired()
        and not same_server_token
        else None
    )
    if source == "device":
        token_store = save_device_credentials_or_revoke(
            creds,
            output=output,
            cleanup_replaced=False,
        )
    else:
        from osmosis_ai.platform.auth.credentials import save_credentials

        token_store = save_credentials(
            creds,
            cleanup_replaced=False,
            recover_invalid_metadata=True,
        )

    if old_credentials_to_revoke is not None:
        if not old_credentials_to_revoke.token_id:
            try:
                with output.status("Resolving old session..."):
                    verified_old = verify_token(old_credentials_to_revoke.access_token)
            except LoginError as exc:
                if _is_auth_login_error(exc):
                    old_credentials_to_revoke = None
                else:
                    from osmosis_ai.cli.console import console

                    console.print_warning(
                        "The new login is active, but the previous session "
                        "could not be checked or revoked.",
                        code="TOKEN_REVOKE_FAILED",
                    )
            else:
                old_credentials_to_revoke = replace(
                    old_credentials_to_revoke, token_id=verified_old.token_id
                )
                if verified_old.token_id and verified_old.token_id == creds.token_id:
                    old_credentials_to_revoke = None
                elif not verified_old.token_id:
                    from osmosis_ai.cli.console import console

                    console.print_warning(
                        "The new login is active, but the previous session "
                        "did not expose a token ID and could not be revoked.",
                        code="TOKEN_REVOKE_FAILED",
                    )

        if old_credentials_to_revoke is not None and old_credentials_to_revoke.token_id:
            with output.status("Revoking old session..."):
                revoke_cli_token(old_credentials_to_revoke)

    if old_credentials is not None:
        _cleanup_replaced_credentials(old_credentials, creds)

    return _login_operation_result(
        email=result.user.email,
        name=result.user.name,
        expires_at=result.expires_at,
        source=source,
        saved=True,
        token_store=token_store,
    )


def save_device_credentials_or_revoke(
    credentials: Any,
    *,
    output: Any | None = None,
    cleanup_replaced: bool = True,
) -> str:
    """Persist a new device-login token or revoke it before re-raising."""
    from osmosis_ai.cli.console import console
    from osmosis_ai.platform.auth.credentials import save_credentials
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    try:
        return save_credentials(
            credentials,
            cleanup_replaced=cleanup_replaced,
            recover_invalid_metadata=True,
        )
    except Exception:
        revoked = False
        try:
            if output is None:
                revoked = revoke_cli_token(credentials, warn_on_failure=False)
            else:
                with output.status("Revoking new session..."):
                    revoked = revoke_cli_token(credentials, warn_on_failure=False)
        except Exception:
            revoked = False
        if not revoked:
            console.print_warning(
                "The new device-login token could not be saved or revoked.",
                code="TOKEN_REVOKE_FAILED",
            )
        raise


def _login_with_token(*, token: str) -> OperationResult:
    """Verify and persist an explicit token, returning structured output."""
    from osmosis_ai.platform.auth import load_credentials, verify_token
    from osmosis_ai.platform.auth.credentials import Credentials
    from osmosis_ai.platform.auth.flow import LoginResult

    output = get_output_context()
    try:
        old_credentials = load_credentials(include_env=False)
    except CLIError as exc:
        if exc.code not in _RECOVERABLE_CREDENTIAL_METADATA_CODES:
            raise
        old_credentials = None

    with output.status("Verifying token..."):
        verified = verify_token(token)
    creds = Credentials.from_verify_result(token, verified)
    result = LoginResult.from_verify_result(verified)

    return _finish_login(
        creds=creds,
        old_credentials=old_credentials,
        result=result,
        source="token",
        output=output,
    )


def _login_with_device_flow() -> OperationResult:
    """Interactive device-code login. Caller must have already gated on interactivity."""
    from osmosis_ai.platform.auth import device_login, load_credentials
    from osmosis_ai.platform.auth.credentials import ensure_keyring_available

    output = get_output_context()
    ensure_keyring_available()
    try:
        old_credentials = load_credentials(include_env=False)
    except CLIError as exc:
        if exc.code not in _RECOVERABLE_CREDENTIAL_METADATA_CODES:
            raise
        old_credentials = None
    result, creds = device_login()

    return _finish_login(
        creds=creds,
        old_credentials=old_credentials,
        result=result,
        source="device",
        output=output,
    )


def login(*, token: str | None = None) -> OperationResult:
    """Authenticate with Osmosis AI."""
    from osmosis_ai.platform.auth import LoginError

    try:
        if token is not None:
            return _login_with_token(token=token)

        if os.environ.get("OSMOSIS_TOKEN"):
            raise CLIError(
                "OSMOSIS_TOKEN is already active for this process. "
                "Skip 'auth login' for CI/CD, or unset it before logging in.",
                code="CONFLICT",
            )

        output = get_output_context()
        if output.format is not OutputFormat.rich:
            raise CLIError(
                "Login is interactive in this mode. Pass --token to persist a PAT; "
                "for CI/CD, set OSMOSIS_TOKEN and skip 'auth login'.",
                code="INTERACTIVE_REQUIRED",
            )
        return _login_with_device_flow()
    except LoginError as exc:
        raise _cli_error_from_login_error(exc) from exc


def logout(*, skip_confirm: bool = False) -> OperationResult:
    """Logout from Osmosis AI CLI."""
    from osmosis_ai.cli.prompts import confirm
    from osmosis_ai.platform.auth import get_credential_store, load_credentials
    from osmosis_ai.platform.auth.config import get_platform_url
    from osmosis_ai.platform.auth.local_config import reset_session
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    output = get_output_context()
    env_token_set = bool(os.environ.get("OSMOSIS_TOKEN"))
    try:
        persistent_store = get_credential_store(include_env=False)
        credentials = (
            load_credentials(include_env=False)
            if persistent_store is not None
            else None
        )
    except CLIError as exc:
        if exc.code in _RECOVERABLE_CREDENTIAL_METADATA_CODES:
            persistent_store = "unknown"
            credentials = None
        elif exc.code == "KEYRING_UNAVAILABLE":
            persistent_store = "keyring"
            credentials = None
        else:
            raise

    if persistent_store is None:
        if env_token_set:
            return OperationResult(
                operation="auth.logout",
                status="noop",
                resource={
                    "logged_in": True,
                    "env_token_set": True,
                    "persistent_login": False,
                    "effective_source": "environment",
                    "platform_url": get_platform_url(),
                },
                message="OSMOSIS_TOKEN is set. Run 'unset OSMOSIS_TOKEN' to logout.",
                next_steps_structured=[
                    {"action": "unset_env", "name": "OSMOSIS_TOKEN"}
                ],
            )
        return OperationResult(
            operation="auth.logout",
            status="noop",
            resource={
                "logged_in": False,
                "persistent_login": False,
                "effective_source": None,
                "platform_url": get_platform_url(),
            },
            message="Not logged in.",
        )

    if not skip_confirm:
        if output.format is not OutputFormat.rich or not output.interactive:
            raise CLIError(
                "Use --yes to confirm in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
            )
        confirmed = confirm("Logout from Osmosis AI?", default=False)
        if not confirmed:
            return OperationResult(
                operation="auth.logout",
                status="cancelled",
                resource={
                    "logged_in": True,
                    "persistent_login": True,
                    "effective_source": (
                        "environment" if env_token_set else persistent_store
                    ),
                    "platform_url": get_platform_url(),
                },
                message="Cancelled.",
            )

    revoked = False
    if credentials is not None and not credentials.is_expired():
        with output.status("Revoking session..."):
            revoked = revoke_cli_token(credentials)

    reset_session(
        recover_invalid_credentials=True,
        tolerate_keyring_unavailable=True,
    )
    effective_source = "environment" if env_token_set else None

    return OperationResult(
        operation="auth.logout",
        status="success",
        resource={
            "logged_in": env_token_set,
            "revoked": revoked,
            "env_token_set": env_token_set,
            "persistent_login": False,
            "effective_source": effective_source,
            "platform_url": get_platform_url(),
        },
        message=(
            "Saved login removed; OSMOSIS_TOKEN is still active."
            if env_token_set
            else "Logged out successfully."
        ),
        display_next_steps=(
            ["Unset OSMOSIS_TOKEN to fully logout."] if env_token_set else []
        ),
        next_steps_structured=(
            [{"action": "unset_env", "name": "OSMOSIS_TOKEN"}] if env_token_set else []
        ),
    )


def whoami() -> DetailResult:
    """Show current authenticated user."""
    from osmosis_ai.cli.console import console
    from osmosis_ai.platform.auth import (
        AuthenticationExpiredError,
        Credentials,
        LoginError,
        get_credential_store,
        load_credentials,
        verify_token,
    )
    from osmosis_ai.platform.auth.config import get_platform_url
    from osmosis_ai.platform.cli.workspace_directory_context import (
        resolve_optional_git_identity,
    )
    from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

    output = get_output_context()
    env_token = os.environ.get("OSMOSIS_TOKEN")
    source = "environment" if env_token else "credentials"
    persistent_store = None
    persistent_login = False
    effective_source = "environment" if env_token else None
    git_identity = resolve_optional_git_identity()

    if env_token:
        try:
            with output.status("Verifying token..."):
                verified = _verify_with_optional_workspace(
                    lambda *, git_identity: _verify_env_token(
                        env_token, git_identity=git_identity
                    ),
                    git_identity=git_identity,
                )
        except LoginError as exc:
            raise _cli_error_from_login_error(exc) from exc
        credentials = Credentials.from_verify_result(env_token, verified)
        try:
            persistent_store = get_credential_store(include_env=False)
        except CLIError:
            persistent_store = None
        persistent_login = persistent_store is not None
    else:
        persistent_store = get_credential_store(include_env=False)
        credentials = load_credentials()
        if credentials is None:
            raise CLIError(MSG_NOT_LOGGED_IN, code="AUTH_REQUIRED")
        persistent_login = True
        effective_source = persistent_store or "credentials"
        if credentials.is_expired():
            raise AuthenticationExpiredError()
        access_token = credentials.access_token
        try:
            with output.status("Verifying session..."):
                verified = _verify_with_optional_workspace(
                    lambda *, git_identity: verify_token(
                        access_token, git_identity=git_identity
                    ),
                    git_identity=git_identity,
                )
        except LoginError as exc:
            platform_code = getattr(exc, "code", None)
            if getattr(exc, "status_code", None) == 401 or (
                isinstance(platform_code, str)
                and platform_code in _AUTH_LOGIN_ERROR_CODES
            ):
                message = (
                    "Your session has been revoked. "
                    "Please run 'osmosis auth login' to re-authenticate."
                    if platform_code == "TOKEN_REVOKED"
                    else None
                )
                if message is not None:
                    raise AuthenticationExpiredError(message) from exc
                raise AuthenticationExpiredError() from exc
            raise _cli_error_from_login_error(exc) from exc
        credentials = Credentials.from_verify_result(credentials.access_token, verified)

    workspace = (
        {
            "id": verified.workspace.id,
            "name": verified.workspace.name,
            "role": verified.workspace.role,
        }
        if verified.workspace is not None
        else None
    )

    account = {
        "email": credentials.user.email,
        "name": credentials.user.name,
        "expires_at": credentials.expires_at.isoformat(),
        "platform_url": get_platform_url(),
        "source": source,
        "effective_source": effective_source,
        "persistent_login": persistent_login,
    }
    if credentials.token_id:
        account["token_id"] = credentials.token_id
    data = {
        "account": account,
        "email": credentials.user.email,
        "name": credentials.user.name,
        "expires_at": credentials.expires_at.isoformat(),
        "platform_url": get_platform_url(),
        "source": source,
        "effective_source": effective_source,
        "persistent_login": persistent_login,
        "workspace": workspace,
    }

    fields = [
        DetailField(
            label="Email", value=console.format_text(credentials.user.email or "")
        )
    ]
    if credentials.user.name:
        fields.append(
            DetailField(label="Name", value=console.format_text(credentials.user.name))
        )
    if workspace is not None:
        fields.append(
            DetailField(
                label="Workspace", value=console.format_text(workspace["name"] or "")
            )
        )
        if workspace["role"]:
            fields.append(
                DetailField(label="Role", value=console.format_text(workspace["role"]))
            )
    fields.append(DetailField(label="Platform", value=get_platform_url()))
    fields.append(DetailField(label="Auth Source", value=effective_source or source))
    fields.append(
        DetailField(
            label="Saved Login",
            value=(
                f"yes ({persistent_store})" if persistent_store is not None else "no"
            ),
        )
    )
    fields.append(
        DetailField(label="Expires", value=credentials.expires_at.strftime("%Y-%m-%d"))
    )

    return DetailResult(title="Account", data=data, fields=fields)


__all__ = ["login", "logout", "save_device_credentials_or_revoke", "whoami"]

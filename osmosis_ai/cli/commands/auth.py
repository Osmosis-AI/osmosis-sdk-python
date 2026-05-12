"""Authentication commands: login, logout, whoami."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials

app: typer.Typer = typer.Typer(
    help="Manage authentication (login, logout, whoami).", no_args_is_help=True
)


ASCII_ART = r"""
                       ___           ___           ___           ___           ___                       ___
            ___       /\  \         /\  \         /\__\         /\  \         /\  \          ___        /\  \
      __   /\__\     /::\  \       /::\  \       /::|  |       /::\  \       /::\  \        /\  \      /::\  \
    /\__\  \/__/    /:/\:\  \     /:/\ \  \     /:|:|  |      /:/\:\  \     /:/\ \  \       \:\  \    /:/\ \  \
   /:/  /  /\__\   /:/  \:\  \   _\:\~\ \  \   /:/|:|__|__   /:/  \:\  \   _\:\~\ \  \      /::\__\  _\:\~\ \  \
  /:/  /  /:/  /  /:/__/ \:\__\ /\ \:\ \ \__\ /:/ |::::\__\ /:/__/ \:\__\ /\ \:\ \ \__\  __/:/\/__/ /\ \:\ \ \__\
  \/__/  /:/  /   \:\  \ /:/  / \:\ \:\ \/__/ \/__/~~/:/  / \:\  \ /:/  / \:\ \:\ \/__/ /\/:/  /    \:\ \:\ \/__/
  /\__\  \/__/     \:\  /:/  /   \:\ \:\__\         /:/  /   \:\  /:/  /   \:\ \:\__\   \::/__/      \:\ \:\__\
  \/__/             \:\/:/  /     \:\/:/  /        /:/  /     \:\/:/  /     \:\/:/  /    \:\__\       \:\/:/  /
                     \::/  /       \::/  /        /:/  /       \::/  /       \::/  /      \/__/        \::/  /
                      \/__/         \/__/         \/__/         \/__/         \/__/                     \/__/
"""
ASCII_ART_MIN_WIDTH = 113

_AUTH_LOGIN_ERROR_CODES = {
    "AUTH_HEADER_MISSING",
    "TOKEN_MISSING",
    "TOKEN_EXPIRED",
    "TOKEN_INVALID",
    "TOKEN_REVOKED",
    "UNKNOWN_AUTH_ERROR",
}
_EXISTING_PROJECT_LINK_STEP = (
    "From an existing Osmosis project repo:\n  "
    "osmosis project link --workspace <workspace-id-or-name>"
)
_PLATFORM_WORKSPACE_SETUP_STEP = (
    "Set up a workspace in the Osmosis Platform and connect a project repository "
    "with Git Sync."
)
_PLATFORM_PROJECT_REPO_STEP = (
    "Clone the Platform/Git Sync managed project repo, then run Osmosis project "
    "commands from that checkout."
)


def _normalize_workspaces(raw_workspaces: Any) -> list[dict[str, str]]:
    """Return workspace entries that have both an ID and a name."""
    if not isinstance(raw_workspaces, list):
        return []

    workspaces: list[dict[str, str]] = []
    for workspace in raw_workspaces:
        if not isinstance(workspace, dict):
            continue
        ws_id = workspace.get("id")
        ws_name = workspace.get("name")
        if isinstance(ws_id, str) and ws_id and isinstance(ws_name, str) and ws_name:
            workspaces.append({"id": ws_id, "name": ws_name})
    return workspaces


def _load_login_workspaces(
    creds: Credentials,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Best-effort workspace lookup for post-login messaging."""
    from osmosis_ai.platform.auth import (
        AuthenticationExpiredError,
        PlatformAPIError,
        platform_request,
    )

    try:
        with console.spinner("Loading workspaces..."):
            data = platform_request(
                "/api/cli/workspaces",
                credentials=creds,
                require_workspace=False,
                cleanup_on_401=False,
            )
    except (AuthenticationExpiredError, PlatformAPIError) as exc:
        return None, str(exc)

    return _normalize_workspaces(data.get("workspaces")), None


def _login_operation_result(
    *,
    email: str,
    name: str | None,
    expires_at: Any,
    source: str,
    saved: bool,
    token_store: str | None = None,
    workspace_lookup_error: str | None = None,
    local_data_cleared: bool = False,
    workspace_count: int | None = None,
) -> Any:
    """Build the structured login result for JSON/plain output."""
    from osmosis_ai.cli.output import OperationResult

    resource: dict[str, Any] = {
        "email": email,
        "name": name,
        "expires_at": expires_at.isoformat(),
        "workspace": None,
        "source": source,
        "verified": True,
        "saved": saved,
    }
    if token_store is not None:
        resource["token_store"] = token_store
    if workspace_lookup_error is not None:
        resource["workspace_lookup_error"] = workspace_lookup_error
    if local_data_cleared:
        resource["local_data_cleared"] = True
    if workspace_count is not None:
        resource["workspace_count"] = workspace_count

    next_steps: list[str] = []
    next_steps_structured: list[dict[str, Any]] = []
    if saved:
        if workspace_count == 0:
            next_steps.extend(
                [_PLATFORM_WORKSPACE_SETUP_STEP, _PLATFORM_PROJECT_REPO_STEP]
            )
            next_steps_structured.extend(
                [
                    {
                        "action": "platform.setup_workspace",
                        "description": _PLATFORM_WORKSPACE_SETUP_STEP,
                    },
                    {
                        "action": "git_sync.clone_project_repo",
                        "description": _PLATFORM_PROJECT_REPO_STEP,
                    },
                ]
            )
        else:
            next_steps.append(_EXISTING_PROJECT_LINK_STEP)
            next_steps_structured.append(
                {
                    "action": "project.link",
                    "workspace": "<workspace-id-or-name>",
                }
            )

    return OperationResult(
        operation="auth.login",
        status="success",
        resource=resource,
        message=f"Logged in as {email}.",
        display_next_steps=next_steps,
        next_steps_structured=next_steps_structured,
    )


def _verify_env_token(env_token: str) -> Any:
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
        return verify_token(env_token)
    except LoginError as exc:
        code = exc.code
        message = env_messages.get(code) if code is not None else None
        if message is None and exc.status_code == 401:
            message = MSG_ENV_TOKEN_INVALID
        if message is not None:
            raise LoginError(message, code=code, status_code=exc.status_code) from exc
        raise


def _cli_error_from_login_error(exc: Any) -> CLIError:
    status_code = getattr(exc, "status_code", None)
    platform_code = getattr(exc, "code", None)
    if status_code == 401 or (
        isinstance(platform_code, str) and platform_code in _AUTH_LOGIN_ERROR_CODES
    ):
        return CLIError(str(exc), code="AUTH_REQUIRED")

    details: dict[str, Any] = {}
    if isinstance(status_code, int):
        details["status_code"] = status_code
    if isinstance(platform_code, str):
        details["platform_code"] = platform_code
    return CLIError(str(exc), code="PLATFORM_ERROR", details=details)


def _machine_login_with_token(*, token: str, force: bool) -> Any:
    """Verify and persist an explicit token, returning structured output."""
    from osmosis_ai.cli.output import get_output_context
    from osmosis_ai.platform.auth import load_credentials, verify_token
    from osmosis_ai.platform.auth.credentials import (
        Credentials,
        save_credentials,
    )
    from osmosis_ai.platform.auth.flow import LoginResult
    from osmosis_ai.platform.auth.local_config import clear_all_local_data
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    output = get_output_context()
    old_credentials = load_credentials(include_env=False)

    with output.status("Verifying token..."):
        verified = verify_token(token)
    creds = Credentials.from_verify_result(token, verified)
    result = LoginResult.from_verify_result(verified)

    token_store = save_credentials(creds)

    if (
        old_credentials
        and not old_credentials.is_expired()
        and old_credentials.token_id
        and old_credentials.token_id != creds.token_id
    ):
        with output.status("Revoking old session..."):
            revoke_cli_token(old_credentials)

    workspaces, workspace_lookup_error = _load_login_workspaces(creds)

    local_data_cleared = force or (
        old_credentials and old_credentials.user.id != creds.user.id
    )
    if local_data_cleared:
        clear_all_local_data()

    return _login_operation_result(
        email=result.user.email,
        name=result.user.name,
        expires_at=result.expires_at,
        source="token",
        saved=True,
        token_store=token_store,
        workspace_lookup_error=workspace_lookup_error,
        local_data_cleared=bool(local_data_cleared),
        workspace_count=len(workspaces) if workspaces is not None else None,
    )


def _machine_login_with_env_token(*, env_token: str) -> Any:
    """Verify OSMOSIS_TOKEN without mutating local credential/session state."""
    from osmosis_ai.cli.output import OperationResult, get_output_context

    output = get_output_context()
    with output.status("Verifying environment token..."):
        verified = _verify_env_token(env_token)

    return OperationResult(
        operation="auth.login",
        status="success",
        resource={
            "email": verified.user.email,
            "name": verified.user.name,
            "expires_at": verified.expires_at.isoformat(),
            "workspace": None,
            "source": "environment",
            "verified": True,
            "saved": False,
        },
        message=f"Verified OSMOSIS_TOKEN for {verified.user.email}.",
        display_next_steps=[
            "OSMOSIS_TOKEN was not saved to local credentials.",
            "Unset OSMOSIS_TOKEN to use interactive device login.",
        ],
        next_steps_structured=[
            {"action": "unset_env", "name": "OSMOSIS_TOKEN"},
        ],
    )


def _rich_login(force: bool, token: str | None) -> Any:
    """Run rich, interactive login behavior."""
    from osmosis_ai.platform.auth import (
        LoginError,
        device_login,
        load_credentials,
        verify_token,
    )
    from osmosis_ai.platform.auth.credentials import (
        Credentials,
        save_credentials,
    )
    from osmosis_ai.platform.auth.flow import LoginResult
    from osmosis_ai.platform.auth.local_config import clear_all_local_data
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    if console.rich.width >= ASCII_ART_MIN_WIDTH:
        console.print(ASCII_ART, markup=False, highlight=False)
    else:
        console.print()
        console.print("  Osmosis AI", style="bold magenta")
        console.print()

    try:
        env_token = os.environ.get("OSMOSIS_TOKEN")
        if env_token and token is None:
            return _machine_login_with_env_token(env_token=env_token)

        old_credentials = (
            load_credentials(include_env=False) if token else load_credentials()
        )

        # Two login paths: token or device flow
        if token:
            with console.spinner("Verifying token..."):
                verified = verify_token(token)
            creds = Credentials.from_verify_result(token, verified)
            result = LoginResult.from_verify_result(verified)
        else:
            result, creds = device_login()

        save_credentials(creds)

        # Revoke old token server-side after the new credentials are saved,
        # so a failed login attempt does not destroy the current session.
        # Skip when re-logging with the same PAT to avoid revoking the
        # token we just verified.
        if (
            old_credentials
            and not old_credentials.is_expired()
            and old_credentials.token_id
            and old_credentials.token_id != creds.token_id
        ):
            with console.spinner("Revoking old session..."):
                revoke_cli_token(old_credentials)

        workspaces, workspace_lookup_error = _load_login_workspaces(creds)

        # Clear legacy auth-local state when user identity changes or when
        # explicitly forcing a fresh start.
        local_data_cleared = force or (
            old_credentials and old_credentials.user.id != creds.user.id
        )
        if local_data_cleared:
            clear_all_local_data()

        # Display login success
        esc = console.escape

        info_lines = [f"Email: {esc(result.user.email)}"]
        if result.user.name:
            info_lines.append(f"Name: {esc(result.user.name)}")
        info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")

        console.panel("Login Successful", "\n".join(info_lines), style="green")

        if workspace_lookup_error is not None:
            console.print(
                "\nAuthenticated, but could not load your workspaces yet.",
                style="yellow",
            )
        elif workspaces == []:
            console.print(
                "\nNo workspaces were found for this account. "
                f"{_PLATFORM_WORKSPACE_SETUP_STEP} {_PLATFORM_PROJECT_REPO_STEP}",
                style="dim",
            )

        if workspaces != []:
            console.print(f"\n{_EXISTING_PROJECT_LINK_STEP}", style="dim")

    except LoginError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n\nLogin cancelled.")
        raise typer.Exit(1) from None


@app.command("login")
def login(
    force: bool = typer.Option(
        False, "-f", "--force", help="Force re-login, clearing existing credentials."
    ),
    token: str | None = typer.Option(
        None, "--token", help="Authenticate with a personal access token (for CI/CD)."
    ),
) -> Any:
    """Authenticate with Osmosis AI."""
    from osmosis_ai.cli.output import OutputFormat, get_output_context
    from osmosis_ai.platform.auth import LoginError

    output = get_output_context()
    if output.format is OutputFormat.rich:
        return _rich_login(force=force, token=token)

    try:
        if token is not None:
            return _machine_login_with_token(token=token, force=force)

        env_token = os.environ.get("OSMOSIS_TOKEN")
        if env_token:
            return _machine_login_with_env_token(env_token=env_token)

        raise CLIError(
            "Login is interactive in this mode. Pass --token or set OSMOSIS_TOKEN.",
            code="INTERACTIVE_REQUIRED",
        )
    except LoginError as exc:
        raise _cli_error_from_login_error(exc) from exc


@app.command("logout")
def logout(
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> Any:
    """Logout from Osmosis AI CLI."""
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
    from osmosis_ai.cli.prompts import confirm
    from osmosis_ai.platform.auth import load_credentials
    from osmosis_ai.platform.auth.local_config import reset_session
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    output = get_output_context()
    credentials = load_credentials(include_env=False)
    env_token_set = bool(os.environ.get("OSMOSIS_TOKEN"))

    if credentials is None:
        if env_token_set:
            message = "OSMOSIS_TOKEN is set. Run 'unset OSMOSIS_TOKEN' to logout."
            if output.format is OutputFormat.rich:
                console.print(message, style="yellow")
                return None
            return OperationResult(
                operation="auth.logout",
                status="noop",
                resource={"logged_in": True, "env_token_set": True},
                message=message,
                display_next_steps=["Run 'unset OSMOSIS_TOKEN' to logout."],
                next_steps_structured=[
                    {"action": "unset_env", "name": "OSMOSIS_TOKEN"}
                ],
            )
        if output.format is OutputFormat.rich:
            console.print("Not logged in.")
            return None
        return OperationResult(
            operation="auth.logout",
            status="noop",
            resource={"logged_in": False},
            message="Not logged in.",
        )

    if not skip_confirm:
        if output.format is not OutputFormat.rich or not output.interactive:
            raise CLIError(
                "Use --yes to confirm in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
            )
        result = confirm("Logout from Osmosis AI?", default=False)
        if result is None:  # User cancelled with Ctrl+C
            return
        if not result:
            console.print("Cancelled.")
            return

    # Best-effort server-side revocation
    revoked = False
    if not credentials.is_expired():
        with output.status("Revoking session..."):
            revoked = revoke_cli_token(credentials)

    # Delete local credentials and workspace/local state
    reset_session()

    if output.format is not OutputFormat.rich:
        return OperationResult(
            operation="auth.logout",
            status="success",
            resource={
                "logged_in": False,
                "revoked": revoked,
                "env_token_set": env_token_set,
            },
            message="Logged out successfully.",
            display_next_steps=(
                ["Unset OSMOSIS_TOKEN to fully logout."] if env_token_set else []
            ),
            next_steps_structured=(
                [{"action": "unset_env", "name": "OSMOSIS_TOKEN"}]
                if env_token_set
                else []
            ),
        )

    console.print("Logged out successfully.", style="green")

    if env_token_set:
        console.print(
            "Warning: OSMOSIS_TOKEN environment variable is still set. "
            "Run 'unset OSMOSIS_TOKEN' to fully logout.",
            style="yellow",
        )
    return None


@app.command("whoami")
def whoami() -> Any:
    """Show current authenticated user."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.platform.auth import (
        AuthenticationExpiredError,
        Credentials,
        LoginError,
        load_credentials,
        verify_token,
    )
    from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

    output = get_output_context()
    env_token = os.environ.get("OSMOSIS_TOKEN")
    source = "environment" if env_token else "credentials"

    if env_token:
        try:
            with output.status("Verifying token..."):
                verified = _verify_env_token(env_token)
        except LoginError as exc:
            raise _cli_error_from_login_error(exc) from exc
        credentials = Credentials.from_verify_result(env_token, verified)
    else:
        credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN, code="AUTH_REQUIRED")
    if not env_token and credentials.is_expired():
        raise AuthenticationExpiredError()
    if not env_token:
        try:
            with output.status("Verifying session..."):
                verified = verify_token(credentials.access_token)
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

    account = {
        "email": credentials.user.email,
        "name": credentials.user.name,
        "expires_at": credentials.expires_at.isoformat(),
        "source": source,
    }
    if credentials.token_id:
        account["token_id"] = credentials.token_id
    data = {
        "account": account,
        "local_linked_project": None,
        "email": credentials.user.email,
        "name": credentials.user.name,
        "workspace": None,
        "linked_project": None,
        "expires_at": credentials.expires_at.isoformat(),
        "source": source,
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
    fields.append(DetailField(label="Auth source", value=source))
    fields.append(
        DetailField(label="Expires", value=credentials.expires_at.strftime("%Y-%m-%d"))
    )

    result = DetailResult(title="Account", data=data, fields=fields)

    # Some existing unit tests call this handler directly, outside Typer's
    # result callback. Preserve that rich rendering path without duplicating
    # output during normal CLI invocations.
    if output.format is OutputFormat.rich:
        import click

        if click.get_current_context(silent=True) is None:
            console.table([(field.label, field.value) for field in fields])

    return result

"""Authentication commands: login, logout, whoami."""

from __future__ import annotations

import contextlib
import os
import shlex
from typing import TYPE_CHECKING, Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.errors import CLIError

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
        data = platform_request(
            "/api/cli/workspaces",
            credentials=creds,
            require_workspace=False,
            cleanup_on_401=False,
        )
    except (AuthenticationExpiredError, PlatformAPIError) as exc:
        return None, str(exc)

    return _normalize_workspaces(data.get("workspaces")), None


def _validate_workspace_context(
    creds: Credentials,
    *,
    workspaces: list[dict[str, str]] | None = None,
) -> None:
    """Validate the stored workspace is still accessible with new credentials.

    After login, the workspace ID in config.json may be stale (e.g. the user
    switched between local dev and production, or the local DB was recreated).
    This check prevents confusing 403 errors on subsequent commands.
    """
    from osmosis_ai.platform.auth.local_config import (
        clear_all_local_data,
        get_active_workspace,
        set_active_workspace,
    )

    ws = get_active_workspace()
    if not ws:
        return

    try:
        if workspaces is None:
            workspaces, _ = _load_login_workspaces(creds)
        if workspaces is None:
            return

        ws_by_id = {w["id"]: w for w in workspaces if "id" in w}
        ws_by_name = {w["name"]: w for w in workspaces if "name" in w}

        if ws["id"] in ws_by_id:
            return  # Still valid

        # ID is stale -- try to fix by matching workspace name
        if ws["name"] in ws_by_name:
            correct = ws_by_name[ws["name"]]
            set_active_workspace(correct["id"], correct["name"])
            return

        # Workspace no longer accessible at all
        clear_all_local_data()
        console.print(
            "\nPrevious workspace is no longer accessible. "
            "Resetting local workspace selection.",
            style="yellow",
        )
    except Exception:
        pass  # Don't block login for validation errors


def _ensure_login_workspace_selection(
    workspaces: list[dict[str, str]] | None,
) -> tuple[dict[str, str] | None, bool]:
    """Return the active workspace, auto-selecting the only available one."""
    from osmosis_ai.platform.auth.local_config import (
        get_active_workspace,
        set_active_workspace,
    )

    active_workspace = get_active_workspace()
    if active_workspace is not None:
        return active_workspace, False
    if workspaces is None or len(workspaces) != 1:
        return None, False

    workspace = workspaces[0]
    set_active_workspace(workspace["id"], workspace["name"])
    return workspace, True


@app.command("login")
def login(
    force: bool = typer.Option(
        False, "-f", "--force", help="Force re-login, clearing existing credentials."
    ),
    token: str | None = typer.Option(
        None, "--token", help="Authenticate with a personal access token (for CI/CD)."
    ),
) -> None:
    """Authenticate with Osmosis AI."""
    from osmosis_ai.platform.auth import (
        LoginError,
        device_login,
        load_credentials,
        verify_token,
    )
    from osmosis_ai.platform.auth.credentials import (
        Credentials,
        delete_credentials,
        save_credentials,
    )
    from osmosis_ai.platform.auth.flow import LoginResult
    from osmosis_ai.platform.auth.local_config import clear_all_local_data
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    if console.rich.width >= ASCII_ART_MIN_WIDTH:
        print(ASCII_ART)
    else:
        console.print()
        console.print("  Osmosis AI", style="bold magenta")
        console.print()

    # Refuse to login if OSMOSIS_TOKEN env var is set
    if os.environ.get("OSMOSIS_TOKEN"):
        console.print_error(
            "The OSMOSIS_TOKEN environment variable is set. "
            "Unset it to use 'osmosis auth login'."
        )
        raise typer.Exit(1)

    try:
        old_credentials = load_credentials()

        # Clear existing credentials if forcing re-login
        if force and old_credentials:
            delete_credentials()

        # Two login paths: token or device flow
        if token:
            verified = verify_token(token)
            creds = Credentials.from_verify_result(token, verified)
            result = LoginResult.from_verify_result(verified)
        else:
            result, creds = device_login()

        # Revoke old token server-side before overwriting local credentials,
        # so it doesn't become an unmanageable orphan on the platform.
        # Skip when re-logging with the same PAT to avoid revoking the
        # token we just verified.
        if (
            old_credentials
            and not old_credentials.is_expired()
            and old_credentials.token_id
            and old_credentials.token_id != creds.token_id
        ):
            revoke_cli_token(old_credentials)

        save_credentials(creds)

        workspaces, workspace_lookup_error = _load_login_workspaces(creds)

        # Clear stale workspace and local state when user identity changes
        # or when explicitly forcing a fresh start, to prevent subsequent
        # commands from sending the old workspace ID in X-Osmosis-Org.
        local_data_cleared = force or (
            old_credentials and old_credentials.user.id != creds.user.id
        )
        if local_data_cleared:
            clear_all_local_data()
        elif workspaces is not None:
            _validate_workspace_context(creds, workspaces=workspaces)

        active_workspace, auto_selected = _ensure_login_workspace_selection(workspaces)

        # Display login success
        esc = console.escape

        info_lines = [f"Email: {esc(result.user.email)}"]
        if result.user.name:
            info_lines.append(f"Name: {esc(result.user.name)}")
        if active_workspace:
            info_lines.append(f"Workspace: {esc(active_workspace['name'])}")
        info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")

        console.panel("Login Successful", "\n".join(info_lines), style="green")

        if workspace_lookup_error is not None:
            if active_workspace is not None:
                console.print(
                    "\nAuthenticated, but could not refresh your workspace list. "
                    "Current workspace selection was kept.",
                    style="yellow",
                )
            else:
                console.print(
                    "\nAuthenticated, but could not load your workspaces yet. "
                    "Run 'osmosis workspace' later to choose one.",
                    style="yellow",
                )
        elif auto_selected and active_workspace is not None:
            console.print(
                f"\nAutomatically selected your only workspace: "
                f"{esc(active_workspace['name'])}",
                style="green",
                highlight=False,
            )
        elif workspaces and len(workspaces) > 1:
            console.print(
                "\nMultiple workspaces are available. Switch with:",
                style="dim",
            )
            for workspace in workspaces:
                command = f"osmosis workspace switch {shlex.quote(workspace['name'])}"
                console.print(f"  {esc(command)}")
            console.print(
                "Or run 'osmosis workspace' for interactive selection.",
                style="dim",
            )
        elif active_workspace is None and workspaces == []:
            console.print(
                "\nNo workspaces were found for this account. "
                "Run 'osmosis workspace' to manage workspaces.",
                style="dim",
            )
        elif active_workspace is None:
            console.print(
                "\nRun 'osmosis workspace' to select a workspace.",
                style="dim",
            )

    except LoginError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n\nLogin cancelled.")
        raise typer.Exit(1) from None


@app.command("logout")
def logout(
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> None:
    """Logout from Osmosis AI CLI."""
    from osmosis_ai.cli.prompts import confirm
    from osmosis_ai.platform.auth import load_credentials
    from osmosis_ai.platform.auth.local_config import reset_session
    from osmosis_ai.platform.auth.platform_client import revoke_cli_token

    credentials = load_credentials()

    if credentials is None:
        console.print("Not logged in.")
        return

    if not skip_confirm:
        result = confirm("Logout from Osmosis AI?", default=False)
        if result is None:  # User cancelled with Ctrl+C
            return
        if not result:
            console.print("Cancelled.")
            return

    # Best-effort server-side revocation
    if not credentials.is_expired():
        revoke_cli_token(credentials)

    # Delete local credentials and workspace/local state
    reset_session()

    console.print("Logged out successfully.", style="green")

    if os.environ.get("OSMOSIS_TOKEN"):
        console.print(
            "Warning: OSMOSIS_TOKEN environment variable is still set. "
            "Run 'unset OSMOSIS_TOKEN' to fully logout.",
            style="yellow",
        )


@app.command("whoami")
def whoami() -> None:
    """Show current authenticated user and workspace."""
    from osmosis_ai.platform.auth import (
        AuthenticationExpiredError,
        PlatformAPIError,
        ensure_active_workspace,
        load_credentials,
    )
    from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

    credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)

    active_workspace = None
    with contextlib.suppress(AuthenticationExpiredError, PlatformAPIError):
        active_workspace = ensure_active_workspace(
            credentials=credentials,
            cleanup_on_401=False,
        )
    esc = console.escape

    rows = [("Email", esc(credentials.user.email))]
    if credentials.user.name:
        rows.append(("Name", esc(credentials.user.name)))
    if active_workspace:
        rows.append(("Workspace", esc(active_workspace["name"])))
    rows.append(("Expires", credentials.expires_at.strftime("%Y-%m-%d")))

    console.table(rows)

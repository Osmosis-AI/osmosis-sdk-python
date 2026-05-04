"""Tests for login command workspace cleanup behavior."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest

import osmosis_ai.cli.commands.auth as auth_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError, PlatformAPIError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import LoginResult


def _make_credentials(
    user_id: str = "user_1", email: str = "a@example.com"
) -> Credentials:
    now = datetime.now(UTC)
    return Credentials(
        access_token="tok",
        token_type="Bearer",
        expires_at=now + timedelta(days=30),
        created_at=now,
        user=UserInfo(id=user_id, email=email, name="User"),
    )


def _make_login_result(email: str = "a@example.com") -> LoginResult:
    return LoginResult(
        user=UserInfo(id="user_1", email=email, name="User"),
        expires_at=datetime.now(UTC) + timedelta(days=30),
    )


@pytest.fixture(autouse=True)
def _stub_workspace_resolution(monkeypatch) -> None:
    """Keep login tests offline unless a test overrides workspace resolution."""
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {"workspaces": []},
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: None,
    )


# ---------------------------------------------------------------------------
# --force always clears workspace context
# ---------------------------------------------------------------------------


def test_force_login_clears_workspace_data(monkeypatch) -> None:
    """--force must clear workspace and local state after successful login."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")  # same user
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: old_creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials", lambda: True
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: clear_calls.append(True),
    )

    auth_module.login(force=True, token=None)

    assert clear_calls, "clear_all_local_data must be called when --force is used"


# ---------------------------------------------------------------------------
# User identity change triggers cleanup (non-force)
# ---------------------------------------------------------------------------


def test_login_clears_workspace_when_user_changes(monkeypatch) -> None:
    """Logging in as a different user must clear stale workspace/local state."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_2", email="b@example.com")
    result = _make_login_result(email="b@example.com")

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: old_creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: clear_calls.append(True),
    )

    auth_module.login(force=False, token=None)

    assert clear_calls, "clear_all_local_data must be called when user identity changes"


# ---------------------------------------------------------------------------
# Same user re-login preserves workspace context
# ---------------------------------------------------------------------------


def test_login_preserves_workspace_when_same_user(monkeypatch) -> None:
    """Re-login as the same user (no --force) should keep workspace/local state."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: old_creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: clear_calls.append(True),
    )

    auth_module.login(force=False, token=None)

    assert not clear_calls, "workspace/local state should be preserved for same user"


# ---------------------------------------------------------------------------
# First-time login (no previous credentials) skips cleanup
# ---------------------------------------------------------------------------


def test_first_login_does_not_clear_workspace(monkeypatch) -> None:
    """First login (no previous credentials) should not attempt cleanup."""
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: clear_calls.append(True),
    )

    auth_module.login(force=False, token=None)

    assert not clear_calls, "no cleanup needed for first-time login"


def test_login_success_prompts_project_link_not_workspace_switch(
    monkeypatch, capsys
) -> None:
    """Login should point users at project linking, not global workspace selection."""
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {
            "workspaces": [{"id": "ws_only", "name": "solo-workspace"}]
        },
    )
    monkeypatch.setattr(
        auth_module,
        "console",
        Console(force_terminal=False, no_color=True, width=100),
    )

    auth_module.login(force=False, token=None)

    rendered = capsys.readouterr().out
    assert "Login Successful" in rendered
    assert "osmosis project link --workspace <workspace-id-or-name>" in rendered
    assert "workspace switch" not in rendered
    assert "Workspace: solo-workspace" not in rendered


def test_login_omits_switch_commands_for_multiple_workspaces(
    monkeypatch, capsys
) -> None:
    """Login should not print removed workspace switch commands."""
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {
            "workspaces": [
                {"id": "ws_alpha", "name": "team-alpha"},
                {"id": "ws_beta", "name": "ML Team"},
            ]
        },
    )
    monkeypatch.setattr(
        auth_module,
        "console",
        Console(force_terminal=False, no_color=True, width=100),
    )

    auth_module.login(force=False, token=None)

    rendered = capsys.readouterr().out
    assert "osmosis project link --workspace <workspace-id-or-name>" in rendered
    assert "workspace switch" not in rendered


@pytest.mark.parametrize(
    "error",
    [
        PlatformAPIError("workspace endpoint unavailable"),
        AuthenticationExpiredError("session expired"),
    ],
)
def test_login_does_not_fail_when_workspace_lookup_fails(monkeypatch, error) -> None:
    """Workspace lookup errors should not turn a successful login into a failure."""
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()
    output = io.StringIO()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda c: "keyring"
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )

    def raise_workspace_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        raise_workspace_error,
    )
    monkeypatch.setattr(
        auth_module,
        "console",
        Console(file=output, force_terminal=False, no_color=True, width=80),
    )

    auth_module.login(force=False, token=None)

    rendered = output.getvalue()
    assert "Login Successful" in rendered
    assert "Authenticated, but could not load your workspaces yet." in rendered


def test_whoami_prints_local_identity_outside_project(monkeypatch) -> None:
    """whoami should not require a workspace or project link outside projects."""
    creds = _make_credentials(user_id="user_1")
    output = io.StringIO()

    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.resolve_project_root_from_cwd",
        lambda: (_ for _ in ()).throw(CLIError("not in project")),
    )
    monkeypatch.setattr(
        auth_module,
        "console",
        Console(file=output, force_terminal=False, no_color=True, width=80),
    )

    auth_module.whoami()

    rendered = output.getvalue()
    assert "a@example.com" in rendered
    assert "User" in rendered
    assert creds.expires_at.strftime("%Y-%m-%d") in rendered
    assert "Workspace" not in rendered

"""Tests for login command workspace cleanup behavior."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest

import osmosis_ai.cli.commands.auth as auth_module
from osmosis_ai.cli.console import Console
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
        "osmosis_ai.platform.auth.ensure_active_workspace",
        lambda credentials=None, **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.get_active_workspace",
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


def test_login_auto_selects_only_workspace(monkeypatch) -> None:
    """Login should auto-select the only available workspace and skip the prompt."""
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
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.ensure_active_workspace",
        lambda credentials=None, **kwargs: {
            "id": "ws_only",
            "name": "solo-workspace",
        },
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.get_active_workspace",
        lambda: {"id": "ws_only", "name": "solo-workspace"},
    )
    monkeypatch.setattr(
        auth_module,
        "console",
        Console(file=output, force_terminal=False, no_color=True, width=80),
    )

    auth_module.login(force=False, token=None)

    rendered = output.getvalue()
    assert "Workspace: solo-workspace" in rendered
    assert "Run 'osmosis workspace' to select a workspace." not in rendered

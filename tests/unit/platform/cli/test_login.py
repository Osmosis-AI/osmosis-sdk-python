"""Tests for login command workspace cleanup behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Module under test
import osmosis_ai.platform.cli.login as login_module
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import LoginResult


def _make_credentials(
    user_id: str = "user_1", email: str = "a@example.com"
) -> Credentials:
    now = datetime.now(timezone.utc)
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
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )


# ---------------------------------------------------------------------------
# --force always clears workspace context
# ---------------------------------------------------------------------------


def test_force_login_clears_workspace_data(monkeypatch) -> None:
    """--force must clear workspace/project context after successful login."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")  # same user
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "load_credentials", lambda: old_creds)
    monkeypatch.setattr(login_module, "delete_credentials", lambda: True)
    monkeypatch.setattr(login_module, "save_credentials", lambda c: "keyring")

    monkeypatch.setattr(
        login_module,
        "device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        login_module, "clear_all_local_data", lambda: clear_calls.append(True)
    )

    login_module.login_cmd(force=True, token=None)

    assert clear_calls, "clear_all_local_data must be called when --force is used"


# ---------------------------------------------------------------------------
# User identity change triggers cleanup (non-force)
# ---------------------------------------------------------------------------


def test_login_clears_workspace_when_user_changes(monkeypatch) -> None:
    """Logging in as a different user must clear stale workspace context."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_2", email="b@example.com")
    result = _make_login_result(email="b@example.com")

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "load_credentials", lambda: old_creds)
    monkeypatch.setattr(login_module, "save_credentials", lambda c: "keyring")
    monkeypatch.setattr(
        login_module,
        "device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        login_module, "clear_all_local_data", lambda: clear_calls.append(True)
    )

    login_module.login_cmd(force=False, token=None)

    assert clear_calls, "clear_all_local_data must be called when user identity changes"


# ---------------------------------------------------------------------------
# Same user re-login preserves workspace context
# ---------------------------------------------------------------------------


def test_login_preserves_workspace_when_same_user(monkeypatch) -> None:
    """Re-login as the same user (no --force) should keep workspace context."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "load_credentials", lambda: old_creds)
    monkeypatch.setattr(login_module, "save_credentials", lambda c: "keyring")
    monkeypatch.setattr(
        login_module,
        "device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        login_module, "clear_all_local_data", lambda: clear_calls.append(True)
    )

    login_module.login_cmd(force=False, token=None)

    assert not clear_calls, "workspace context should be preserved for same user"


# ---------------------------------------------------------------------------
# First-time login (no previous credentials) skips cleanup
# ---------------------------------------------------------------------------


def test_first_login_does_not_clear_workspace(monkeypatch) -> None:
    """First login (no previous credentials) should not attempt cleanup."""
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "load_credentials", lambda: None)
    monkeypatch.setattr(login_module, "save_credentials", lambda c: "keyring")
    monkeypatch.setattr(
        login_module,
        "device_login",
        lambda **kw: (result, new_creds),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        login_module, "clear_all_local_data", lambda: clear_calls.append(True)
    )

    login_module.login_cmd(force=False, token=None)

    assert not clear_calls, "no cleanup needed for first-time login"

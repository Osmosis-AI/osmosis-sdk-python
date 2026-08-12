"""Tests for login command workspace cleanup behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import osmosis_ai.platform.cli.auth as auth_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat
from osmosis_ai.cli.output.context import override_output_context
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import LoginResult, VerifyResult


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
def _interactive_rich() -> None:
    """Device-login unit tests call the handler outside Typer."""
    with override_output_context(format=OutputFormat.rich, interactive=True):
        yield


@pytest.fixture(autouse=True)
def _stub_workspace_resolution(monkeypatch) -> None:
    """Login is account-only and must not query workspace APIs."""
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "auth login must not look up platform workspaces"
        ),
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


def test_force_login_leaves_new_credentials_saved(monkeypatch) -> None:
    """Destructive local cleanup must not delete newly saved credentials."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()
    saved_tokens: list[str] = ["old-token"]
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: old_creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda c: saved_tokens.append(c.access_token) or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: (calls.append("clear_all_local_data"), saved_tokens.clear()),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: calls.append("revoke_cli_token") or True,
    )

    auth_module.login(force=True, token=None)

    assert saved_tokens == [new_creds.access_token]
    assert calls == ["clear_all_local_data"]


def test_force_login_restores_old_credentials_when_new_save_fails(monkeypatch) -> None:
    """A failed replacement save should not leave the user logged out locally."""
    old_creds = _make_credentials(user_id="user_1")
    new_creds = _make_credentials(user_id="user_1")
    result = _make_login_result()
    calls: list[str] = []

    def save_credentials(creds: Credentials) -> str:
        if creds is new_creds:
            calls.append("save_new")
            raise OSError("disk full")
        if creds is old_creds:
            calls.append("restore_old")
            return "keyring"
        raise AssertionError("unexpected credentials")

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: old_creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", save_credentials
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (result, new_creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: calls.append("clear_all_local_data"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: calls.append("revoke_cli_token") or True,
    )

    with pytest.raises(OSError, match="disk full"):
        auth_module.login(force=True, token=None)

    assert calls == ["clear_all_local_data", "save_new", "restore_old"]


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


def test_login_success_prompts_clone_and_doctor_not_workspace_link(
    monkeypatch,
) -> None:
    """Login should point users at Platform clones and the doctor command."""
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

    login_result = auth_module.login(force=False, token=None)

    assert login_result.message == "Logged in as a@example.com."
    assert (
        "Create or open a workspace in the Osmosis Platform"
        in (login_result.display_next_steps[0])
    )
    assert any("osmosis doctor" in step for step in login_result.display_next_steps)
    serialized = " ".join(login_result.display_next_steps)
    assert "workspace link" not in serialized
    assert "workspace.validate" not in serialized
    assert "workspace.link" not in serialized
    assert "workspace switch" not in serialized


def test_login_next_steps_omit_workspace_specific_guidance(monkeypatch) -> None:
    """Login guidance should be generic account bootstrap guidance."""
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

    login_result = auth_module.login(force=False, token=None)

    assert login_result.message == "Logged in as a@example.com."
    serialized = " ".join(login_result.display_next_steps)
    assert "Create or open a workspace in the Osmosis Platform" in serialized
    assert "clone the repository created there" in serialized
    assert "osmosis doctor" in serialized
    assert "Git Sync" not in serialized
    assert "workspace create" not in serialized
    assert "workspace list" not in serialized
    assert "workspace link" not in serialized


def test_login_keyboardinterrupt_propagates(monkeypatch) -> None:
    """Ctrl+C during device login must not be swallowed as a generic failure."""
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        auth_module.login(force=False, token=None)


def test_device_login_allows_rich_mode_with_redirected_stdin(monkeypatch) -> None:
    """Rich device login does not read stdin and must work without a TTY."""
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    expected = auth_module._login_operation_result(
        email="a@example.com",
        name="User",
        expires_at=datetime.now(UTC) + timedelta(days=30),
        source="device",
        saved=True,
    )
    monkeypatch.setattr(
        auth_module,
        "_login_with_device_flow",
        lambda *, force: expected,
    )

    with override_output_context(format=OutputFormat.rich, interactive=False):
        result = auth_module.login(force=False, token=None)

    assert result is expected


def test_whoami_prints_local_identity_outside_workspace_directory(monkeypatch) -> None:
    """whoami should not require workspace directory setup outside repositories."""
    creds = _make_credentials(user_id="user_1")

    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: VerifyResult(
            user=creds.user,
            expires_at=creds.expires_at,
            token_id=creds.token_id,
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_directory_contract.resolve_workspace_directory_from_cwd",
        lambda: (_ for _ in ()).throw(CLIError("not in workspace directory")),
    )

    result = auth_module.whoami()

    assert result.data["email"] == "a@example.com"
    assert result.data["name"] == "User"
    assert creds.expires_at.strftime("%Y-%m-%d") in str(
        [field.value for field in result.fields]
    )
    assert result.data["workspace"] is None
    assert all(field.label != "Workspace" for field in result.fields)

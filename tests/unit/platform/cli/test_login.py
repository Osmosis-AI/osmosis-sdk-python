"""Tests for authentication command business logic."""

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
    *,
    access_token: str = "tok",
    token_id: str | None = None,
    user_id: str = "user_1",
    email: str = "a@example.com",
) -> Credentials:
    now = datetime.now(UTC)
    return Credentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=now + timedelta(days=30),
        created_at=now,
        user=UserInfo(id=user_id, email=email, name="User"),
        token_id=token_id,
    )


def _make_login_result(*, email: str = "a@example.com") -> LoginResult:
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


def test_login_revokes_before_cleaning_up_old_keyring_token(monkeypatch) -> None:
    old_creds = _make_credentials(access_token="old", token_id="tok_old")
    new_creds = _make_credentials(access_token="new", token_id="tok_new")
    result = _make_login_result()
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: calls.append("save") or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    auth_module.login()

    assert calls == ["save", "revoke", "cleanup"]


def test_login_keeps_old_keyring_token_when_server_revoke_fails(monkeypatch) -> None:
    old_creds = _make_credentials(access_token="old", token_id="tok_old")
    new_creds = _make_credentials(access_token="new", token_id="tok_new")
    result = _make_login_result()
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: calls.append("save") or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or False,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    auth_module.login()

    assert calls == ["save", "revoke"]


def test_login_resolves_missing_old_token_id_before_revoke(monkeypatch) -> None:
    old_creds = _make_credentials(access_token="old", token_id=None)
    new_creds = _make_credentials(access_token="new", token_id="tok_new")
    result = _make_login_result()
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: calls.append("save") or "keyring",
    )

    def verify_old(token: str) -> VerifyResult:
        assert token == "old"
        calls.append("verify")
        return VerifyResult(
            user=old_creds.user,
            expires_at=old_creds.expires_at,
            token_id="tok_old",
        )

    def fail_revoke(credentials: Credentials) -> bool:
        assert credentials.token_id == "tok_old"
        calls.append("revoke")
        return False

    monkeypatch.setattr("osmosis_ai.platform.auth.flow.verify_token", verify_old)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token", fail_revoke
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    auth_module.login()

    assert calls == ["save", "verify", "revoke"]


def test_login_cleans_original_keyring_account_after_resolving_token_id(
    monkeypatch,
) -> None:
    old_creds = _make_credentials(access_token="old", token_id=None)
    new_creds = _make_credentials(access_token="new", token_id="tok_new")
    result = _make_login_result()
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: calls.append("save") or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.flow.verify_token",
        lambda token: (
            calls.append("verify")
            or VerifyResult(
                user=old_creds.user,
                expires_at=old_creds.expires_at,
                token_id="tok_old",
            )
        ),
    )

    def revoke(resolved: Credentials) -> bool:
        assert resolved.token_id == "tok_old"
        calls.append("revoke")
        return True

    def cleanup(original: Credentials, current: Credentials) -> bool:
        assert original is old_creds
        assert original.token_id is None
        assert current is new_creds
        calls.append("cleanup")
        return True

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token", revoke
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        cleanup,
    )

    auth_module.login()

    assert calls == ["save", "verify", "revoke", "cleanup"]


def test_login_keeps_old_keyring_when_both_token_ids_are_missing(
    monkeypatch,
) -> None:
    old_creds = _make_credentials(access_token="old", token_id=None)
    new_creds = _make_credentials(access_token="new", token_id=None)
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.flow.verify_token",
        lambda token: VerifyResult(
            user=old_creds.user,
            expires_at=old_creds.expires_at,
            token_id=None,
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: pytest.fail("a token without an ID cannot be revoked"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: pytest.fail(
            "the old keyring entry must remain when revocation is impossible"
        ),
    )
    warnings: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "osmosis_ai.cli.console.console.print_warning",
        lambda message, **kwargs: warnings.append((message, kwargs.get("code"))),
    )

    auth_module.login()

    assert warnings == [
        (
            "The new login is active, but the previous session did not expose "
            "a token ID and could not be revoked. Its local keyring entry was kept.",
            "TOKEN_REVOKE_FAILED",
        )
    ]


def test_login_save_failure_does_not_revoke_old_token(monkeypatch) -> None:
    old_creds = _make_credentials(access_token="old", token_id="tok_old")
    new_creds = _make_credentials(access_token="new", token_id="tok_new")
    result = _make_login_result()
    calls: list[str] = []

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_creds,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )

    def fail_save(credentials: Credentials, **kwargs) -> str:
        calls.append("save")
        raise OSError("disk full")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", fail_save
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    with pytest.raises(OSError, match="disk full"):
        auth_module.login()

    assert calls == ["save"]


def test_device_login_loads_persistent_credentials_without_env(monkeypatch) -> None:
    calls: list[bool] = []
    new_creds = _make_credentials(token_id="tok_new")
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    def load_credentials(*, include_env: bool = True):
        calls.append(include_env)
        return None

    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", load_credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )

    auth_module.login()

    assert calls == [False]


def test_login_success_prompts_clone_and_doctor(monkeypatch) -> None:
    new_creds = _make_credentials()
    result = _make_login_result()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login", lambda: (result, new_creds)
    )

    login_result = auth_module.login()

    assert login_result.message == "Logged in as a@example.com."
    serialized = " ".join(login_result.display_next_steps)
    assert "Create or open a workspace in the Osmosis Platform" in serialized
    assert "clone the repository created there" in serialized
    assert "osmosis doctor" in serialized
    assert "workspace link" not in serialized


def test_login_keyboardinterrupt_propagates(monkeypatch) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        auth_module.login()


def test_device_login_allows_rich_mode_with_redirected_stdin(monkeypatch) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    expected = auth_module._login_operation_result(
        email="a@example.com",
        name="User",
        expires_at=datetime.now(UTC) + timedelta(days=30),
        source="device",
        saved=True,
    )
    monkeypatch.setattr(auth_module, "_login_with_device_flow", lambda: expected)

    with override_output_context(format=OutputFormat.rich, interactive=False):
        result = auth_module.login()

    assert result is expected


def test_whoami_prints_local_identity_outside_workspace_directory(monkeypatch) -> None:
    creds = _make_credentials()

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_credential_store",
        lambda *, include_env=False: "keyring",
    )
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
    assert result.data["effective_source"] == "keyring"
    assert result.data["persistent_login"] is True
    assert result.data["workspace"] is None
    assert all(field.label != "Workspace" for field in result.fields)

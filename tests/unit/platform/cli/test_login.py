"""Tests for authentication command business logic."""

from __future__ import annotations

import json
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


@pytest.fixture(autouse=True)
def _stub_keyring_available(monkeypatch) -> None:
    """Exercise login behavior independently of the host keyring backend."""
    monkeypatch.setattr("keyring.get_keyring", lambda: object())


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
        lambda credentials, **kwargs: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    auth_module.login()

    assert calls == ["save", "revoke", "cleanup"]


def test_login_cleans_old_keyring_token_when_server_revoke_fails(monkeypatch) -> None:
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

    assert calls == ["save", "revoke", "cleanup"]


def test_login_cleans_old_keyring_token_when_server_revoke_raises(
    monkeypatch,
) -> None:
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

    def raise_revoke(_credentials: Credentials) -> bool:
        calls.append("revoke")
        raise OSError("network failure")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token", raise_revoke
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    with pytest.raises(OSError, match="network failure"):
        auth_module.login()

    assert calls == ["save", "revoke", "cleanup"]


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

    assert calls == ["save", "verify", "revoke", "cleanup"]


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


def test_login_cleans_old_keyring_when_both_token_ids_are_missing(
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
    cleaned: list[tuple[Credentials, Credentials]] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: cleaned.append((old, current)) or True,
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
            "a token ID and could not be revoked.",
            "TOKEN_REVOKE_FAILED",
        )
    ]
    assert cleaned == [(old_creds, new_creds)]


def test_device_login_save_failure_revokes_new_token(monkeypatch) -> None:
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
        lambda credentials, **kwargs: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    with pytest.raises(OSError, match="disk full"):
        auth_module.login()

    assert calls == ["save", "revoke"]


@pytest.mark.parametrize(
    "revoke_outcome",
    [False, OSError("network unavailable")],
    ids=["returns-false", "raises"],
)
def test_device_login_save_failure_warns_when_revoke_fails(
    monkeypatch, revoke_outcome
) -> None:
    credentials = _make_credentials(access_token="new", token_id="tok_new")

    def fail_save(_credentials: Credentials, **kwargs) -> str:
        raise OSError("disk full")

    def fail_revoke(_credentials: Credentials, **kwargs) -> bool:
        assert kwargs == {"warn_on_failure": False}
        if isinstance(revoke_outcome, Exception):
            raise revoke_outcome
        return revoke_outcome

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", fail_save
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token", fail_revoke
    )
    warnings: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "osmosis_ai.cli.console.console.print_warning",
        lambda message, **kwargs: warnings.append((message, kwargs.get("code"))),
    )

    with pytest.raises(OSError, match="disk full"):
        auth_module.save_device_credentials_or_revoke(credentials)

    assert warnings == [
        (
            "The new device-login token could not be saved or revoked.",
            "TOKEN_REVOKE_FAILED",
        )
    ]


def test_device_login_stops_before_minting_without_a_pinned_keyring(
    monkeypatch,
) -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setenv("OSMOSIS_TOKEN_STORE", "keyring")
    monkeypatch.setattr("keyring.get_keyring", lambda: FailKeyring())
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: pytest.fail("device flow must not mint a token without a keyring"),
    )

    with pytest.raises(CLIError) as exc_info:
        auth_module.login()

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_token_login_stops_before_verifying_without_a_pinned_keyring(
    monkeypatch,
) -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    monkeypatch.setenv("OSMOSIS_TOKEN_STORE", "keyring")
    monkeypatch.setattr("keyring.get_keyring", lambda: FailKeyring())
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: pytest.fail("token must not be verified without a keyring"),
    )

    with pytest.raises(CLIError) as exc_info:
        auth_module._login_with_token(token="token")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_device_login_proceeds_without_a_keyring_by_default(
    tmp_path, monkeypatch
) -> None:
    """Headless hosts fall back to the credentials file instead of aborting."""
    from keyring.backends.fail import Keyring as FailKeyring

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE",
        tmp_path / "creds.json",
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.delenv("OSMOSIS_TOKEN_STORE", raising=False)
    monkeypatch.setattr("keyring.get_keyring", lambda: FailKeyring())

    credentials = _make_credentials()
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: (_make_login_result(), credentials),
    )
    saved: list[Credentials] = []

    def fake_save(creds: Credentials, **kwargs) -> str:
        saved.append(creds)
        return "file"

    monkeypatch.setattr(auth_module, "save_device_credentials_or_revoke", fake_save)

    result = auth_module.login()

    assert saved == [credentials]
    assert result.exit_code == 0


def _write_old_login(creds_file, monkeypatch, *, token_store: str) -> None:
    """Persist a previous login for the default platform in *creds_file*."""
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.delenv("OSMOSIS_PLATFORM_URL", raising=False)
    monkeypatch.delenv("OSMOSIS_TOKEN_PLATFORM_URL", raising=False)
    entry = _make_credentials(access_token="old-token", token_id="tok_old").to_dict()
    entry["token_store"] = token_store
    if token_store == "keyring":
        entry.pop("access_token")
    creds_file.write_text(json.dumps(entry))


def _record_warnings(monkeypatch) -> list[str | None]:
    codes: list[str | None] = []
    monkeypatch.setattr(
        "osmosis_ai.cli.console.console.print_warning",
        lambda message, *, code=None, **kwargs: codes.append(code),
    )
    return codes


def test_device_login_replaces_an_unreadable_keyring_login_by_default(
    tmp_path, monkeypatch
) -> None:
    """A host whose keyring went away (e.g. over SSH) can still log in again."""
    from keyring.backends.fail import Keyring as FailKeyring

    _write_old_login(tmp_path / "creds.json", monkeypatch, token_store="keyring")
    monkeypatch.delenv("OSMOSIS_TOKEN_STORE", raising=False)
    monkeypatch.setattr("keyring.get_keyring", lambda: FailKeyring())
    credentials = _make_credentials(access_token="new-token", token_id="tok_new")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: (_make_login_result(), credentials),
    )
    saved: list[Credentials] = []

    def fake_save(creds: Credentials, **kwargs) -> str:
        saved.append(creds)
        return "file"

    monkeypatch.setattr(auth_module, "save_device_credentials_or_revoke", fake_save)
    warnings = _record_warnings(monkeypatch)

    result = auth_module.login()

    assert result.exit_code == 0
    assert saved == [credentials]
    # The old token cannot be read, so the user is told it was not revoked.
    assert warnings == ["TOKEN_REVOKE_FAILED"]


def test_pinned_keyring_login_fails_when_the_old_token_cannot_be_read(
    tmp_path, monkeypatch
) -> None:
    _write_old_login(tmp_path / "creds.json", monkeypatch, token_store="keyring")
    monkeypatch.setenv("OSMOSIS_TOKEN_STORE", "keyring")

    def locked(account: str) -> str | None:
        raise CLIError("keyring locked", code="KEYRING_UNAVAILABLE")

    monkeypatch.setattr("osmosis_ai.platform.auth.credentials._keyring_get", locked)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: pytest.fail("no token may be minted when the keyring is pinned"),
    )

    with pytest.raises(CLIError) as exc_info:
        auth_module.login()

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_second_headless_login_warns_once(tmp_path, monkeypatch) -> None:
    """Replacing a file-backed login without a keyring reports only the fallback."""
    from keyring.backends.fail import Keyring as FailKeyring

    creds_file = tmp_path / "creds.json"
    _write_old_login(creds_file, monkeypatch, token_store="file")
    monkeypatch.delenv("OSMOSIS_TOKEN_STORE", raising=False)
    monkeypatch.setattr("keyring.get_keyring", lambda: FailKeyring())
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: (
            _make_login_result(),
            _make_credentials(access_token="new-token", token_id="tok_new"),
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda *args, **kwargs: True,
    )
    warnings = _record_warnings(monkeypatch)

    result = auth_module.login()

    assert result.exit_code == 0
    assert warnings == ["KEYRING_UNAVAILABLE"]
    platforms = json.loads(creds_file.read_text())["platforms"]
    assert [entry["access_token"] for entry in platforms.values()] == ["new-token"]


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
        "osmosis_ai.platform.cli.workspace_directory_context.resolve_optional_git_identity",
        lambda: None,
    )

    result = auth_module.whoami()

    assert result.data["email"] == "a@example.com"
    assert result.data["name"] == "User"
    assert result.data["effective_source"] == "keyring"
    assert result.data["persistent_login"] is True
    assert result.data["workspace"] is None
    assert all(field.label != "Workspace" for field in result.fields)

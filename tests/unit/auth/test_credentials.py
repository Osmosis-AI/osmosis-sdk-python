"""Tests for osmosis_ai.platform.auth.credentials - user-scoped authentication."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import patch

import pytest
from keyring.errors import KeyringError

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import DEFAULT_PLATFORM_URL, normalize_platform_url
from osmosis_ai.platform.auth.credentials import (
    KEYRING_ACCOUNT,
    TOKEN_STORE_ENV,
    TOKEN_STORE_FILE,
    TOKEN_STORE_KEYRING,
    Credentials,
    UserInfo,
    get_credential_store,
    keyring_account_for_platform,
)

DEFAULT_PLATFORM = normalize_platform_url(DEFAULT_PLATFORM_URL)
STAGING_PLATFORM = "https://platform-staging.osmosis.ai"
LOCAL_PLATFORM = "http://localhost:3000"


@pytest.fixture(autouse=True)
def _default_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSMOSIS_PLATFORM_URL", raising=False)
    monkeypatch.delenv("OSMOSIS_TOKEN_PLATFORM_URL", raising=False)


def _make_credentials(
    *,
    access_token: str = "test-token",
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
    token_id: str | None = None,
    user_id: str = "user_1",
    email: str = "user@example.com",
) -> Credentials:
    now = datetime.now(UTC)
    return Credentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=expires_at or (now + timedelta(minutes=5)),
        created_at=created_at or now,
        user=UserInfo(id=user_id, email=email, name="User"),
        token_id=token_id,
    )


def _platform_entry(
    data: dict,
    platform_url: str = DEFAULT_PLATFORM,
) -> dict:
    return data["platforms"][normalize_platform_url(platform_url)]


def test_credentials_roundtrip_preserves_tz_aware_expires_at() -> None:
    now_utc = datetime.now(UTC)
    creds = _make_credentials(
        expires_at=now_utc + timedelta(minutes=5),
        created_at=now_utc,
    )
    data = creds.to_dict()
    loaded = Credentials.from_dict(data)
    assert loaded.expires_at.tzinfo is not None
    assert loaded.is_expired() is False


def test_from_dict_rejects_naive_expires_at() -> None:
    creds = _make_credentials()
    data = creds.to_dict()
    data["expires_at"] = datetime.now().isoformat()  # naive
    try:
        Credentials.from_dict(data)
    except ValueError as exc:
        assert "expires_at must be timezone-aware" in str(exc)
    else:
        raise AssertionError("Expected ValueError for naive expires_at")


# ---------------------------------------------------------------------------
# save / load with keyring
# ---------------------------------------------------------------------------


def test_save_uses_keyring_when_available(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    stored: dict[str, str] = {}

    def fake_set(account: str, token: str) -> bool:
        stored[account] = token
        return True

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set", side_effect=fake_set
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=True
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        store = save_credentials(_make_credentials())

    assert store == TOKEN_STORE_KEYRING
    data = json.loads(creds_file.read_text())
    assert "active_platform_url" not in data
    entry = _platform_entry(data)
    platform_account = keyring_account_for_platform(DEFAULT_PLATFORM)
    keyring_account = entry["keyring_account"]
    assert keyring_account.startswith(f"{platform_account}:")
    assert stored.get(keyring_account) == "test-token"
    assert stored.get(platform_account) is None
    assert stored.get(KEYRING_ACCOUNT) is None
    assert "access_token" not in entry
    assert entry["token_store"] == TOKEN_STORE_KEYRING


def test_save_cleans_up_old_keyring_on_account_change(tmp_path, monkeypatch) -> None:
    """Re-logging as a different user should clean up the old keyring entry."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    # Simulate existing credentials for alice (legacy email-based keyring)
    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    old_data["user"]["email"] = "alice@example.com"
    creds_file.write_text(json.dumps(old_data))

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    def fake_set(account: str, token: str) -> bool:
        return True

    # Now save credentials for bob
    bob_creds = Credentials(
        access_token="bob-token",
        token_type="Bearer",
        expires_at=_make_credentials().expires_at,
        created_at=_make_credentials().created_at,
        user=UserInfo(id="user_2", email="bob@example.com", name="Bob"),
    )

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            side_effect=fake_set,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        save_credentials(bob_creds)

    # Both the fixed account and the legacy alice account should be cleaned up
    assert KEYRING_ACCOUNT in deleted_accounts
    assert "alice@example.com" in deleted_accounts


def test_save_writes_new_token_and_metadata_before_legacy_cleanup(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    # Simulate existing credentials with legacy email-based keyring
    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(old_data))

    events: list[tuple[str, str]] = []

    def fake_delete(account: str) -> bool:
        events.append(("delete", account))
        return True

    def fake_set(account: str, token: str) -> bool:
        events.append(("set", account))
        return True

    def fake_write(path, data, *, mode):
        events.append(
            ("metadata", data["platforms"][DEFAULT_PLATFORM]["keyring_account"])
        )
        path.write_text(json.dumps(data))

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            side_effect=fake_set,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials.atomic_write_json",
            side_effect=fake_write,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        save_credentials(_make_credentials())

    current_account = events[0][1]
    assert current_account.startswith(
        f"{keyring_account_for_platform(DEFAULT_PLATFORM)}:"
    )
    assert events[0] == ("set", current_account)
    assert events[1] == ("metadata", current_account)
    assert ("delete", current_account) not in events
    assert ("delete", KEYRING_ACCOUNT) in events[2:]
    assert ("delete", "user@example.com") in events[2:]


def test_save_requires_keyring_and_does_not_write_plaintext(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    with patch("osmosis_ai.platform.auth.credentials._keyring_set", return_value=False):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"
    assert "OSMOSIS_TOKEN" in str(exc_info.value)
    assert not creds_file.exists()


def test_load_from_keyring(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    # Write metadata without token
    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    def fake_get(account: str) -> str | None:
        if account == KEYRING_ACCOUNT:
            return "keyring-secret"
        return None

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_get",
        side_effect=fake_get,
    ):
        from osmosis_ai.platform.auth.credentials import load_credentials

        loaded = load_credentials()

    assert loaded is not None
    assert loaded.access_token == "keyring-secret"
    assert loaded.user.email == "user@example.com"


def test_load_returns_none_when_keyring_entry_missing(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    # Both fixed account and legacy email return None
    with (
        patch("osmosis_ai.platform.auth.credentials._keyring_get", return_value=None),
        patch("osmosis_ai.cli.console.console.print_warning") as mock_warn,
    ):
        from osmosis_ai.platform.auth.credentials import load_credentials

        loaded = load_credentials()

    assert loaded is None
    mock_warn.assert_called_once()
    assert (
        "Token not found for the current Osmosis platform"
        in (mock_warn.call_args.args[0])
    )
    assert mock_warn.call_args.kwargs.get("code") == "TOKEN_NOT_FOUND"


def test_load_from_file_fallback(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    creds = _make_credentials(token_id="tok_abc")
    data = creds.to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))

    from osmosis_ai.platform.auth.credentials import load_credentials

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "test-token"
    assert loaded.token_id == "tok_abc"


def test_load_legacy_default_file_ignores_non_default_platform(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    stored = _make_credentials()
    data = stored.to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))

    from osmosis_ai.platform.auth.credentials import load_credentials

    assert load_credentials() is None


def test_platform_registry_loads_current_platform_entry(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    prod = _make_credentials()
    prod_data = prod.to_dict()
    prod_data["platform_url"] = DEFAULT_PLATFORM
    prod_data["token_store"] = TOKEN_STORE_FILE
    staging = _make_credentials(token_id="tok_staging")
    staging_data = staging.to_dict()
    staging_data["access_token"] = "staging-token"
    staging_data["platform_url"] = STAGING_PLATFORM
    staging_data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(
        json.dumps(
            {
                "version": 2,
                "platforms": {
                    DEFAULT_PLATFORM: prod_data,
                    STAGING_PLATFORM: staging_data,
                },
            }
        )
    )

    from osmosis_ai.platform.auth.credentials import load_credentials

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "staging-token"
    assert loaded.token_id == "tok_staging"


def test_keyring_credentials_are_isolated_across_supported_platforms(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    keyring_tokens: dict[str, str] = {}

    def fake_set(account: str, token: str) -> bool:
        keyring_tokens[account] = token
        return True

    def fake_get(account: str) -> str | None:
        return keyring_tokens.get(account)

    def fake_delete(account: str) -> bool:
        keyring_tokens.pop(account, None)
        return True

    credentials_by_platform = {
        DEFAULT_PLATFORM: _make_credentials(
            access_token="prod-token", token_id="tok_prod", email="prod@example.com"
        ),
        STAGING_PLATFORM: _make_credentials(
            access_token="staging-token",
            token_id="tok_staging",
            email="staging@example.com",
        ),
        LOCAL_PLATFORM: _make_credentials(
            access_token="local-token",
            token_id="tok_local",
            email="local@example.com",
        ),
    }

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            side_effect=fake_set,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_get",
            side_effect=fake_get,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import (
            load_credentials,
            save_credentials,
        )

        for platform_url, credentials in credentials_by_platform.items():
            monkeypatch.setenv("OSMOSIS_PLATFORM_URL", platform_url)
            assert save_credentials(credentials) == TOKEN_STORE_KEYRING

        for platform_url, expected in credentials_by_platform.items():
            monkeypatch.setenv("OSMOSIS_PLATFORM_URL", platform_url)
            loaded = load_credentials()
            assert loaded is not None
            assert loaded.access_token == expected.access_token
            assert loaded.user.email == expected.user.email
            assert loaded.token_id == expected.token_id

    registry = json.loads(creds_file.read_text())
    assert set(registry["platforms"]) == set(credentials_by_platform)
    accounts = {entry["keyring_account"] for entry in registry["platforms"].values()}
    assert len(accounts) == 3
    for platform_url, entry in registry["platforms"].items():
        assert entry["keyring_account"].startswith(
            f"{keyring_account_for_platform(platform_url)}:"
        )


# ---------------------------------------------------------------------------
# load: legacy file without token_store field
# ---------------------------------------------------------------------------


def test_load_legacy_file_without_token_store(tmp_path, monkeypatch) -> None:
    """Files from before keyring support have no token_store; default to 'file'."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    creds = _make_credentials()
    data = creds.to_dict()
    # Simulate legacy: no token_store key
    assert "token_store" not in data
    creds_file.write_text(json.dumps(data))

    from osmosis_ai.platform.auth.credentials import load_credentials

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "test-token"


# ---------------------------------------------------------------------------
# load: environment variable has highest priority
# ---------------------------------------------------------------------------


def test_load_credentials_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token-abc")
    from osmosis_ai.platform.auth.credentials import load_credentials

    creds = load_credentials()
    assert creds is not None
    assert creds.access_token == "env-token-abc"
    assert creds.user.id == ""  # minimal user info for env token


def test_load_credentials_can_skip_env_token(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token-abc")

    stored = _make_credentials(token_id="tok_stored")
    data = stored.to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))

    from osmosis_ai.platform.auth.credentials import load_credentials

    assert load_credentials().access_token == "env-token-abc"
    loaded = load_credentials(include_env=False)
    assert loaded is not None
    assert loaded.access_token == "test-token"
    assert loaded.token_id == "tok_stored"


def test_load_env_token_requires_platform_binding_for_staging(monkeypatch) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "staging-token")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)

    from osmosis_ai.platform.auth.credentials import load_credentials

    with pytest.raises(CLIError) as exc_info:
        load_credentials()

    assert exc_info.value.code == "ENV_TOKEN_PLATFORM_REQUIRED"


def test_load_env_token_accepts_matching_platform_binding(monkeypatch) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "staging-token")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    monkeypatch.setenv("OSMOSIS_TOKEN_PLATFORM_URL", f"{STAGING_PLATFORM}/")

    from osmosis_ai.platform.auth.credentials import load_credentials

    credentials = load_credentials()
    assert credentials is not None
    assert credentials.access_token == "staging-token"


# ---------------------------------------------------------------------------
# load: version mismatch
# ---------------------------------------------------------------------------


def test_load_fails_closed_for_version_mismatch(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"version": 1, "workspaces": {}}))

    from osmosis_ai.platform.auth.credentials import load_credentials

    original = creds_file.read_text()
    with pytest.raises(CLIError) as exc_info:
        load_credentials()

    assert exc_info.value.code == "CREDENTIALS_VERSION_CHANGED"
    assert creds_file.read_text() == original


def test_credentials_version_error_is_structured_json_in_json_mode(
    tmp_path, monkeypatch, capsys
) -> None:
    """Unknown metadata versions surface as a structured error."""
    from osmosis_ai.cli import main as cli

    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"version": 1, "workspaces": {}}))

    assert cli.main(["--json", "auth", "whoami"]) == 1

    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "CREDENTIALS_VERSION_CHANGED"
    assert "left unchanged" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_clears_keyring_and_file(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=fake_delete,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        result = delete_credentials()

    assert result is True
    # Fixed account is always cleaned up; legacy email is also cleaned
    assert KEYRING_ACCOUNT in deleted_accounts
    assert "user@example.com" in deleted_accounts
    assert not creds_file.exists()


def test_delete_preserves_metadata_when_keyring_delete_fails(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=False
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        with pytest.raises(CLIError) as exc_info:
            delete_credentials()

    assert exc_info.value.code == "KEYRING_CLEANUP_FAILED"
    assert creds_file.exists()


def test_delete_with_corrupt_json_preserves_file_and_keyring(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    creds_file.write_text("{invalid json!!!}")

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=lambda account: pytest.fail(
            f"corrupt metadata must not delete keyring account {account}"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        with pytest.raises(CLIError) as exc_info:
            delete_credentials()

    assert exc_info.value.code == "CREDENTIALS_PARSE_FAILED"
    assert creds_file.read_text() == "{invalid json!!!}"


def test_delete_with_unknown_metadata_version_preserves_file_and_keyring(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    contents = json.dumps({"version": 999, "platforms": {}})
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=lambda account: pytest.fail(
            f"unknown metadata must not delete keyring account {account}"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        with pytest.raises(CLIError) as exc_info:
            delete_credentials()

    assert exc_info.value.code == "CREDENTIALS_VERSION_CHANGED"
    assert creds_file.read_text() == contents


def test_delete_with_malformed_registry_preserves_file_and_keyring(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    contents = json.dumps({"version": 2, "platforms": []})
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=lambda account: pytest.fail(
            f"malformed metadata must not delete keyring account {account}"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        with pytest.raises(CLIError) as exc_info:
            delete_credentials()

    assert exc_info.value.code == "CREDENTIALS_PARSE_FAILED"
    assert creds_file.read_text() == contents


def test_delete_file_only(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    creds = _make_credentials()
    data = creds.to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=True
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        assert delete_credentials() is True

    assert not creds_file.exists()


# ---------------------------------------------------------------------------
# Keyring backend failures must not masquerade as successful cleanup
# ---------------------------------------------------------------------------


def test_keyring_delete_fails_without_a_backend() -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    from osmosis_ai.platform.auth.credentials import _keyring_delete

    with patch("keyring.get_keyring", return_value=FailKeyring()):
        with pytest.raises(CLIError) as exc_info:
            _keyring_delete("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_explicit_delete_removes_metadata_without_a_keyring(
    tmp_path, monkeypatch
) -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    entry = _make_credentials(token_id="tok_saved").to_dict()
    entry.pop("access_token")
    entry["platform_url"] = DEFAULT_PLATFORM
    entry["token_store"] = TOKEN_STORE_KEYRING
    entry["keyring_account"] = f"{keyring_account_for_platform(DEFAULT_PLATFORM)}:saved"
    creds_file.write_text(
        json.dumps({"version": 2, "platforms": {DEFAULT_PLATFORM: entry}})
    )

    with (
        patch("keyring.get_keyring", return_value=FailKeyring()),
        patch("osmosis_ai.cli.console.console.print_warning") as mock_warn,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        assert delete_credentials(tolerate_keyring_unavailable=True) is True

    assert not creds_file.exists()
    mock_warn.assert_called_once()


def test_keyring_delete_fails_when_the_backend_reports_no_keyring() -> None:
    """The host has a backend object, but keyring still resolves none."""
    from keyring.errors import NoKeyringError

    from osmosis_ai.platform.auth.credentials import _keyring_delete

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch("keyring.delete_password", side_effect=NoKeyringError("no backend")),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_delete("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_keyring_delete_reports_a_real_failure_without_warning_itself() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_delete

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch("keyring.delete_password", side_effect=RuntimeError("keychain locked")),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_delete("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_keyring_delete_treats_missing_password_as_already_clean() -> None:
    from keyring.errors import PasswordDeleteError

    from osmosis_ai.platform.auth.credentials import _keyring_delete

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch(
            "keyring.delete_password",
            side_effect=PasswordDeleteError("missing"),
        ),
    ):
        assert _keyring_delete("platform:abc") is True


def test_keyring_read_error_is_not_misreported_as_logged_out() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_get

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch("keyring.get_password", side_effect=KeyringError("locked")),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_get("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"
    assert "Unlock or repair" in str(exc_info.value)


def test_keyring_read_wraps_non_keyring_backend_failure() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_get

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch(
            "keyring.get_password",
            side_effect=RuntimeError("backend disconnected"),
        ),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_get("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_keyring_read_fails_without_a_backend() -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    from osmosis_ai.platform.auth.credentials import _keyring_get

    with patch("keyring.get_keyring", return_value=FailKeyring()):
        with pytest.raises(CLIError) as exc_info:
            _keyring_get("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_keyring_read_fails_when_backend_reports_no_keyring() -> None:
    from keyring.errors import NoKeyringError

    from osmosis_ai.platform.auth.credentials import _keyring_get

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch("keyring.get_password", side_effect=NoKeyringError("no backend")),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_get("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_keyring_write_error_is_not_treated_as_missing_backend() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_set

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch("keyring.set_password", side_effect=KeyringError("locked")),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_set("platform:abc", "token")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"
    assert "Unlock or repair" in str(exc_info.value)


def test_keyring_write_wraps_non_keyring_backend_failure() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_set

    with (
        patch("keyring.get_keyring", return_value=object()),
        patch(
            "keyring.set_password",
            side_effect=RuntimeError("backend disconnected"),
        ),
    ):
        with pytest.raises(CLIError) as exc_info:
            _keyring_set("platform:abc", "token")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


@pytest.mark.parametrize("operation", ["get", "set", "delete"])
def test_keyring_operations_wrap_backend_discovery_failure(operation: str) -> None:
    from osmosis_ai.platform.auth.credentials import (
        _keyring_delete,
        _keyring_get,
        _keyring_set,
    )

    with patch(
        "keyring.get_keyring",
        side_effect=RuntimeError("backend discovery failed"),
    ):
        with pytest.raises(CLIError) as exc_info:
            if operation == "get":
                _keyring_get("platform:abc")
            elif operation == "set":
                _keyring_set("platform:abc", "token")
            else:
                _keyring_delete("platform:abc")

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"


def test_delete_does_not_warn_when_the_token_lived_in_the_file(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    data = _make_credentials().to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=False
        ),
        patch("osmosis_ai.cli.console.console.print_warning") as mock_warn,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        assert delete_credentials() is True

    assert not creds_file.exists()
    mock_warn.assert_not_called()


def test_save_without_keyring_uses_env_token_guidance(tmp_path, monkeypatch) -> None:
    from keyring.backends.fail import Keyring as FailKeyring

    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    with patch("keyring.get_keyring", return_value=FailKeyring()):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"
    assert "OSMOSIS_TOKEN" in str(exc_info.value)
    assert not creds_file.exists()


# ---------------------------------------------------------------------------
# A failed replacement must leave the existing metadata untouched
# ---------------------------------------------------------------------------


def test_save_without_keyring_preserves_existing_metadata(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    # Simulate existing keyring-based credentials
    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(old_data))

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=lambda account: pytest.fail(
                "failed save must not delete existing keyring entries"
            ),
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            return_value=False,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "KEYRING_UNAVAILABLE"
    assert json.loads(creds_file.read_text()) == old_data


@pytest.mark.parametrize(
    ("contents", "error_code"),
    [
        ("{invalid json!!!}", "CREDENTIALS_PARSE_FAILED"),
        (json.dumps({"version": 999, "platforms": {}}), "CREDENTIALS_VERSION_CHANGED"),
    ],
)
def test_save_preserves_invalid_shared_metadata(
    tmp_path, monkeypatch, contents: str, error_code: str
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_set",
        side_effect=lambda *args: pytest.fail(
            "invalid metadata must be rejected before writing the keyring"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == error_code
    assert creds_file.read_text() == contents


@pytest.mark.parametrize(
    "contents",
    [
        "{invalid json!!!}",
        json.dumps({"version": 1, "access_token": "legacy-token"}),
    ],
)
def test_explicit_save_backs_up_invalid_metadata_and_recovers(
    tmp_path, monkeypatch, contents: str
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    creds_file.write_text(contents)

    with (
        patch("osmosis_ai.platform.auth.credentials._keyring_set", return_value=True),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            return_value=True,
        ),
        patch("osmosis_ai.cli.console.console.print_warning"),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        save_credentials(_make_credentials(), recover_invalid_metadata=True)

    assert creds_file.exists()
    assert json.loads(creds_file.read_text())["version"] == 2
    assert (tmp_path / "creds.json.bak").read_text() == contents


def test_token_without_id_uses_random_keyring_account_suffix() -> None:
    from osmosis_ai.platform.auth.credentials import _keyring_account_for_credentials

    first = _keyring_account_for_credentials(
        _make_credentials(access_token="shared-secret"), DEFAULT_PLATFORM
    )
    second = _keyring_account_for_credentials(
        _make_credentials(access_token="shared-secret"), DEFAULT_PLATFORM
    )

    token_digest = sha256(b"shared-secret").hexdigest()[:24]
    assert first != second
    assert token_digest not in first
    assert token_digest not in second


@pytest.mark.parametrize(
    "platforms",
    [
        [],
        {DEFAULT_PLATFORM: "damaged-entry"},
    ],
)
def test_save_rejects_malformed_registry_before_writing_keyring(
    tmp_path, monkeypatch, platforms: object
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    contents = json.dumps({"version": 2, "platforms": platforms})
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_set",
        side_effect=lambda *args: pytest.fail(
            "malformed metadata must be rejected before writing the keyring"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "CREDENTIALS_PARSE_FAILED"
    assert creds_file.read_text() == contents


def test_save_rejects_unknown_token_store_in_another_platform(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    prod = _make_credentials().to_dict()
    prod["platform_url"] = DEFAULT_PLATFORM
    prod["token_store"] = "future-store"
    contents = json.dumps({"version": 2, "platforms": {DEFAULT_PLATFORM: prod}})
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_set",
        side_effect=lambda *args: pytest.fail(
            "unknown storage must be rejected before writing the keyring"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "CREDENTIALS_PARSE_FAILED"
    assert creds_file.read_text() == contents


def test_save_rejects_platform_key_and_entry_url_mismatch(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    entry = _make_credentials().to_dict()
    entry["platform_url"] = STAGING_PLATFORM
    entry["token_store"] = TOKEN_STORE_FILE
    contents = json.dumps({"version": 2, "platforms": {DEFAULT_PLATFORM: entry}})
    creds_file.write_text(contents)

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_set",
        side_effect=lambda *args: pytest.fail(
            "contradictory metadata must be rejected before writing the keyring"
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(CLIError) as exc_info:
            save_credentials(_make_credentials())

    assert exc_info.value.code == "CREDENTIALS_PARSE_FAILED"
    assert creds_file.read_text() == contents


def test_save_preserves_unknown_registry_fields_for_other_platforms(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)

    prod = _make_credentials(access_token="prod-token").to_dict()
    prod["platform_url"] = DEFAULT_PLATFORM
    prod["token_store"] = TOKEN_STORE_FILE
    prod["future_entry_field"] = {"keep": True}
    registry = {
        "version": 2,
        "future_top_level": {"keep": True},
        "platforms": {DEFAULT_PLATFORM: prod},
    }
    creds_file.write_text(json.dumps(registry))

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            return_value=True,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            return_value=True,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        save_credentials(
            _make_credentials(access_token="staging-token", token_id="tok_stage")
        )

    saved = json.loads(creds_file.read_text())
    assert saved["future_top_level"] == {"keep": True}
    assert saved["platforms"][DEFAULT_PLATFORM] == prod
    assert STAGING_PLATFORM in saved["platforms"]


def test_save_metadata_failure_preserves_old_token_and_removes_new_token(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    old_account = f"{keyring_account_for_platform(DEFAULT_PLATFORM)}:old-token"
    old_entry = _make_credentials(
        access_token="old-token", token_id="tok_old"
    ).to_dict()
    old_entry.pop("access_token")
    old_entry["platform_url"] = DEFAULT_PLATFORM
    old_entry["token_store"] = TOKEN_STORE_KEYRING
    old_entry["keyring_account"] = old_account
    old_registry = {
        "version": old_entry["version"],
        "platforms": {DEFAULT_PLATFORM: old_entry},
    }
    creds_file.write_text(json.dumps(old_registry))

    keyring_tokens = {old_account: "old-token"}
    deleted_accounts: list[str] = []

    def fake_set(account: str, token: str) -> bool:
        keyring_tokens[account] = token
        return True

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        keyring_tokens.pop(account, None)
        return True

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            side_effect=fake_set,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials.atomic_write_json",
            side_effect=OSError("disk full"),
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(OSError, match="disk full"):
            save_credentials(
                _make_credentials(access_token="new-token", token_id="tok_new")
            )

    assert json.loads(creds_file.read_text()) == old_registry
    assert keyring_tokens == {old_account: "old-token"}
    assert old_account not in deleted_accounts
    assert len(deleted_accounts) == 1
    assert deleted_accounts[0].startswith(
        f"{keyring_account_for_platform(DEFAULT_PLATFORM)}:"
    )


def test_save_metadata_failure_preserves_original_error_when_rollback_fails(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    warnings: list[tuple[str, str]] = []

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            return_value=True,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=CLIError("keyring locked", code="KEYRING_UNAVAILABLE"),
        ),
        patch(
            "osmosis_ai.platform.auth.credentials.atomic_write_json",
            side_effect=OSError("disk full"),
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._warn",
            side_effect=lambda message, *, code: warnings.append((message, code)),
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        with pytest.raises(OSError, match="disk full"):
            save_credentials(_make_credentials(token_id="tok_new"))

    assert warnings == [
        (
            "Credential metadata could not be saved, and the new keyring "
            "entry could not be rolled back.",
            "KEYRING_CLEANUP_FAILED",
        )
    ]


def test_save_non_default_platform_preserves_legacy_default_keyring(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)

    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(old_data))

    deleted_accounts: list[str] = []
    stored: dict[str, str] = {}

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    def fake_set(account: str, token: str) -> bool:
        stored[account] = token
        return True

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            side_effect=fake_set,
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        store = save_credentials(_make_credentials())

    assert store == TOKEN_STORE_KEYRING
    data = json.loads(creds_file.read_text())
    staging_account = data["platforms"][STAGING_PLATFORM]["keyring_account"]
    assert staging_account.startswith(
        f"{keyring_account_for_platform(STAGING_PLATFORM)}:"
    )
    assert KEYRING_ACCOUNT not in deleted_accounts
    assert staging_account not in deleted_accounts
    assert stored[staging_account] == "test-token"

    assert "active_platform_url" not in data
    assert DEFAULT_PLATFORM in data["platforms"]
    assert STAGING_PLATFORM in data["platforms"]
    assert data["platforms"][DEFAULT_PLATFORM]["keyring_account"] == KEYRING_ACCOUNT
    assert data["platforms"][STAGING_PLATFORM]["keyring_account"] == staging_account


# ---------------------------------------------------------------------------
# Backward compat: load credentials from legacy email-based keyring
# ---------------------------------------------------------------------------


def test_load_falls_back_to_legacy_email_keyring(tmp_path, monkeypatch) -> None:
    """Tokens stored under the old email-based account can still be loaded."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    def fake_get(account: str) -> str | None:
        # Only the legacy email-based entry exists
        if account == "user@example.com":
            return "legacy-keyring-secret"
        return None

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_get",
        side_effect=fake_get,
    ):
        from osmosis_ai.platform.auth.credentials import load_credentials

        loaded = load_credentials()

    assert loaded is not None
    assert loaded.access_token == "legacy-keyring-secret"


def test_non_default_platform_ignores_foreign_legacy_keyring_accounts(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    staging = _make_credentials(token_id="tok_staging").to_dict()
    staging.pop("access_token")
    staging["platform_url"] = STAGING_PLATFORM
    staging["token_store"] = TOKEN_STORE_KEYRING
    staging["keyring_account"] = (
        f"{keyring_account_for_platform(DEFAULT_PLATFORM)}:prod-token"
    )
    staging["user"]["email"] = "production@example.com"
    creds_file.write_text(
        json.dumps(
            {
                "version": 2,
                "platforms": {STAGING_PLATFORM: staging},
            }
        )
    )

    queried_accounts: list[str] = []

    def fake_get(account: str) -> str | None:
        queried_accounts.append(account)
        if account in {KEYRING_ACCOUNT, "production@example.com"}:
            return "production-secret"
        return None

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_get",
            side_effect=fake_get,
        ),
        patch("osmosis_ai.cli.console.console.print_warning"),
    ):
        from osmosis_ai.platform.auth.credentials import load_credentials

        assert load_credentials() is None

    assert queried_accounts == [keyring_account_for_platform(STAGING_PLATFORM)]


def test_delete_credentials_removes_only_current_platform(
    tmp_path, monkeypatch
) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)

    prod = _make_credentials(token_id="tok_prod").to_dict()
    prod["platform_url"] = DEFAULT_PLATFORM
    prod["token_store"] = TOKEN_STORE_FILE
    staging = _make_credentials(token_id="tok_staging").to_dict()
    staging["access_token"] = "staging-token"
    staging["platform_url"] = STAGING_PLATFORM
    staging["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(
        json.dumps(
            {
                "version": 2,
                "platforms": {
                    DEFAULT_PLATFORM: prod,
                    STAGING_PLATFORM: staging,
                },
            }
        )
    )

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=fake_delete,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        assert delete_credentials() is True

    data = json.loads(creds_file.read_text())
    assert DEFAULT_PLATFORM in data["platforms"]
    assert STAGING_PLATFORM not in data["platforms"]
    assert KEYRING_ACCOUNT not in deleted_accounts


# ---------------------------------------------------------------------------
# P2 regression: delete with missing metadata file still cleans keyring
# ---------------------------------------------------------------------------


def test_delete_with_missing_file_still_cleans_keyring(tmp_path, monkeypatch) -> None:
    """Keyring cleanup happens even when the metadata file does not exist."""
    creds_file = tmp_path / "nonexistent_creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    with patch(
        "osmosis_ai.platform.auth.credentials._keyring_delete",
        side_effect=fake_delete,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        result = delete_credentials()

    # Keyring cleanup must be attempted even without a metadata file
    assert KEYRING_ACCOUNT in deleted_accounts
    # No file existed → nothing was actually deleted
    assert result is False


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------


def test_is_expired_true() -> None:
    creds = _make_credentials(expires_at=datetime.now(UTC) - timedelta(hours=1))
    assert creds.is_expired() is True


def test_is_expired_false() -> None:
    creds = _make_credentials()
    assert creds.is_expired() is False


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_includes_token_id() -> None:
    creds = _make_credentials(token_id="tok_123")
    d = creds.to_dict()
    assert d["token_id"] == "tok_123"


def test_to_dict_excludes_token_id_when_none() -> None:
    creds = _make_credentials()
    d = creds.to_dict()
    assert "token_id" not in d


# ---------------------------------------------------------------------------
# get_credential_store
# ---------------------------------------------------------------------------


def test_get_credential_store_env(monkeypatch) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "tok")
    assert get_credential_store() == TOKEN_STORE_ENV


def test_get_credential_store_can_ignore_env(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_TOKEN", "tok")
    data = _make_credentials().to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    assert get_credential_store(include_env=False) == TOKEN_STORE_KEYRING


def test_get_credential_store_keyring(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    data = _make_credentials().to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))
    assert get_credential_store() == TOKEN_STORE_KEYRING


def test_get_credential_store_file(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    data = _make_credentials().to_dict()
    data["token_store"] = TOKEN_STORE_FILE
    creds_file.write_text(json.dumps(data))
    assert get_credential_store() == TOKEN_STORE_FILE


def test_get_credential_store_uses_current_platform(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", STAGING_PLATFORM)
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    prod = _make_credentials().to_dict()
    prod["platform_url"] = DEFAULT_PLATFORM
    prod["token_store"] = TOKEN_STORE_FILE
    staging = _make_credentials().to_dict()
    staging.pop("access_token")
    staging["platform_url"] = STAGING_PLATFORM
    staging["token_store"] = TOKEN_STORE_KEYRING
    staging["keyring_account"] = keyring_account_for_platform(STAGING_PLATFORM)
    creds_file.write_text(
        json.dumps(
            {
                "version": 2,
                "platforms": {
                    DEFAULT_PLATFORM: prod,
                    STAGING_PLATFORM: staging,
                },
            }
        )
    )

    assert get_credential_store() == TOKEN_STORE_KEYRING


def test_get_credential_store_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE",
        tmp_path / "nonexistent.json",
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    assert get_credential_store() is None

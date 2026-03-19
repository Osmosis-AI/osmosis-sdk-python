"""Tests for osmosis_ai.platform.auth.credentials - user-scoped authentication."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from osmosis_ai.platform.auth.credentials import (
    KEYRING_ACCOUNT,
    TOKEN_STORE_ENV,
    TOKEN_STORE_FILE,
    TOKEN_STORE_KEYRING,
    Credentials,
    UserInfo,
    get_credential_store,
)


def _make_credentials(
    *,
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
    token_id: str | None = None,
) -> Credentials:
    now = datetime.now(timezone.utc)
    return Credentials(
        access_token="test-token",
        token_type="Bearer",
        expires_at=expires_at or (now + timedelta(minutes=5)),
        created_at=created_at or now,
        user=UserInfo(id="user_1", email="user@example.com", name="User"),
        token_id=token_id,
    )


def test_credentials_roundtrip_preserves_tz_aware_expires_at() -> None:
    now_utc = datetime.now(timezone.utc)
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
    # Token should be stored under the fixed KEYRING_ACCOUNT, not the email
    assert stored.get(KEYRING_ACCOUNT) == "test-token"
    data = json.loads(creds_file.read_text())
    assert "access_token" not in data
    assert data["token_store"] == TOKEN_STORE_KEYRING


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


def test_save_always_cleans_up_keyring_before_saving(tmp_path, monkeypatch) -> None:
    """Re-logging as the same user should still clean up old keyring entries."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    # Simulate existing credentials with legacy email-based keyring
    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(old_data))

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    def fake_set(account: str, token: str) -> bool:
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

        save_credentials(_make_credentials())

    # Fixed account is always cleaned up; legacy email is also cleaned
    assert KEYRING_ACCOUNT in deleted_accounts
    assert "user@example.com" in deleted_accounts


def test_save_falls_back_to_file(tmp_path, monkeypatch, capsys) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    with (
        patch("osmosis_ai.platform.auth.credentials._keyring_set", return_value=False),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=True
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        store = save_credentials(_make_credentials())

    assert store == TOKEN_STORE_FILE
    data = json.loads(creds_file.read_text())
    assert data["access_token"] == "test-token"
    assert data["token_store"] == TOKEN_STORE_FILE
    captured = capsys.readouterr()
    assert "keyring unavailable" in captured.err


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


def test_load_returns_none_when_keyring_entry_missing(
    tmp_path, monkeypatch, capsys
) -> None:
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
    with patch("osmosis_ai.platform.auth.credentials._keyring_get", return_value=None):
        from osmosis_ai.platform.auth.credentials import load_credentials

        loaded = load_credentials()

    assert loaded is None
    captured = capsys.readouterr()
    assert "not found in keyring" in captured.err


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


# ---------------------------------------------------------------------------
# load: version mismatch
# ---------------------------------------------------------------------------


def test_load_returns_none_for_version_mismatch(tmp_path, monkeypatch, capsys) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"version": 1, "workspaces": {}}))

    from osmosis_ai.platform.auth.credentials import load_credentials

    assert load_credentials() is None
    captured = capsys.readouterr()
    assert "osmosis login" in captured.err


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


def test_delete_warns_when_keyring_delete_fails(tmp_path, monkeypatch, capsys) -> None:
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

        result = delete_credentials()

    # File is still deleted, but warning is emitted about keyring failure
    assert result is True  # file deletion counts
    assert not creds_file.exists()
    captured = capsys.readouterr()
    assert "could not remove token from system keyring" in captured.err


def test_delete_with_corrupt_json_still_cleans_keyring(tmp_path, monkeypatch) -> None:
    """Corrupt JSON must not prevent keyring cleanup (P2 fix)."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    creds_file.write_text("{invalid json!!!}")

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

    # Keyring cleanup still happens even with corrupt metadata
    assert KEYRING_ACCOUNT in deleted_accounts
    assert not creds_file.exists()


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
# P1 regression: keyring→file fallback must clean up old keyring token
# ---------------------------------------------------------------------------


def test_save_fallback_to_file_cleans_up_old_keyring(tmp_path, monkeypatch) -> None:
    """When save falls back from keyring to file, old keyring token is removed."""
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    # Simulate existing keyring-based credentials
    old_data = _make_credentials().to_dict()
    old_data.pop("access_token")
    old_data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(old_data))

    deleted_accounts: list[str] = []

    def fake_delete(account: str) -> bool:
        deleted_accounts.append(account)
        return True

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete",
            side_effect=fake_delete,
        ),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_set",
            return_value=False,  # keyring unavailable → falls back to file
        ),
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        store = save_credentials(_make_credentials())

    assert store == TOKEN_STORE_FILE
    # Old keyring entry must have been cleaned up before fallback
    assert KEYRING_ACCOUNT in deleted_accounts


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
    creds = _make_credentials(
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
    )
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


def test_get_credential_store_keyring(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"token_store": TOKEN_STORE_KEYRING}))
    assert get_credential_store() == TOKEN_STORE_KEYRING


def test_get_credential_store_file(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"token_store": TOKEN_STORE_FILE}))
    assert get_credential_store() == TOKEN_STORE_FILE


def test_get_credential_store_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE",
        tmp_path / "nonexistent.json",
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    assert get_credential_store() is None

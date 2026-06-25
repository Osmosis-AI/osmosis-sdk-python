"""Tests for osmosis_ai.platform.auth.credentials - user-scoped authentication."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

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
STAGING_PLATFORM = "https://staging.osmosis.ai"


@pytest.fixture(autouse=True)
def _default_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSMOSIS_PLATFORM_URL", raising=False)


def _make_credentials(
    *,
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
    token_id: str | None = None,
) -> Credentials:
    now = datetime.now(UTC)
    return Credentials(
        access_token="test-token",
        token_type="Bearer",
        expires_at=expires_at or (now + timedelta(minutes=5)),
        created_at=created_at or now,
        user=UserInfo(id="user_1", email="user@example.com", name="User"),
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
    platform_account = keyring_account_for_platform(DEFAULT_PLATFORM)
    assert stored.get(platform_account) == "test-token"
    assert stored.get(KEYRING_ACCOUNT) is None
    data = json.loads(creds_file.read_text())
    assert "active_platform_url" not in data
    entry = _platform_entry(data)
    assert "access_token" not in entry
    assert entry["token_store"] == TOKEN_STORE_KEYRING
    assert entry["keyring_account"] == platform_account


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


def test_save_falls_back_to_file(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    with (
        patch("osmosis_ai.platform.auth.credentials._keyring_set", return_value=False),
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=True
        ),
        patch("osmosis_ai.cli.console.console.print_warning") as mock_warn,
    ):
        from osmosis_ai.platform.auth.credentials import save_credentials

        store = save_credentials(_make_credentials())

    assert store == TOKEN_STORE_FILE
    data = json.loads(creds_file.read_text())
    assert "active_platform_url" not in data
    entry = _platform_entry(data)
    assert entry["access_token"] == "test-token"
    assert entry["token_store"] == TOKEN_STORE_FILE
    # Warning routes through print_warning (output-mode aware) with a code,
    # not a raw stderr write that would corrupt the --json stderr contract.
    mock_warn.assert_called_once()
    assert "Keyring unavailable" in mock_warn.call_args.args[0]
    assert mock_warn.call_args.kwargs.get("code") == "KEYRING_UNAVAILABLE"


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


# ---------------------------------------------------------------------------
# load: version mismatch
# ---------------------------------------------------------------------------


def test_load_returns_none_for_version_mismatch(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"version": 1, "workspaces": {}}))

    from osmosis_ai.platform.auth.credentials import load_credentials

    with patch("osmosis_ai.cli.console.console.print_warning") as mock_warn:
        assert load_credentials() is None
    mock_warn.assert_called_once()
    assert "osmosis auth login" in mock_warn.call_args.args[0]
    assert mock_warn.call_args.kwargs.get("code") == "CREDENTIALS_VERSION_CHANGED"


def test_credentials_warning_is_structured_json_in_json_mode(
    tmp_path, monkeypatch, capsys
) -> None:
    """In --json mode a credentials warning is a JSON-lines envelope on stderr.

    This is the contract that the old raw ``sys.stderr.write`` broke: a plain
    "Warning: ..." line on stderr is not valid JSON Lines, so a machine reading
    `osmosis ... --json 2>err` would fail to parse err. Routing through
    print_warning emits a structured ``{"warning": {...}}`` envelope instead.
    """
    from osmosis_ai.cli.output.context import OutputFormat, override_output_context
    from osmosis_ai.platform.auth.credentials import load_credentials

    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    creds_file.write_text(json.dumps({"version": 1, "workspaces": {}}))

    with override_output_context(format=OutputFormat.json):
        assert load_credentials() is None

    # stderr must be parseable as JSON Lines, not raw text.
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["schema_version"] == 1
    assert payload["warning"]["code"] == "CREDENTIALS_VERSION_CHANGED"
    assert "osmosis auth login" in payload["warning"]["message"]


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


def test_delete_warns_when_keyring_delete_fails(tmp_path, monkeypatch) -> None:
    creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.CREDENTIALS_FILE", creds_file
    )

    creds = _make_credentials()
    data = creds.to_dict()
    data.pop("access_token")
    data["token_store"] = TOKEN_STORE_KEYRING
    creds_file.write_text(json.dumps(data))

    with (
        patch(
            "osmosis_ai.platform.auth.credentials._keyring_delete", return_value=False
        ),
        patch("osmosis_ai.cli.console.console.print_warning") as mock_warn,
    ):
        from osmosis_ai.platform.auth.credentials import delete_credentials

        result = delete_credentials()

    # File is still deleted, but a keyring-failure warning is emitted via
    # print_warning (output-mode aware) with a code.
    assert result is True  # file deletion counts
    assert not creds_file.exists()
    mock_warn.assert_called_once()
    assert "Could not remove token from system keyring" in mock_warn.call_args.args[0]
    assert mock_warn.call_args.kwargs.get("code") == "KEYRING_CLEANUP_FAILED"


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
    assert KEYRING_ACCOUNT not in deleted_accounts
    assert keyring_account_for_platform(STAGING_PLATFORM) in deleted_accounts
    assert stored[keyring_account_for_platform(STAGING_PLATFORM)] == "test-token"

    data = json.loads(creds_file.read_text())
    assert "active_platform_url" not in data
    assert DEFAULT_PLATFORM in data["platforms"]
    assert STAGING_PLATFORM in data["platforms"]
    assert data["platforms"][DEFAULT_PLATFORM]["keyring_account"] == KEYRING_ACCOUNT
    assert data["platforms"][STAGING_PLATFORM]["keyring_account"] == (
        keyring_account_for_platform(STAGING_PLATFORM)
    )


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

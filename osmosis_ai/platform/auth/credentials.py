"""Credential storage and retrieval for Osmosis CLI authentication.

Supports three token sources with descending priority:

    1. ``OSMOSIS_TOKEN`` environment variable  (CI / headless)
    2. System keyring  (macOS Keychain, GNOME Keyring, …)
    3. Platform-scoped plain-text JSON file
       (``~/.config/osmosis/credentials.json``)

When *saving*, the module tries the keyring first.  If successful the
JSON metadata file is written **without** the token.  If the keyring is
unavailable the token is stored in the JSON file and a warning is
emitted so the user knows the secret is on disk in clear text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import keyring
from keyring.backends.fail import Keyring as FailKeyring
from keyring.errors import PasswordDeleteError

from ._fileutil import atomic_write_json
from .config import (
    CREDENTIALS_FILE,
    CREDENTIALS_VERSION,
    DEFAULT_PLATFORM_URL,
    get_platform_url,
    normalize_platform_url,
)

if TYPE_CHECKING:
    from .flow import VerifyResult

# ---------------------------------------------------------------------------
# Keyring helpers
# ---------------------------------------------------------------------------

KEYRING_SERVICE = "osmosis-cli"
KEYRING_ACCOUNT = "default"
KEYRING_ACCOUNT_PREFIX = "platform:"

# Token store backend identifiers (persisted in credentials.json)
TOKEN_STORE_KEYRING = "keyring"
TOKEN_STORE_FILE = "file"
TOKEN_STORE_ENV = "env"


def _warn(message: str, *, code: str) -> None:
    """Emit an output-mode-aware warning from this low-level auth module.

    Routes through ``console.print_warning`` (not a raw ``sys.stderr.write``) so
    these best-effort diagnostics stay structured in ``--json`` mode instead of
    corrupting the stderr JSON-lines contract, and pause any active spinner in
    rich mode. The console is imported lazily to keep this module — imported on
    nearly every authenticated command via ``load_credentials`` — free of a CLI
    import at module load time.
    """
    from osmosis_ai.cli.console import console

    console.print_warning(message, code=code)


def keyring_account_for_platform(platform_url: str | None = None) -> str:
    """Return the keyring account for a platform-scoped CLI token."""
    normalized_url = normalize_platform_url(platform_url or get_platform_url())
    digest = sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    return f"{KEYRING_ACCOUNT_PREFIX}{digest}"


def _default_platform_url() -> str:
    return normalize_platform_url(DEFAULT_PLATFORM_URL)


def _is_default_platform_url(platform_url: str) -> bool:
    return normalize_platform_url(platform_url) == _default_platform_url()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _cleanup_legacy_keyring_entries(metadata: dict | None = None) -> None:
    """Delete any legacy email-based keyring entry from a previous version.

    Args:
        metadata: Pre-parsed metadata dict. If ``None``, reads the credentials
            file from disk to discover the old account name.

    Silently does nothing if the file is missing, corrupt, or
    no legacy entry exists.
    """
    if metadata is None:
        try:
            with open(CREDENTIALS_FILE, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return

    if not isinstance(metadata, dict):
        return

    if metadata.get("token_store") == TOKEN_STORE_KEYRING:
        old_account = metadata.get("user", {}).get("email", "")
        if old_account and old_account != KEYRING_ACCOUNT:
            _keyring_delete(old_account)


def _is_platform_registry(data: dict[str, Any]) -> bool:
    return isinstance(data.get("platforms"), dict)


def _legacy_entry_from_metadata(data: dict[str, Any]) -> dict[str, Any]:
    entry = {
        key: value
        for key, value in data.items()
        if key not in {"platforms", "active_platform_url"}
    }
    entry["platform_url"] = _default_platform_url()
    if entry.get("token_store", TOKEN_STORE_FILE) == TOKEN_STORE_KEYRING:
        entry.setdefault("keyring_account", KEYRING_ACCOUNT)
    return entry


def _registry_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "version": CREDENTIALS_VERSION,
        "platforms": {},
    }
    if not isinstance(metadata, dict):
        return registry
    if metadata.get("version") != CREDENTIALS_VERSION:
        return registry

    if _is_platform_registry(metadata):
        platforms = metadata.get("platforms", {})
        for raw_platform_url, raw_entry in platforms.items():
            if not isinstance(raw_entry, dict):
                continue
            platform_url = normalize_platform_url(
                raw_entry.get("platform_url") or str(raw_platform_url)
            )
            entry = dict(raw_entry)
            entry["platform_url"] = platform_url
            registry["platforms"][platform_url] = entry

        return registry

    default_platform_url = _default_platform_url()
    registry["platforms"][default_platform_url] = _legacy_entry_from_metadata(metadata)
    return registry


def _entry_for_platform(
    metadata: dict[str, Any], platform_url: str
) -> dict[str, Any] | None:
    if metadata.get("version") != CREDENTIALS_VERSION:
        return None

    normalized_url = normalize_platform_url(platform_url)
    if _is_platform_registry(metadata):
        entry = metadata.get("platforms", {}).get(normalized_url)
        return entry if isinstance(entry, dict) else None

    if normalized_url == _default_platform_url():
        return metadata
    return None


def _keyring_accounts_for_entry(
    entry: dict[str, Any] | None,
    platform_url: str,
) -> list[str]:
    accounts = [keyring_account_for_platform(platform_url)]
    if _is_default_platform_url(platform_url):
        accounts.append(KEYRING_ACCOUNT)

    if entry and entry.get("token_store", TOKEN_STORE_FILE) == TOKEN_STORE_KEYRING:
        account = entry.get("keyring_account")
        if isinstance(account, str):
            accounts.append(account)
        old_account = entry.get("user", {}).get("email", "")
        if (
            isinstance(old_account, str)
            and old_account
            and old_account != KEYRING_ACCOUNT
        ):
            accounts.append(old_account)

    return _dedupe(accounts)


def _cleanup_platform_keyring_entries(
    entry: dict[str, Any] | None,
    platform_url: str,
) -> bool:
    cleaned = True
    for account in _keyring_accounts_for_entry(entry, platform_url):
        cleaned = _keyring_delete(account) and cleaned
    return cleaned


def _resolve_entry_token(
    entry: dict[str, Any],
    platform_url: str,
) -> str | None:
    token_store = entry.get("token_store", TOKEN_STORE_FILE)
    if token_store != TOKEN_STORE_KEYRING:
        token = entry.get("access_token")
        return token if isinstance(token, str) else None

    for account in _keyring_accounts_for_entry(entry, platform_url):
        token = _keyring_get(account)
        if token is not None:
            return token
    return None


def _keyring_set(account: str, token: str) -> bool:
    """Store *token* in the system keyring. Returns ``True`` on success."""
    try:
        if isinstance(keyring.get_keyring(), FailKeyring):
            return False
        keyring.set_password(KEYRING_SERVICE, account, token)
        return True
    except Exception:
        return False


def _keyring_get(account: str) -> str | None:
    """Retrieve a token from the system keyring."""
    try:
        return keyring.get_password(KEYRING_SERVICE, account)
    except Exception:
        return None


def _keyring_delete(account: str) -> bool:
    """Delete a token from the system keyring. Returns ``True`` on success."""
    try:
        keyring.delete_password(KEYRING_SERVICE, account)
        return True
    except PasswordDeleteError:
        # Entry does not exist — nothing to clean up.
        return True
    except Exception:
        _warn(
            f"Could not remove token from keyring for {account}.",
            code="KEYRING_DELETE_FAILED",
        )
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UserInfo:
    """User information from authentication."""

    id: str
    email: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "email": self.email, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserInfo:
        return cls(
            id=data["id"],
            email=data["email"],
            name=data.get("name"),
        )


@dataclass
class Credentials:
    """User-scoped credentials for the active platform."""

    access_token: str
    token_type: str
    expires_at: datetime
    created_at: datetime
    user: UserInfo
    token_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": CREDENTIALS_VERSION,
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "user": self.user.to_dict(),
        }
        if self.token_id:
            result["token_id"] = self.token_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Credentials:
        expires_at = datetime.fromisoformat(data["expires_at"])
        if expires_at.tzinfo is None:
            raise ValueError(
                "expires_at must be timezone-aware (ISO8601 with timezone offset)"
            )
        created_at = datetime.fromisoformat(data["created_at"])
        return cls(
            access_token=data["access_token"],
            token_type=data["token_type"],
            expires_at=expires_at,
            created_at=created_at,
            user=UserInfo.from_dict(data["user"]),
            token_id=data.get("token_id"),
        )

    @classmethod
    def from_verify_result(cls, token: str, verified: VerifyResult) -> Credentials:
        """Build Credentials from a raw token and its verification result."""
        return cls(
            access_token=token,
            token_type="Bearer",
            expires_at=verified.expires_at,
            created_at=datetime.now(UTC),
            user=verified.user,
            token_id=verified.token_id,
        )

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.now(UTC) >= self.expires_at.astimezone(UTC)


# ---------------------------------------------------------------------------
# Save / Load / Delete
# ---------------------------------------------------------------------------


def save_credentials(credentials: Credentials) -> str:
    """Save user credentials.

    Tries the system keyring first; falls back to plain-text JSON.
    Always cleans up previous keyring entries for the active platform to avoid
    orphaned tokens (e.g. when re-logging, switching accounts, or falling back
    to file storage).

    Returns:
        The storage backend used: ``"keyring"`` or ``"file"``.
    """
    platform_url = get_platform_url()
    old_metadata: dict | None = None
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            old_metadata = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    registry = _registry_from_metadata(old_metadata)
    old_entry = registry["platforms"].get(platform_url)
    _cleanup_platform_keyring_entries(old_entry, platform_url)

    data = credentials.to_dict()
    data["platform_url"] = platform_url

    # Try keyring first
    keyring_account = keyring_account_for_platform(platform_url)
    if _keyring_set(keyring_account, credentials.access_token):
        data.pop("access_token", None)
        data["token_store"] = TOKEN_STORE_KEYRING
        data["keyring_account"] = keyring_account
        registry["platforms"][platform_url] = data
        atomic_write_json(CREDENTIALS_FILE, registry, mode=0o600)
        return TOKEN_STORE_KEYRING

    # Fallback: store everything in the JSON file
    data["token_store"] = TOKEN_STORE_FILE
    data.pop("keyring_account", None)
    registry["platforms"][platform_url] = data
    atomic_write_json(CREDENTIALS_FILE, registry, mode=0o600)
    _warn(
        f"Keyring unavailable — token stored in plain text at {CREDENTIALS_FILE}",
        code="KEYRING_UNAVAILABLE",
    )
    return TOKEN_STORE_FILE


def load_credentials(*, include_env: bool = True) -> Credentials | None:
    """Load credentials with priority: env var → keyring → plain-text file.

    Args:
        include_env: When ``False``, skip ``OSMOSIS_TOKEN`` and load only
            credentials persisted by the CLI.

    Returns:
        The loaded credentials, or ``None`` if no credentials exist.
    """
    # 1. Environment variable
    env_token = os.environ.get("OSMOSIS_TOKEN") if include_env else None
    if env_token:
        return Credentials(
            access_token=env_token,
            token_type="Bearer",
            expires_at=datetime.max.replace(tzinfo=UTC),
            created_at=datetime.now(UTC),
            user=UserInfo(id="", email="", name=None),
            token_id=None,
        )

    platform_url = get_platform_url()

    # 2. Load metadata file
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        _warn(
            f"Could not parse credentials file ({type(exc).__name__}); "
            "run 'osmosis auth login' to re-authenticate.",
            code="CREDENTIALS_PARSE_FAILED",
        )
        return None

    if data.get("version") != CREDENTIALS_VERSION:
        _warn(
            "Credentials format has changed. "
            "Please run 'osmosis auth login' to re-authenticate.",
            code="CREDENTIALS_VERSION_CHANGED",
        )
        return None

    entry = _entry_for_platform(data, platform_url)
    if entry is None:
        return None

    credential_data = dict(entry)
    credential_data.setdefault("version", CREDENTIALS_VERSION)
    token = _resolve_entry_token(credential_data, platform_url)
    if token is None:
        _warn(
            "Token not found for the current Osmosis platform. "
            "Please run 'osmosis auth login' to re-authenticate.",
            code="TOKEN_NOT_FOUND",
        )
        return None
    credential_data["access_token"] = token

    # 4. Parse into Credentials
    try:
        return Credentials.from_dict(credential_data)
    except (KeyError, ValueError) as exc:
        _warn(
            f"Could not parse credentials ({type(exc).__name__}); "
            "run 'osmosis auth login' to re-authenticate.",
            code="CREDENTIALS_PARSE_FAILED",
        )
        return None


def delete_credentials() -> bool:
    """Delete credentials for the current platform.

    Legacy single-platform files are treated as credentials for the default
    production platform.  A corrupt metadata file is still removed because it
    cannot be safely merged with platform-scoped entries.
    """
    platform_url = get_platform_url()
    metadata_missing = False
    metadata_corrupt = False
    old_metadata: dict[str, Any] | None = None
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            old_metadata = json.load(f)
    except FileNotFoundError:
        metadata_missing = True
    except (OSError, json.JSONDecodeError, ValueError):
        metadata_corrupt = True

    old_entry = (
        _entry_for_platform(old_metadata, platform_url)
        if isinstance(old_metadata, dict)
        else None
    )
    keyring_cleaned = _cleanup_platform_keyring_entries(old_entry, platform_url)

    if not keyring_cleaned:
        _warn(
            "Could not remove token from system keyring. "
            "You may want to remove it manually.",
            code="KEYRING_CLEANUP_FAILED",
        )

    if metadata_missing:
        return False

    if metadata_corrupt or not isinstance(old_metadata, dict):
        try:
            CREDENTIALS_FILE.unlink()
            return True
        except FileNotFoundError:
            return False

    if old_metadata.get("version") != CREDENTIALS_VERSION:
        try:
            CREDENTIALS_FILE.unlink()
            return True
        except FileNotFoundError:
            return False

    if _is_platform_registry(old_metadata):
        registry = _registry_from_metadata(old_metadata)
        if platform_url not in registry["platforms"]:
            return False

        del registry["platforms"][platform_url]
        if registry["platforms"]:
            atomic_write_json(CREDENTIALS_FILE, registry, mode=0o600)
            return True

        try:
            CREDENTIALS_FILE.unlink()
            return True
        except FileNotFoundError:
            return False

    if platform_url != _default_platform_url():
        return False

    try:
        CREDENTIALS_FILE.unlink()
        return True
    except FileNotFoundError:
        return False


def get_valid_credentials() -> Credentials | None:
    """Get credentials if they exist and are not expired."""
    credentials = load_credentials()
    if credentials is None:
        return None
    if credentials.is_expired():
        return None
    return credentials


def get_credential_store() -> str | None:
    """Return the active storage backend.

    Returns:
        ``"env"`` if ``OSMOSIS_TOKEN`` is set, ``"keyring"`` or ``"file"``
        based on the metadata file, or ``None`` if not logged in.
    """
    if os.environ.get("OSMOSIS_TOKEN"):
        return TOKEN_STORE_ENV
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        entry = _entry_for_platform(data, get_platform_url())
        if entry is None:
            return None
        return entry.get("token_store", TOKEN_STORE_FILE)
    except Exception:
        return None

"""Credential storage and retrieval for Osmosis CLI authentication.

Supports three token sources with descending priority:

    1. ``OSMOSIS_TOKEN`` environment variable  (CI / headless)
    2. System keyring  (macOS Keychain, GNOME Keyring, …)
    3. Plain-text JSON file  (``~/.config/osmosis/credentials.json``)

When *saving*, the module tries the keyring first.  If successful the
JSON metadata file is written **without** the token.  If the keyring is
unavailable the token is stored in the JSON file and a warning is
emitted so the user knows the secret is on disk in clear text.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import keyring
from keyring.backends.fail import Keyring as FailKeyring
from keyring.errors import PasswordDeleteError

from ._fileutil import atomic_write_json
from .config import CREDENTIALS_FILE, CREDENTIALS_VERSION

if TYPE_CHECKING:
    from .flow import VerifyResult

# ---------------------------------------------------------------------------
# Keyring helpers
# ---------------------------------------------------------------------------

KEYRING_SERVICE = "osmosis-cli"
KEYRING_ACCOUNT = "default"

# Token store backend identifiers (persisted in credentials.json)
TOKEN_STORE_KEYRING = "keyring"
TOKEN_STORE_FILE = "file"
TOKEN_STORE_ENV = "env"


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
        sys.stderr.write(
            f"Warning: failed to remove token from keyring for {account}\n"
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
    """User-scoped credentials (single token for all workspaces)."""

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
    Always cleans up any previous keyring entries to avoid orphaned tokens
    (e.g. when re-logging, switching accounts, or falling back to file storage).

    Returns:
        The storage backend used: ``"keyring"`` or ``"file"``.
    """
    # Always clean up previous keyring entries before saving new credentials.
    # This prevents orphaned tokens when:
    #   - The user re-logs as a different account
    #   - The storage backend switches from keyring to file
    #   - A previous login used an email-based account name (legacy)
    # Read existing metadata once to pass to _cleanup_legacy_keyring_entries,
    # avoiding a redundant file read inside that function.
    old_metadata: dict | None = None
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            old_metadata = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    _keyring_delete(KEYRING_ACCOUNT)
    _cleanup_legacy_keyring_entries(old_metadata)

    data = credentials.to_dict()

    # Try keyring first
    if _keyring_set(KEYRING_ACCOUNT, credentials.access_token):
        data.pop("access_token", None)
        data["token_store"] = TOKEN_STORE_KEYRING
        atomic_write_json(CREDENTIALS_FILE, data, mode=0o600)
        return TOKEN_STORE_KEYRING

    # Fallback: store everything in the JSON file
    data["token_store"] = TOKEN_STORE_FILE
    atomic_write_json(CREDENTIALS_FILE, data, mode=0o600)
    sys.stderr.write(
        "Warning: keyring unavailable — token stored in plain text at "
        f"{CREDENTIALS_FILE}\n"
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

    # 2. Load metadata file
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        sys.stderr.write(
            f"Warning: could not parse credentials file ({type(exc).__name__}); "
            "run 'osmosis auth login' to re-authenticate.\n"
        )
        return None

    if data.get("version") != CREDENTIALS_VERSION:
        sys.stderr.write(
            "Credentials format has changed. "
            "Please run 'osmosis auth login' to re-authenticate.\n"
        )
        return None

    token_store = data.get("token_store", TOKEN_STORE_FILE)

    # 3. Resolve token
    if token_store == TOKEN_STORE_KEYRING:
        # Try the fixed account name first, then fall back to legacy
        # email-based account for backward compatibility.
        token = _keyring_get(KEYRING_ACCOUNT)
        if token is None:
            legacy_account = data.get("user", {}).get("email", "")
            if legacy_account and legacy_account != KEYRING_ACCOUNT:
                token = _keyring_get(legacy_account)
        if token is None:
            sys.stderr.write(
                "Token not found in keyring. "
                "Please run 'osmosis auth login' to re-authenticate.\n"
            )
            return None
        data["access_token"] = token

    # 4. Parse into Credentials
    try:
        return Credentials.from_dict(data)
    except (KeyError, ValueError) as exc:
        sys.stderr.write(
            f"Warning: could not parse credentials ({type(exc).__name__}); "
            "run 'osmosis auth login' to re-authenticate.\n"
        )
        return None


def delete_credentials() -> bool:
    """Delete all stored credentials (keyring entry + file).

    Always attempts to clean up the keyring entry using the fixed account
    name, regardless of metadata state.  Also cleans up any legacy
    email-based keyring entries if the metadata file is readable.
    A corrupt or missing JSON file cannot prevent keyring cleanup.

    Returns:
        ``True`` if credentials were found and removed, ``False`` if no
        credentials existed.  The metadata file is the source of truth:
        ``save_credentials`` always writes it, so its presence indicates
        that credentials were stored.
    """
    # 1. Read metadata once for legacy cleanup (before we delete the file).
    old_metadata: dict | None = None
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            old_metadata = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    # 2. Always attempt keyring cleanup with the fixed account name.
    #    This does not depend on metadata — it works even if the file
    #    is missing or corrupt.
    keyring_cleaned = _keyring_delete(KEYRING_ACCOUNT)

    # 3. Also clean up any legacy email-based keyring entry.
    _cleanup_legacy_keyring_entries(old_metadata)

    if not keyring_cleaned:
        sys.stderr.write(
            "Warning: could not remove token from system keyring. "
            "You may want to remove it manually.\n"
        )

    # 4. Remove the metadata file — this is the canonical indicator of
    #    whether credentials existed, since save_credentials() always
    #    writes this file regardless of keyring availability.
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
        return data.get("token_store", TOKEN_STORE_FILE)
    except Exception:
        return None

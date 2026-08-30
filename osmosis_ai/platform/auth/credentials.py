"""Credential storage and retrieval for Osmosis CLI authentication.

Supports three token sources with descending priority:

    1. ``OSMOSIS_TOKEN`` environment variable  (CI / headless)
    2. System keyring  (macOS Keychain, GNOME Keyring, …)
    3. Legacy platform-scoped plain-text JSON files

New credentials are saved only to the system keyring. Existing file-backed
credentials remain readable so upgrading does not force a new login.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import keyring
from keyring.backends.fail import Keyring as FailKeyring
from keyring.errors import NoKeyringError, PasswordDeleteError

from osmosis_ai.cli.errors import CLIError

from .config import (
    CREDENTIALS_FILE,
    CREDENTIALS_VERSION,
    DEFAULT_PLATFORM_URL,
    get_platform_url,
    normalize_platform_url,
    validate_env_token_platform,
)
from .fileutil import atomic_write_json

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


def _metadata_parse_error() -> CLIError:
    return CLIError(
        "Credential metadata is invalid and was left unchanged. "
        f"Repair or move {CREDENTIALS_FILE}, then try again.",
        code="CREDENTIALS_PARSE_FAILED",
    )


def _backup_invalid_metadata() -> str:
    """Move invalid metadata aside without overwriting an earlier backup."""
    backup = CREDENTIALS_FILE.with_name(f"{CREDENTIALS_FILE.name}.bak")
    suffix = 1
    while backup.exists():
        backup = CREDENTIALS_FILE.with_name(f"{CREDENTIALS_FILE.name}.bak.{suffix}")
        suffix += 1
    try:
        CREDENTIALS_FILE.replace(backup)
    except OSError as exc:
        raise CLIError(
            "Credential metadata could not be backed up for recovery. "
            f"Move {CREDENTIALS_FILE} manually, then try again.",
            code="CREDENTIALS_UNAVAILABLE",
        ) from exc
    _warn(
        f"Invalid credential metadata was moved to {backup}.",
        code="CREDENTIALS_METADATA_BACKED_UP",
    )
    return str(backup)


def _handle_invalid_metadata(
    error: CLIError,
    *,
    recover_invalid: bool,
    cause: BaseException | None = None,
) -> None:
    if not recover_invalid:
        if cause is not None:
            raise error from cause
        raise error
    _backup_invalid_metadata()


def _read_metadata(*, recover_invalid: bool = False) -> dict[str, Any] | None:
    """Read credential metadata without treating corruption as logout."""
    try:
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeError) as exc:
        _handle_invalid_metadata(
            _metadata_parse_error(), recover_invalid=recover_invalid, cause=exc
        )
        return None
    except OSError as exc:
        raise CLIError(
            "Could not read credential metadata. Check permissions for "
            f"{CREDENTIALS_FILE}.",
            code="CREDENTIALS_UNAVAILABLE",
        ) from exc

    if not isinstance(data, dict):
        _handle_invalid_metadata(
            _metadata_parse_error(), recover_invalid=recover_invalid
        )
        return None
    if data.get("version") != CREDENTIALS_VERSION:
        _handle_invalid_metadata(
            CLIError(
                "Credential metadata uses an unsupported format and was left "
                f"unchanged: {CREDENTIALS_FILE}.",
                code="CREDENTIALS_VERSION_CHANGED",
                details={"version": data.get("version")},
            ),
            recover_invalid=recover_invalid,
        )
        return None
    try:
        _validate_metadata(data)
    except CLIError as exc:
        _handle_invalid_metadata(exc, recover_invalid=recover_invalid)
        return None
    return data


def _validate_entry(entry: dict[str, Any]) -> None:
    token_store = entry.get("token_store", TOKEN_STORE_FILE)
    if not isinstance(token_store, str) or token_store not in {
        TOKEN_STORE_FILE,
        TOKEN_STORE_KEYRING,
    }:
        raise _metadata_parse_error()

    platform_url = entry.get("platform_url")
    if platform_url is not None and (
        not isinstance(platform_url, str) or not platform_url.strip()
    ):
        raise _metadata_parse_error()

    keyring_account = entry.get("keyring_account")
    if keyring_account is not None and not isinstance(keyring_account, str):
        raise _metadata_parse_error()


def _validate_metadata(data: dict[str, Any]) -> None:
    """Validate fields that storage logic interprets before any mutation."""
    if "platforms" not in data:
        _validate_entry(data)
        return

    platforms = data["platforms"]
    if not isinstance(platforms, dict):
        raise _metadata_parse_error()

    normalized_platforms: set[str] = set()
    for raw_platform_url, entry in platforms.items():
        if (
            not isinstance(raw_platform_url, str)
            or not raw_platform_url.strip()
            or not isinstance(entry, dict)
        ):
            raise _metadata_parse_error()
        _validate_entry(entry)
        candidate = entry.get("platform_url") or raw_platform_url
        try:
            normalized_key = normalize_platform_url(raw_platform_url)
            normalized_url = normalize_platform_url(candidate)
        except CLIError as exc:
            raise _metadata_parse_error() from exc
        if normalized_url != normalized_key:
            raise _metadata_parse_error()
        if normalized_url in normalized_platforms:
            raise _metadata_parse_error()
        normalized_platforms.add(normalized_url)


def keyring_account_for_platform(platform_url: str | None = None) -> str:
    """Return the stable keyring-account prefix for a platform."""
    normalized_url = normalize_platform_url(platform_url or get_platform_url())
    digest = sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    return f"{KEYRING_ACCOUNT_PREFIX}{digest}"


def _keyring_account_for_credentials(
    credentials: Credentials, platform_url: str
) -> str:
    """Return a platform-scoped account unique to this token identity."""
    platform_account = keyring_account_for_platform(platform_url)
    if credentials.keyring_account and (
        credentials.keyring_account == platform_account
        or credentials.keyring_account.startswith(f"{platform_account}:")
    ):
        return credentials.keyring_account
    suffix = (
        sha256(credentials.token_id.encode("utf-8")).hexdigest()[:24]
        if credentials.token_id
        else uuid4().hex[:24]
    )
    credentials.keyring_account = f"{platform_account}:{suffix}"
    return credentials.keyring_account


def _default_platform_url() -> str:
    return normalize_platform_url(DEFAULT_PLATFORM_URL)


def _is_default_platform_url(platform_url: str) -> bool:
    return normalize_platform_url(platform_url) == _default_platform_url()


def _is_platform_registry(data: dict[str, Any]) -> bool:
    return "platforms" in data


def _platform_entry_key(metadata: dict[str, Any], platform_url: str) -> str | None:
    if not _is_platform_registry(metadata):
        return None

    normalized_url = normalize_platform_url(platform_url)
    for raw_platform_url, entry in metadata["platforms"].items():
        candidate = entry.get("platform_url") or raw_platform_url
        if normalize_platform_url(candidate) == normalized_url:
            return raw_platform_url
    return None


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
        registry = dict(metadata)
        registry["platforms"] = dict(metadata["platforms"])
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
        key = _platform_entry_key(metadata, normalized_url)
        entry = metadata["platforms"].get(key) if key is not None else None
        return entry if isinstance(entry, dict) else None

    if normalized_url == _default_platform_url():
        return metadata
    return None


def _entry_used_keyring(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    return entry.get("token_store", TOKEN_STORE_FILE) == TOKEN_STORE_KEYRING


def _is_platform_keyring_account(account: str, platform_url: str) -> bool:
    prefix = keyring_account_for_platform(platform_url)
    return account == prefix or account.startswith(f"{prefix}:")


def _keyring_accounts_for_entry(
    entry: dict[str, Any] | None,
    platform_url: str,
) -> list[str]:
    platform_url = normalize_platform_url(platform_url)
    is_default_platform = _is_default_platform_url(platform_url)
    accounts: list[str] = []
    if entry is not None and _entry_used_keyring(entry):
        account = entry.get("keyring_account")
        if isinstance(account, str) and (
            _is_platform_keyring_account(account, platform_url)
            or (is_default_platform and account == KEYRING_ACCOUNT)
        ):
            accounts.append(account)

    if entry is not None and is_default_platform:
        user = entry.get("user")
        old_account = user.get("email", "") if isinstance(user, dict) else ""
        if (
            isinstance(old_account, str)
            and old_account
            and old_account != KEYRING_ACCOUNT
        ):
            accounts.append(old_account)

    accounts.append(keyring_account_for_platform(platform_url))
    if is_default_platform:
        accounts.append(KEYRING_ACCOUNT)

    return list(dict.fromkeys(account for account in accounts if account))


def _cleanup_platform_keyring_entries(
    entry: dict[str, Any] | None,
    platform_url: str,
    *,
    tolerate_unavailable: bool = False,
) -> bool:
    cleaned = True
    for account in _keyring_accounts_for_entry(entry, platform_url):
        try:
            cleaned = _keyring_delete(account) and cleaned
        except CLIError:
            if not tolerate_unavailable:
                raise
            cleaned = False
    return cleaned


def _cleanup_replaced_keyring_entries(
    entry: dict[str, Any] | None,
    platform_url: str,
    current_account: str,
) -> bool:
    """Best-effort cleanup of legacy accounts after the new token is saved."""
    cleaned = True
    for account in _keyring_accounts_for_entry(entry, platform_url):
        if account != current_account:
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
    except NoKeyringError:
        return False
    except Exception as exc:
        raise CLIError(
            "Could not save credentials in the system keyring. "
            "Unlock or repair the keyring, or use OSMOSIS_TOKEN for "
            "non-interactive authentication.",
            code="KEYRING_UNAVAILABLE",
        ) from exc


def ensure_keyring_available() -> None:
    """Fail before device login when no system keyring backend is available."""
    try:
        if isinstance(keyring.get_keyring(), FailKeyring):
            raise CLIError(
                "No system keyring is available. Set OSMOSIS_TOKEN for CI/CD or "
                "configure a keyring backend for interactive login.",
                code="KEYRING_UNAVAILABLE",
            )
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            "Could not access the system keyring. Unlock or repair the keyring, "
            "or use OSMOSIS_TOKEN for non-interactive authentication.",
            code="KEYRING_UNAVAILABLE",
        ) from exc


def _keyring_get(account: str) -> str | None:
    """Retrieve a token from the system keyring."""
    try:
        if isinstance(keyring.get_keyring(), FailKeyring):
            raise CLIError(
                "No system keyring is available. Use OSMOSIS_TOKEN for CI/CD or "
                "configure a keyring backend for interactive login.",
                code="KEYRING_UNAVAILABLE",
            )
        return keyring.get_password(KEYRING_SERVICE, account)
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            "Could not read credentials from the system keyring. "
            "Unlock or repair the keyring and try again.",
            code="KEYRING_UNAVAILABLE",
        ) from exc


def _keyring_delete(account: str) -> bool:
    """Delete a keyring token. A missing entry is already clean."""
    try:
        if isinstance(keyring.get_keyring(), FailKeyring):
            raise CLIError(
                "No system keyring is available, so saved credentials could not "
                "be removed. Configure or unlock the keyring and try again.",
                code="KEYRING_UNAVAILABLE",
            )
        keyring.delete_password(KEYRING_SERVICE, account)
        return True
    except CLIError:
        raise
    except PasswordDeleteError:
        return True
    except Exception as exc:
        raise CLIError(
            "Could not remove saved credentials from the system keyring. "
            "Unlock or repair the keyring and try again.",
            code="KEYRING_UNAVAILABLE",
        ) from exc


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
    keyring_account: str | None = field(default=None, repr=False, compare=False)

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
            keyring_account=data.get("keyring_account"),
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


def _cleanup_replaced_credentials(
    old_credentials: Credentials,
    current_credentials: Credentials,
) -> bool:
    """Remove the previous token's local keyring entries after revocation."""
    platform_url = get_platform_url()
    old_entry = old_credentials.to_dict()
    old_entry.pop("access_token", None)
    old_entry["token_store"] = TOKEN_STORE_KEYRING
    old_entry["keyring_account"] = _keyring_account_for_credentials(
        old_credentials, platform_url
    )
    current_account = _keyring_account_for_credentials(
        current_credentials, platform_url
    )

    try:
        cleaned = _cleanup_replaced_keyring_entries(
            old_entry, platform_url, current_account
        )
    except CLIError:
        cleaned = False
    if not cleaned:
        _warn(
            "The new login is active, but an older local keyring entry could "
            "not be removed.",
            code="KEYRING_CLEANUP_FAILED",
        )
    return cleaned


# ---------------------------------------------------------------------------
# Save / Load / Delete
# ---------------------------------------------------------------------------


def save_credentials(
    credentials: Credentials,
    *,
    cleanup_replaced: bool = True,
    recover_invalid_metadata: bool = False,
) -> str:
    """Save user credentials.

    The token is written to the platform-scoped system-keyring account before
    metadata or legacy entries are changed, so a failed save leaves the
    previous login usable.

    Returns:
        The storage backend used: ``"keyring"``.
    """
    platform_url = get_platform_url()
    old_metadata = _read_metadata(recover_invalid=recover_invalid_metadata)
    registry = _registry_from_metadata(old_metadata)
    old_key = _platform_entry_key(registry, platform_url)
    old_entry = registry["platforms"].get(old_key) if old_key is not None else None

    data = credentials.to_dict()
    data["platform_url"] = platform_url

    keyring_account = _keyring_account_for_credentials(credentials, platform_url)
    old_keyring_accounts = set(_keyring_accounts_for_entry(old_entry, platform_url))
    if not _keyring_set(keyring_account, credentials.access_token):
        raise CLIError(
            "No system keyring is available. Set OSMOSIS_TOKEN for CI/CD or "
            "other non-interactive environments.",
            code="KEYRING_UNAVAILABLE",
        )

    data.pop("access_token", None)
    data["token_store"] = TOKEN_STORE_KEYRING
    data["keyring_account"] = keyring_account
    if old_key is not None and old_key != platform_url:
        del registry["platforms"][old_key]
    registry["platforms"][platform_url] = data
    try:
        atomic_write_json(CREDENTIALS_FILE, registry)
    except Exception:
        if keyring_account not in old_keyring_accounts:
            try:
                cleaned = _keyring_delete(keyring_account)
            except CLIError:
                cleaned = False
            if not cleaned:
                _warn(
                    "Credential metadata could not be saved, and the new "
                    "keyring entry could not be rolled back.",
                    code="KEYRING_CLEANUP_FAILED",
                )
        raise
    if cleanup_replaced:
        try:
            cleaned = _cleanup_replaced_keyring_entries(
                old_entry,
                platform_url,
                keyring_account,
            )
        except CLIError:
            cleaned = False
        if not cleaned:
            _warn(
                "Credentials were saved, but an older local keyring entry "
                "could not be removed.",
                code="KEYRING_CLEANUP_FAILED",
            )
    return TOKEN_STORE_KEYRING


def load_credentials(*, include_env: bool = True) -> Credentials | None:
    """Load credentials with priority: env var → keyring → legacy file.

    Args:
        include_env: When ``False``, skip ``OSMOSIS_TOKEN`` and load only
            credentials persisted by the CLI.

    Returns:
        The loaded credentials, or ``None`` if no credentials exist.
    """
    # 1. Environment variable
    env_token = os.environ.get("OSMOSIS_TOKEN") if include_env else None
    if env_token:
        validate_env_token_platform(env_token)
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
    data = _read_metadata()
    if data is None:
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
        raise CLIError(
            "Credential metadata for the current platform is invalid and was "
            "left unchanged. Repair or remove the affected entry, then try again.",
            code="CREDENTIALS_PARSE_FAILED",
        ) from exc


def delete_credentials(
    *,
    recover_invalid_metadata: bool = False,
    tolerate_keyring_unavailable: bool = False,
) -> bool:
    """Delete credentials for the current platform.

    Legacy single-platform files are treated as credentials for the default
    production platform. Corrupt or unknown metadata is preserved so logout
    cannot destroy entries belonging to another platform.
    """
    platform_url = get_platform_url()
    old_metadata = _read_metadata(recover_invalid=recover_invalid_metadata)
    old_entry = (
        _entry_for_platform(old_metadata, platform_url)
        if isinstance(old_metadata, dict)
        else None
    )
    file_backed_entry = old_entry is not None and not _entry_used_keyring(old_entry)
    tolerate_cleanup_failure = tolerate_keyring_unavailable or file_backed_entry
    keyring_cleaned = _cleanup_platform_keyring_entries(
        old_entry,
        platform_url,
        tolerate_unavailable=tolerate_cleanup_failure,
    )
    if not keyring_cleaned:
        if not tolerate_cleanup_failure:
            raise CLIError(
                "Could not remove token from the system keyring. Credential "
                "metadata was left unchanged so logout can be retried.",
                code="KEYRING_CLEANUP_FAILED",
            )
        if not file_backed_entry:
            _warn(
                "Saved credential metadata was removed, but the system keyring "
                "entry could not be verified or deleted on this host.",
                code="KEYRING_CLEANUP_FAILED",
            )

    if old_metadata is None:
        return False

    if _is_platform_registry(old_metadata):
        registry = _registry_from_metadata(old_metadata)
        platform_key = _platform_entry_key(registry, platform_url)
        if platform_key is None:
            return False

        del registry["platforms"][platform_key]
        if registry["platforms"]:
            atomic_write_json(CREDENTIALS_FILE, registry)
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


def get_credential_store(*, include_env: bool = True) -> str | None:
    """Return the active storage backend.

    Args:
        include_env: When ``False``, report only credentials persisted by the
            CLI for the active platform.

    Returns:
        ``"env"`` if ``OSMOSIS_TOKEN`` is set, ``"keyring"`` or ``"file"``
        based on the metadata file, or ``None`` if not logged in.
    """
    env_token = os.environ.get("OSMOSIS_TOKEN") if include_env else None
    if env_token:
        validate_env_token_platform(env_token)
        return TOKEN_STORE_ENV

    data = _read_metadata()
    if data is None:
        return None
    entry = _entry_for_platform(data, get_platform_url())
    if entry is None:
        return None
    return entry.get("token_store", TOKEN_STORE_FILE)

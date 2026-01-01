"""Credential storage and retrieval for Osmosis CLI authentication."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG_DIR, CREDENTIALS_FILE, CREDENTIALS_VERSION


@dataclass
class UserInfo:
    """User information from authentication."""

    id: str
    email: str
    name: Optional[str] = None

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
class OrganizationInfo:
    """Organization (workspace) information from authentication."""

    id: str
    name: str
    role: str  # "owner", "admin", or "member"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrganizationInfo:
        return cls(
            id=data["id"],
            name=data["name"],
            role=data.get("role", "member"),
        )


@dataclass
class WorkspaceCredentials:
    """Credentials for a single workspace."""

    access_token: str
    token_type: str
    expires_at: datetime
    user: UserInfo
    organization: OrganizationInfo
    created_at: datetime
    token_id: Optional[str] = None  # Platform token ID for revocation

    def to_dict(self) -> dict[str, Any]:
        result = {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "user": self.user.to_dict(),
            "organization": self.organization.to_dict(),
            "created_at": self.created_at.isoformat(),
        }
        if self.token_id:
            result["token_id"] = self.token_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceCredentials:
        expires_at = datetime.fromisoformat(data["expires_at"])
        if expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware (ISO8601 with timezone offset)")
        return cls(
            access_token=data["access_token"],
            token_type=data["token_type"],
            expires_at=expires_at,
            user=UserInfo.from_dict(data["user"]),
            organization=OrganizationInfo.from_dict(data["organization"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            token_id=data.get("token_id"),
        )

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.expires_at.tzinfo is None:
            raise ValueError(
                "expires_at must be timezone-aware (ISO8601 with timezone offset)"
            )
        return datetime.now(timezone.utc) >= self.expires_at.astimezone(timezone.utc)


@dataclass
class CredentialsStore:
    """Multi-workspace credentials storage."""

    active_workspace: Optional[str]  # workspace name
    workspaces: dict[str, WorkspaceCredentials]  # keyed by workspace name

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CREDENTIALS_VERSION,
            "active_workspace": self.active_workspace,
            "workspaces": {
                name: creds.to_dict() for name, creds in self.workspaces.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CredentialsStore:
        workspaces = {}
        for name, creds_data in data.get("workspaces", {}).items():
            workspaces[name] = WorkspaceCredentials.from_dict(creds_data)
        return cls(
            active_workspace=data.get("active_workspace"),
            workspaces=workspaces,
        )

    def get_active_credentials(self) -> Optional[WorkspaceCredentials]:
        """Get credentials for the active workspace."""
        if not self.active_workspace:
            return None
        return self.workspaces.get(self.active_workspace)

    def get_workspace_names(self) -> list[str]:
        """Get list of all workspace names."""
        return list(self.workspaces.keys())


def _ensure_config_dir() -> None:
    """Ensure the config directory exists with proper permissions."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, mode=0o700)
    else:
        # Ensure permissions are correct
        current_mode = CONFIG_DIR.stat().st_mode
        if current_mode & 0o077:  # If group or others have any permissions
            os.chmod(CONFIG_DIR, 0o700)


def _set_file_permissions(path: Path) -> None:
    """Set restrictive permissions on credentials file."""
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600


def _load_store() -> Optional[CredentialsStore]:
    """Load the credentials store from file."""
    if not CREDENTIALS_FILE.exists():
        return None

    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CredentialsStore.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_store(store: CredentialsStore) -> None:
    """Save the credentials store to file."""
    _ensure_config_dir()

    data = store.to_dict()

    # Write to a temporary file first, then rename for atomicity
    temp_file = CREDENTIALS_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _set_file_permissions(temp_file)
        temp_file.rename(CREDENTIALS_FILE)
    except Exception:
        # Clean up temp file if something went wrong
        if temp_file.exists():
            temp_file.unlink()
        raise


def save_credentials(credentials: WorkspaceCredentials) -> None:
    """Save credentials for a workspace, adding it to the store.

    This will also set the workspace as active.

    Args:
        credentials: The credentials to save.
    """
    store = _load_store() or CredentialsStore(active_workspace=None, workspaces={})

    workspace_name = credentials.organization.name
    store.workspaces[workspace_name] = credentials
    store.active_workspace = workspace_name

    _save_store(store)


def load_credentials() -> Optional[WorkspaceCredentials]:
    """Load credentials for the active workspace.

    Returns:
        The loaded credentials, or None if no active workspace exists.
    """
    store = _load_store()
    if store is None:
        return None
    return store.get_active_credentials()


def delete_credentials() -> bool:
    """Delete all stored credentials.

    Returns:
        True if credentials were deleted, False if none existed.
    """
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
        return True
    return False


def delete_workspace_credentials(workspace_name: str) -> bool:
    """Delete credentials for a specific workspace.

    Args:
        workspace_name: The name of the workspace to remove.

    Returns:
        True if the workspace was removed, False if it didn't exist.
    """
    store = _load_store()
    if store is None or workspace_name not in store.workspaces:
        return False

    del store.workspaces[workspace_name]

    # If we deleted the active workspace, set a new one or None
    if store.active_workspace == workspace_name:
        if store.workspaces:
            store.active_workspace = next(iter(store.workspaces.keys()))
        else:
            store.active_workspace = None

    _save_store(store)
    return True


def get_valid_credentials() -> Optional[WorkspaceCredentials]:
    """Get credentials for the active workspace if they exist and are not expired.

    Returns:
        Valid credentials, or None if no valid credentials exist.
    """
    credentials = load_credentials()
    if credentials is None:
        return None
    if credentials.is_expired():
        return None
    return credentials


def get_all_workspaces() -> list[tuple[str, WorkspaceCredentials, bool]]:
    """Get all stored workspaces with their credentials.

    Returns:
        List of (workspace_name, credentials, is_active) tuples.
    """
    store = _load_store()
    if store is None:
        return []

    result = []
    for name, creds in store.workspaces.items():
        is_active = name == store.active_workspace
        result.append((name, creds, is_active))
    return result


def get_active_workspace() -> Optional[str]:
    """Get the name of the active workspace.

    Returns:
        The active workspace name, or None if no workspace is active.
    """
    store = _load_store()
    if store is None:
        return None
    return store.active_workspace


def set_active_workspace(workspace_name: str) -> bool:
    """Set the active workspace.

    Args:
        workspace_name: The name of the workspace to make active.

    Returns:
        True if successful, False if the workspace doesn't exist.
    """
    store = _load_store()
    if store is None or workspace_name not in store.workspaces:
        return False

    store.active_workspace = workspace_name
    _save_store(store)
    return True

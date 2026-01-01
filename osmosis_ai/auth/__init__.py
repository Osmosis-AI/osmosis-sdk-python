"""Osmosis CLI authentication module."""

from .config import CONFIG_DIR, CREDENTIALS_FILE, PLATFORM_URL
from .credentials import (
    CredentialsStore,
    OrganizationInfo,
    UserInfo,
    WorkspaceCredentials,
    delete_credentials,
    delete_workspace_credentials,
    get_active_workspace,
    get_all_workspaces,
    get_valid_credentials,
    load_credentials,
    save_credentials,
    set_active_workspace,
)
from .flow import LoginError, LoginResult, login
from .platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    platform_request,
)

__all__ = [
    # Config
    "CONFIG_DIR",
    "CREDENTIALS_FILE",
    "PLATFORM_URL",
    # Credentials
    "CredentialsStore",
    "OrganizationInfo",
    "UserInfo",
    "WorkspaceCredentials",
    "delete_credentials",
    "delete_workspace_credentials",
    "get_active_workspace",
    "get_all_workspaces",
    "get_valid_credentials",
    "load_credentials",
    "save_credentials",
    "set_active_workspace",
    # Flow
    "LoginError",
    "LoginResult",
    "login",
    # Platform Client
    "AuthenticationExpiredError",
    "PlatformAPIError",
    "platform_request",
]

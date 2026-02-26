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
from .local_config import load_workspace_projects, save_workspace_projects
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
    # Platform Client
    "AuthenticationExpiredError",
    # Credentials
    "CredentialsStore",
    # Flow
    "LoginError",
    "LoginResult",
    "OrganizationInfo",
    "PlatformAPIError",
    "UserInfo",
    "WorkspaceCredentials",
    "delete_credentials",
    "delete_workspace_credentials",
    "get_active_workspace",
    "get_all_workspaces",
    "get_valid_credentials",
    "load_credentials",
    "load_workspace_projects",
    "login",
    "platform_request",
    "save_credentials",
    "save_workspace_projects",
    "set_active_workspace",
]

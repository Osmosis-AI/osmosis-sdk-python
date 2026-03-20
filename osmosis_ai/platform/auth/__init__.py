"""Osmosis CLI authentication module."""

from .config import CONFIG_DIR, CREDENTIALS_FILE, PLATFORM_URL
from .credentials import (
    Credentials,
    UserInfo,
    delete_credentials,
    get_credential_store,
    get_valid_credentials,
    load_credentials,
    save_credentials,
)
from .flow import LoginError, LoginResult, device_login, verify_token
from .local_config import (
    get_active_workspace,
    get_active_workspace_id,
    load_workspace_projects,
    reset_session,
    save_workspace_projects,
    set_active_workspace,
)
from .platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    SubscriptionRequiredError,
    platform_request,
)

__all__ = [
    "CONFIG_DIR",
    "CREDENTIALS_FILE",
    "PLATFORM_URL",
    "AuthenticationExpiredError",
    "Credentials",
    "LoginError",
    "LoginResult",
    "PlatformAPIError",
    "SubscriptionRequiredError",
    "UserInfo",
    "delete_credentials",
    "device_login",
    "get_active_workspace",
    "get_active_workspace_id",
    "get_credential_store",
    "get_valid_credentials",
    "load_credentials",
    "load_workspace_projects",
    "platform_request",
    "reset_session",
    "save_credentials",
    "save_workspace_projects",
    "set_active_workspace",
    "verify_token",
]

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
    reset_session,
    set_active_workspace,
)
from .platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    SubscriptionRequiredError,
    ensure_active_workspace,
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
    "ensure_active_workspace",
    "get_active_workspace",
    "get_active_workspace_id",
    "get_credential_store",
    "get_valid_credentials",
    "load_credentials",
    "platform_request",
    "reset_session",
    "save_credentials",
    "set_active_workspace",
    "verify_token",
]

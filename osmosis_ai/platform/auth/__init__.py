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
from .local_config import reset_session
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
    "get_credential_store",
    "get_valid_credentials",
    "load_credentials",
    "platform_request",
    "reset_session",
    "save_credentials",
    "verify_token",
]

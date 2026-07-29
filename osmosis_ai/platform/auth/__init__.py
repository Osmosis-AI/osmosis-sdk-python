"""Osmosis CLI authentication module."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import public_module_dir, resolve_lazy_export

if TYPE_CHECKING:
    from .config import CONFIG_DIR, CREDENTIALS_FILE, PLATFORM_URL, get_platform_url
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
        UpgradeRequiredError,
        platform_request,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "CONFIG_DIR": ("osmosis_ai.platform.auth.config", "CONFIG_DIR"),
    "CREDENTIALS_FILE": ("osmosis_ai.platform.auth.config", "CREDENTIALS_FILE"),
    "PLATFORM_URL": ("osmosis_ai.platform.auth.config", "PLATFORM_URL"),
    "AuthenticationExpiredError": (
        "osmosis_ai.platform.auth.platform_client",
        "AuthenticationExpiredError",
    ),
    "Credentials": ("osmosis_ai.platform.auth.credentials", "Credentials"),
    "LoginError": ("osmosis_ai.platform.auth.flow", "LoginError"),
    "LoginResult": ("osmosis_ai.platform.auth.flow", "LoginResult"),
    "PlatformAPIError": (
        "osmosis_ai.platform.auth.platform_client",
        "PlatformAPIError",
    ),
    "SubscriptionRequiredError": (
        "osmosis_ai.platform.auth.platform_client",
        "SubscriptionRequiredError",
    ),
    "UpgradeRequiredError": (
        "osmosis_ai.platform.auth.platform_client",
        "UpgradeRequiredError",
    ),
    "UserInfo": ("osmosis_ai.platform.auth.credentials", "UserInfo"),
    "delete_credentials": (
        "osmosis_ai.platform.auth.credentials",
        "delete_credentials",
    ),
    "device_login": ("osmosis_ai.platform.auth.flow", "device_login"),
    "get_credential_store": (
        "osmosis_ai.platform.auth.credentials",
        "get_credential_store",
    ),
    "get_platform_url": ("osmosis_ai.platform.auth.config", "get_platform_url"),
    "get_valid_credentials": (
        "osmosis_ai.platform.auth.credentials",
        "get_valid_credentials",
    ),
    "load_credentials": (
        "osmosis_ai.platform.auth.credentials",
        "load_credentials",
    ),
    "platform_request": (
        "osmosis_ai.platform.auth.platform_client",
        "platform_request",
    ),
    "reset_session": ("osmosis_ai.platform.auth.local_config", "reset_session"),
    "save_credentials": (
        "osmosis_ai.platform.auth.credentials",
        "save_credentials",
    ),
    "verify_token": ("osmosis_ai.platform.auth.flow", "verify_token"),
}


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        namespace=globals(),
        exports=_EXPORTS,
    )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


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
    "UpgradeRequiredError",
    "UserInfo",
    "delete_credentials",
    "device_login",
    "get_credential_store",
    "get_platform_url",
    "get_valid_credentials",
    "load_credentials",
    "platform_request",
    "reset_session",
    "save_credentials",
    "verify_token",
]

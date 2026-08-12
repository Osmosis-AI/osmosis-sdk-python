from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class CLIErrorCode(StrEnum):
    """Public CLI error codes in JSON error envelopes and ``CLIError.code``."""

    VALIDATION = "VALIDATION"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UPGRADE_REQUIRED = "UPGRADE_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    PLATFORM_ERROR = "PLATFORM_ERROR"
    INTERNAL = "INTERNAL"
    INTERACTIVE_REQUIRED = "INTERACTIVE_REQUIRED"
    WORKSPACE_REQUIRED = "WORKSPACE_REQUIRED"
    NETWORK = "NETWORK"
    SUBSCRIPTION_REQUIRED = "SUBSCRIPTION_REQUIRED"
    BILLING_REQUIRED = "BILLING_REQUIRED"
    KEYRING_UNAVAILABLE = "KEYRING_UNAVAILABLE"
    CREDENTIALS_PARSE_FAILED = "CREDENTIALS_PARSE_FAILED"
    CREDENTIALS_VERSION_CHANGED = "CREDENTIALS_VERSION_CHANGED"
    TOKEN_NOT_FOUND = "TOKEN_NOT_FOUND"
    KEYRING_CLEANUP_FAILED = "KEYRING_CLEANUP_FAILED"
    TOKEN_REVOKE_FAILED = "TOKEN_REVOKE_FAILED"


class CLIError(Exception):
    """Raised when the CLI encounters a recoverable error."""

    def __init__(
        self,
        message: str = "",
        *,
        code: CLIErrorCode | str = CLIErrorCode.VALIDATION,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = CLIErrorCode(code)
        self.details: dict[str, Any] = dict(details) if details else {}

"""Redact secret values echoed in platform API errors.

Shared by ``secret set`` and submit flows so a platform error cannot bounce a
provided secret back onto stderr or the JSON error envelope.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    if isinstance(data, str):
        return data.replace(secret_value, "[REDACTED]")
    if isinstance(data, dict):
        return {
            key: (
                "[REDACTED]"
                if str(key).lower() in {"value", "secret", "secret_value"}
                else redact_secret_value(value, secret_value)
            )
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [redact_secret_value(value, secret_value) for value in data]
    return data


def redact_secret_text(value: str | None, secret_value: str) -> str | None:
    if value is None:
        return None
    return value.replace(secret_value, "[REDACTED]")


def redact_secret_platform_error(
    exc: PlatformAPIError, secret_value: str
) -> PlatformAPIError:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    return PlatformAPIError(
        redact_secret_text(str(exc), secret_value) or "",
        status_code=exc.status_code,
        error_code=redact_secret_text(exc.error_code, secret_value),
        field=redact_secret_text(exc.field, secret_value),
        details=redact_secret_value(exc.details, secret_value),
    )


def redact_provided_secrets(
    exc: PlatformAPIError, secret_values: Iterable[str]
) -> PlatformAPIError:
    """Redact every provided secret value, longest first to avoid partial overlaps."""
    redacted = exc
    for value in sorted({v for v in secret_values if v}, key=len, reverse=True):
        redacted = redact_secret_platform_error(redacted, value)
    return redacted

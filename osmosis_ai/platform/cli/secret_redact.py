"""Redact secret values echoed in platform API errors.

Shared by ``secret set`` and submit flows so a platform error cannot bounce a
provided secret back onto stderr or the JSON error envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    if not secret_value:
        return data
    if isinstance(data, str):
        return data.replace(secret_value, "[REDACTED]")
    if isinstance(data, dict):
        redacted: dict[Any, Any] = {}
        for key, value in data.items():
            out_key: Any = (
                key.replace(secret_value, "[REDACTED]") if isinstance(key, str) else key
            )
            if str(key).lower() in {"value", "secret", "secret_value"}:
                redacted[out_key] = "[REDACTED]"
            else:
                redacted[out_key] = redact_secret_value(value, secret_value)
        return redacted
    if isinstance(data, list):
        return [redact_secret_value(value, secret_value) for value in data]
    return data


def redact_secret_text(value: str | None, secret_value: str) -> str | None:
    if value is None:
        return None
    if not secret_value:
        return value
    return value.replace(secret_value, "[REDACTED]")


def redact_secret_platform_error(
    exc: PlatformAPIError, secret_value: str
) -> PlatformAPIError:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    redacted = copy.copy(exc)
    redacted.args = (redact_secret_text(str(exc), secret_value) or "",)
    redacted.error_code = redact_secret_text(exc.error_code, secret_value)
    redacted.field = redact_secret_text(exc.field, secret_value)
    redacted.details = redact_secret_value(exc.details, secret_value)
    return redacted


def redact_provided_secrets(
    exc: PlatformAPIError, secret_values: Iterable[str]
) -> PlatformAPIError:
    """Redact every provided secret value, longest first to avoid partial overlaps."""
    redacted = exc
    for value in sorted({v for v in secret_values if v}, key=len, reverse=True):
        redacted = redact_secret_platform_error(redacted, value)
    return redacted

"""Redact secret values echoed in platform API errors.

Shared by ``secret set`` and submit flows so a platform error cannot bounce a
provided secret back onto stderr or the JSON error envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def _redact_str(text: str, secret_value: str) -> str:
    redacted = text.replace(secret_value, "[REDACTED]")
    if secret_value not in redacted:
        return redacted
    # The marker can join its neighbours into the secret again ("]]]" with the
    # secret "]]" becomes "[REDACTED]]"), or contain it outright (the secret
    # "RE"); then no part of the text is safe, and in the second case neither
    # is the marker.
    return "" if secret_value in "[REDACTED]" else "[REDACTED]"


def redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    if not secret_value:
        return data
    if isinstance(data, str):
        return _redact_str(data, secret_value)
    if isinstance(data, dict):
        redacted: dict[Any, Any] = {}
        for key, value in data.items():
            out_key: Any = (
                _redact_str(key, secret_value) if isinstance(key, str) else key
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
    return _redact_str(value, secret_value)


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

"""Redact secret values echoed in platform API errors.

Shared by ``secret set`` and submit flows so a platform error cannot bounce a
provided secret back onto stderr or the JSON error envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def _secrets(values: Iterable[str]) -> tuple[str, ...]:
    """Non-empty secrets, longest first to avoid partial overlaps."""
    return tuple(sorted({v for v in values if v}, key=len, reverse=True))


def _redact_str(text: str, secrets: tuple[str, ...]) -> str:
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    if not any(secret in redacted for secret in secrets):
        return redacted
    # A marker can join its neighbours into a secret again ("]]]" with the
    # secret "]]" becomes "[REDACTED]]", and a later secret's marker can rebuild
    # an earlier one), or contain one outright (the secret "RE"); then no part
    # of the text is safe, and in the second case neither is the marker.
    if any(secret in "[REDACTED]" for secret in secrets):
        return ""
    return "[REDACTED]"


def _redact_text(value: str | None, secrets: tuple[str, ...]) -> str | None:
    return None if value is None else _redact_str(value, secrets)


def _redact_value(data: Any, secrets: tuple[str, ...]) -> Any:
    if not secrets:
        return data
    if isinstance(data, str):
        return _redact_str(data, secrets)
    if isinstance(data, dict):
        redacted: dict[Any, Any] = {}
        for key, value in data.items():
            out_key: Any = _redact_str(key, secrets) if isinstance(key, str) else key
            if str(key).lower() in {"value", "secret", "secret_value"}:
                redacted[out_key] = _redact_str("[REDACTED]", secrets)
            else:
                redacted[out_key] = _redact_value(value, secrets)
        return redacted
    if isinstance(data, list):
        return [_redact_value(value, secrets) for value in data]
    return data


def _redact_error(exc: PlatformAPIError, secrets: tuple[str, ...]) -> PlatformAPIError:
    redacted = copy.copy(exc)
    redacted.args = (_redact_str(str(exc), secrets),)
    redacted.error_code = _redact_text(exc.error_code, secrets)
    redacted.field = _redact_text(exc.field, secrets)
    redacted.details = _redact_value(exc.details, secrets)
    return redacted


def redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    return _redact_value(data, _secrets([secret_value]))


def redact_secret_text(value: str | None, secret_value: str) -> str | None:
    return _redact_text(value, _secrets([secret_value]))


def redact_secret_platform_error(
    exc: PlatformAPIError, secret_value: str
) -> PlatformAPIError:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    return _redact_error(exc, _secrets([secret_value]))


def redact_provided_secrets(
    exc: PlatformAPIError, secret_values: Iterable[str]
) -> PlatformAPIError:
    """Redact every provided secret value, checking each string against all of them."""
    return _redact_error(exc, _secrets(secret_values))

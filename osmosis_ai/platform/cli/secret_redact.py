"""Redact secret values echoed in platform API errors.

Shared by ``secret set`` and submit flows so a platform error cannot bounce a
provided secret back onto stderr or the JSON error envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from osmosis_ai._redaction import SECRET_FIELDS, SecretRedactor
from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def _redact_error(exc: PlatformAPIError, redactor: SecretRedactor) -> PlatformAPIError:
    redacted = copy.copy(exc)
    redacted.args = (redactor.scrub(str(exc)),)
    redacted.error_code = (
        None if exc.error_code is None else redactor.scrub(exc.error_code)
    )
    redacted.field = None if exc.field is None else redactor.scrub(exc.field)
    redacted.details = (
        None
        if exc.details is None
        else redactor.scrub_value(exc.details, sensitive_keys=SECRET_FIELDS)
    )
    return redacted


def redact_secret_value(data: Any, secret_value: str) -> Any:
    """Return a copy of *data* with the current plaintext secret removed."""
    return SecretRedactor([secret_value]).scrub_value(
        data, sensitive_keys=SECRET_FIELDS
    )


def redact_secret_text(value: str | None, secret_value: str) -> str | None:
    return None if value is None else SecretRedactor([secret_value]).scrub(value)


def redact_secret_platform_error(
    exc: PlatformAPIError, secret_value: str
) -> PlatformAPIError:
    """Clone a PlatformAPIError with any echoed secret value removed."""
    return _redact_error(exc, SecretRedactor([secret_value]))


def redact_provided_secrets(
    exc: PlatformAPIError, secret_values: Iterable[str]
) -> PlatformAPIError:
    """Redact every provided secret value, checking each string against all of them."""
    return _redact_error(exc, SecretRedactor(secret_values))

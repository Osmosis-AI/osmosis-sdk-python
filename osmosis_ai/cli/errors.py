from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CLIError(Exception):
    """Raised when the CLI encounters a recoverable error."""

    def __init__(
        self,
        message: str = "",
        *,
        code: str = "VALIDATION",
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}
        self.request_id = request_id

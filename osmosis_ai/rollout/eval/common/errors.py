"""Shared exceptions for local rollout workflows."""

from __future__ import annotations


class LocalExecutionError(Exception):
    """Base exception for local execution errors."""

    pass


class DatasetValidationError(LocalExecutionError):
    """Raised when dataset rows fail schema/value validation."""

    pass


class DatasetParseError(LocalExecutionError):
    """Raised when dataset files cannot be parsed."""

    pass


class ToolValidationError(LocalExecutionError):
    """Raised when tool schemas are invalid for provider APIs."""

    pass


class ProviderError(LocalExecutionError):
    """Raised when external provider calls fail."""

    pass


class SystemicProviderError(ProviderError):
    """Provider error that affects ALL rows (auth, budget, connectivity).

    Unlike transient errors (rate limits, timeouts, context window), these
    errors indicate a systemic configuration problem that will cause every
    single row to fail identically.  Callers should abort the batch early
    rather than retrying each row.
    """

    pass

__all__ = [
    "LocalExecutionError",
    "DatasetValidationError",
    "DatasetParseError",
    "ToolValidationError",
    "ProviderError",
    "SystemicProviderError",
]

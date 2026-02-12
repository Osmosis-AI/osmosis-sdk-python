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

__all__ = [
    "LocalExecutionError",
    "DatasetValidationError",
    "DatasetParseError",
    "ToolValidationError",
    "ProviderError",
]

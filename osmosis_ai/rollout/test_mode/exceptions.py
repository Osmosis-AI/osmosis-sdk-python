"""Exceptions for test mode operations.

This module defines custom exceptions for test mode functionality,
including dataset validation errors and provider errors.
"""

from __future__ import annotations


class TestModeError(Exception):
    """Base exception for test mode errors."""

    pass


class DatasetValidationError(TestModeError):
    """Error raised when dataset validation fails.

    This includes missing required columns, invalid value types,
    or malformed file formats.
    """

    pass


class DatasetParseError(TestModeError):
    """Error raised when dataset parsing fails.

    This includes JSON parse errors, invalid file formats,
    or corrupted parquet files.
    """

    pass


class ProviderError(TestModeError):
    """Error raised by LLM providers.

    This includes API errors, authentication failures,
    and rate limiting.
    """

    pass


class ToolValidationError(TestModeError):
    """Error raised when tool schema validation fails.

    This includes missing required fields, invalid names,
    or malformed parameters.
    """

    pass


__all__ = [
    "TestModeError",
    "DatasetValidationError",
    "DatasetParseError",
    "ProviderError",
    "ToolValidationError",
]

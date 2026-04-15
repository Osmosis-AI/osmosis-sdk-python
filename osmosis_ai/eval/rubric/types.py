from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RubricResult:
    """Result from a single rubric evaluation."""

    score: float
    explanation: str
    raw: Any


class MissingAPIKeyError(RuntimeError):
    """Raised when a required provider API key cannot be found."""


class ProviderRequestError(RuntimeError):
    """Raised when a hosted provider call fails for a known reason."""

    def __init__(self, provider: str, model: str, detail: str) -> None:
        self.provider = provider
        self.model = model
        self.detail: str = (
            detail.strip()
            if detail
            else "Provider request failed with no additional detail."
        )
        message = (
            f"Provider '{provider}' request for model '{model}' failed. {self.detail}"
        )
        super().__init__(message)


class ModelNotFoundError(ProviderRequestError):
    """Raised when a provider reports that the requested model cannot be found."""


__all__ = [
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderRequestError",
    "RubricResult",
]

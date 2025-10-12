from __future__ import annotations

from typing import Any, Optional, TypedDict


class ModelInfo(TypedDict, total=False):
    provider: str
    model: str
    api_key: str
    api_key_env: str
    score_min: float
    score_max: float
    system_prompt: Optional[str]
    original_input: Optional[str]
    timeout: float


class RewardRubricRunResult(TypedDict):
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
        self.detail = detail.strip() if detail else "Provider request failed with no additional detail."
        message = f"Provider '{provider}' request for model '{model}' failed. {self.detail}"
        super().__init__(message)


class ModelNotFoundError(ProviderRequestError):
    """Raised when a provider reports that the requested model cannot be found."""


__all__ = [
    "ModelInfo",
    "RewardRubricRunResult",
    "MissingAPIKeyError",
    "ProviderRequestError",
    "ModelNotFoundError",
]

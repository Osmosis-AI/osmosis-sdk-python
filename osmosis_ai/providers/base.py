from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..rubric_types import RewardRubricRunResult

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ProviderRequest:
    provider: str
    model: str
    api_key: str
    system_content: str
    user_content: str
    score_min: float
    score_max: float
    timeout: float


class RubricProvider:
    """Interface for hosted LLM providers that can score rubrics."""

    name: str

    def default_timeout(self, model: str) -> float:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        raise NotImplementedError


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, RubricProvider] = {}

    def register(self, provider: RubricProvider) -> None:
        key = provider.name
        if key in self._providers:
            raise ValueError(f"Provider '{key}' is already registered.")
        self._providers[key] = provider

    def get(self, name: str) -> RubricProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported provider '{name}'.") from exc

    def supported_providers(self) -> Tuple[str, ...]:
        return tuple(sorted(self._providers))


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "ProviderRequest",
    "RubricProvider",
    "ProviderRegistry",
]

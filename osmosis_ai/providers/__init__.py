from __future__ import annotations

from typing import Tuple

from .anthropic_provider import AnthropicProvider
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRegistry, ProviderRequest, RubricProvider
from .gemini_provider import GeminiProvider
from .openai_family import OpenAIProvider, XAIProvider

_REGISTRY = ProviderRegistry()
_REGISTRY.register(OpenAIProvider())
_REGISTRY.register(XAIProvider())
_REGISTRY.register(AnthropicProvider())
_REGISTRY.register(GeminiProvider())


def get_provider(name: str) -> RubricProvider:
    return _REGISTRY.get(name)


def register_provider(provider: RubricProvider) -> None:
    _REGISTRY.register(provider)


def supported_providers() -> Tuple[str, ...]:
    return _REGISTRY.supported_providers()


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "ProviderRequest",
    "RubricProvider",
    "get_provider",
    "register_provider",
    "supported_providers",
]

from __future__ import annotations

from typing import Tuple

from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRegistry, ProviderRequest, RubricProvider
from .anthropic_provider import AnthropicProvider
from .cerebras_provider import CerebrasProvider
from .gemini_provider import GeminiProvider
from .openai_family import OpenAIProvider, XAIProvider
from .openrouter_provider import OpenRouterProvider

_REGISTRY = ProviderRegistry()
_REGISTRY.register(AnthropicProvider())
_REGISTRY.register(CerebrasProvider())
_REGISTRY.register(GeminiProvider())
_REGISTRY.register(OpenAIProvider())
_REGISTRY.register(OpenRouterProvider())
_REGISTRY.register(XAIProvider())


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

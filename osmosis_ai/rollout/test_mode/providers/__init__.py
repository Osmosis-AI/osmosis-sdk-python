"""Provider registry for test mode LLM clients.

This module provides a decorator-based registry for LLM providers with
lazy loading to avoid import errors when optional dependencies are missing.

Lazy Loading Strategy:
    - Built-in providers are loaded on first access (get_provider, list_providers)
    - Import errors are silently ignored for optional providers
    - Users without certain dependencies can still use available providers

Example:
    # Register a custom provider
    from osmosis_ai.rollout.test_mode.providers import register_provider
    from osmosis_ai.rollout.test_mode.providers.base import TestLLMClient

    @register_provider
    class MyCustomClient(TestLLMClient):
        provider_name = "my_provider"
        ...

    # Get a provider by name
    from osmosis_ai.rollout.test_mode.providers import get_provider

    provider_class = get_provider("openai")
    client = provider_class(api_key="...")

    # List available providers
    from osmosis_ai.rollout.test_mode.providers import list_providers

    print(list_providers())  # ["openai", ...]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from osmosis_ai.rollout.test_mode.providers.base import TestLLMClient

logger = logging.getLogger(__name__)

# Provider registry
_PROVIDERS: Dict[str, Type[TestLLMClient]] = {}
_BUILTIN_LOADED: bool = False


def register_provider(cls: Type[TestLLMClient]) -> Type[TestLLMClient]:
    """Decorator to register a TestLLMClient provider.

    The provider_name class attribute is used as the registry key.

    Example:
        @register_provider
        class MyProvider(TestLLMClient):
            provider_name = "my_provider"
            ...

    Args:
        cls: TestLLMClient subclass to register.

    Returns:
        The input class unchanged.

    Raises:
        ValueError: If provider_name is not set.
    """
    name = getattr(cls, "provider_name", None)
    if not name:
        raise ValueError(
            f"Provider class {cls.__name__} must define 'provider_name' attribute"
        )

    _PROVIDERS[name] = cls
    logger.debug("Registered provider: %s -> %s", name, cls.__name__)
    return cls


def _load_builtin_providers() -> None:
    """Lazily load built-in providers.

    Uses lazy loading to avoid import errors when:
        - Optional provider dependencies are not installed
        - We want fast module import without loading all providers

    Import errors are silently ignored - providers simply won't be available.
    This is intentional: users who don't need a provider shouldn't need its deps.
    """
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True

    # OpenAI - core dependency, should always work
    try:
        from osmosis_ai.rollout.test_mode.providers.openai import (  # noqa: F401
            OpenAITestClient,
        )

        register_provider(OpenAITestClient)
    except ImportError as e:
        logger.debug("Failed to load OpenAI provider: %s", e)

    # Future providers (Phase 2+):
    # - Anthropic: osmosis_ai.rollout.test_mode.providers.anthropic
    # - Google Gemini: osmosis_ai.rollout.test_mode.providers.google
    # Add new provider imports here following the pattern above.


def get_provider(name: str) -> Type[TestLLMClient]:
    """Get a provider class by name.

    Lazily loads built-in providers on first call.

    Args:
        name: Provider name (e.g., "openai", "anthropic").

    Returns:
        TestLLMClient subclass for the provider.

    Raises:
        ValueError: If provider is not found.

    Example:
        provider_class = get_provider("openai")
        client = provider_class(api_key="...", model="gpt-4o")
    """
    _load_builtin_providers()

    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys())) or "(none)"
        raise ValueError(f"Unknown provider: '{name}'. Available: {available}")

    return _PROVIDERS[name]


def list_providers() -> List[str]:
    """List all registered provider names.

    Lazily loads built-in providers on first call.

    Returns:
        Sorted list of provider names.

    Example:
        providers = list_providers()
        print(providers)  # ["openai", ...]
    """
    _load_builtin_providers()
    return sorted(_PROVIDERS.keys())


__all__ = [
    "TestLLMClient",
    "get_provider",
    "list_providers",
    "register_provider",
]

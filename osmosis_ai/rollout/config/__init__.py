"""Configuration management for Osmosis rollout SDK.

This module provides type-safe configuration via pydantic-settings,
with support for environment variables and .env files.

Example:
    from osmosis_ai.rollout.config import get_settings, configure

    # Use default settings (from environment variables)
    settings = get_settings()
    print(settings.client.timeout_seconds)

    # Override settings programmatically
    from osmosis_ai.rollout.config import RolloutSettings, RolloutClientSettings
    configure(RolloutSettings(
        client=RolloutClientSettings(timeout_seconds=120),
    ))
"""

from osmosis_ai.rollout.config.settings import (
    RolloutClientSettings,
    RolloutServerSettings,
    RolloutSettings,
    configure,
    get_settings,
    reset_settings,
)

__all__ = [
    # Settings classes
    "RolloutSettings",
    "RolloutClientSettings",
    "RolloutServerSettings",
    # Functions
    "get_settings",
    "configure",
    "reset_settings",
]

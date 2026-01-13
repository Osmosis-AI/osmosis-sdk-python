"""Configuration settings for Osmosis rollout SDK.

This module provides Pydantic-based settings classes that can be configured
via environment variables, .env files, or programmatically.

Environment Variable Prefixes:
    - OSMOSIS_ROLLOUT_CLIENT_* - Client settings
    - OSMOSIS_ROLLOUT_SERVER_* - Server settings

Example:
    # Set via environment variables
    export OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120

    # Use in code
    from osmosis_ai.rollout.config import get_settings
    settings = get_settings()
    print(settings.client.timeout_seconds)  # 120.0
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from osmosis_ai.rollout._compat import (
    PYDANTIC_SETTINGS_AVAILABLE,
    pydantic_settings,
)


# Conditionally use pydantic-settings BaseSettings or fallback to BaseModel
if PYDANTIC_SETTINGS_AVAILABLE:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _BaseSettings = BaseSettings
else:
    # Fallback: use BaseModel (no env var loading)
    _BaseSettings = BaseModel  # type: ignore[misc]
    SettingsConfigDict = None  # type: ignore[misc, assignment]


class RolloutClientSettings(_BaseSettings):
    """HTTP client configuration.

    Loaded from environment variables with prefix: OSMOSIS_ROLLOUT_CLIENT_

    Attributes:
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for 5xx errors.
        complete_rollout_retries: Maximum retries for completion callback.
        retry_base_delay: Base delay for exponential backoff in seconds.
        retry_max_delay: Maximum delay between retries in seconds.
        max_connections: Maximum number of HTTP connections.
        max_keepalive_connections: Maximum keepalive connections.

    Example:
        export OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120
        export OSMOSIS_ROLLOUT_CLIENT_MAX_RETRIES=5
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_CLIENT_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    timeout_seconds: float = Field(
        default=300.0,
        description="HTTP request timeout in seconds",
        ge=1.0,
        le=3600.0,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for 5xx errors",
        ge=0,
        le=10,
    )
    complete_rollout_retries: int = Field(
        default=2,
        description="Maximum retries for completion callback",
        ge=0,
        le=10,
    )
    retry_base_delay: float = Field(
        default=1.0,
        description="Base delay for exponential backoff in seconds",
        ge=0.1,
        le=60.0,
    )
    retry_max_delay: float = Field(
        default=30.0,
        description="Maximum delay between retries in seconds",
        ge=1.0,
        le=300.0,
    )
    max_connections: int = Field(
        default=100,
        description="Maximum number of HTTP connections",
        ge=1,
        le=1000,
    )
    max_keepalive_connections: int = Field(
        default=20,
        description="Maximum keepalive connections",
        ge=1,
        le=100,
    )


class RolloutServerSettings(_BaseSettings):
    """Server configuration.

    Loaded from environment variables with prefix: OSMOSIS_ROLLOUT_SERVER_

    Attributes:
        max_concurrent_rollouts: Maximum number of concurrent rollouts.
        record_ttl_seconds: How long to keep completed rollout records.
        cleanup_interval_seconds: Interval for cleanup task.
        request_timeout_seconds: Timeout for individual requests.

    Example:
        export OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS=200
        export OSMOSIS_ROLLOUT_SERVER_RECORD_TTL_SECONDS=7200
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_SERVER_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    max_concurrent_rollouts: int = Field(
        default=100,
        description="Maximum number of concurrent rollouts",
        ge=1,
        le=10000,
    )
    record_ttl_seconds: float = Field(
        default=3600.0,
        description="How long to keep completed rollout records in seconds",
        ge=60.0,
        le=86400.0,
    )
    cleanup_interval_seconds: float = Field(
        default=60.0,
        description="Interval for cleanup task in seconds",
        ge=10.0,
        le=3600.0,
    )
    request_timeout_seconds: float = Field(
        default=600.0,
        description="Timeout for individual requests in seconds",
        ge=10.0,
        le=3600.0,
    )
    registration_readiness_timeout_seconds: float = Field(
        default=10.0,
        description=(
            "Maximum time to wait for the server to become ready before platform registration. "
            "The server polls its own health endpoint to confirm readiness."
        ),
        ge=1.0,
        le=60.0,
    )
    registration_readiness_poll_interval_seconds: float = Field(
        default=0.2,
        description="Interval between health check polls during server readiness check.",
        ge=0.05,
        le=5.0,
    )
    registration_shutdown_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for waiting for platform registration to complete during shutdown.",
        ge=1.0,
        le=300.0,
    )


class RolloutSettings(_BaseSettings):
    """Main configuration for Osmosis rollout SDK.

    Aggregates all sub-configurations and supports environment variable overrides.

    Attributes:
        client: HTTP client settings.
        server: Server settings.
        max_metadata_size_bytes: Maximum size for metadata in bytes.

    Example:
        # Use defaults (from environment variables)
        settings = RolloutSettings()

        # Override programmatically
        settings = RolloutSettings(
            client=RolloutClientSettings(timeout_seconds=120),
        )

    Environment Variables:
        OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES - Maximum metadata size
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    client: RolloutClientSettings = Field(default_factory=RolloutClientSettings)
    server: RolloutServerSettings = Field(default_factory=RolloutServerSettings)

    # Global settings
    max_metadata_size_bytes: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum size for metadata in bytes",
        ge=1024,
        le=100 * 1024 * 1024,  # 100MB max
    )


# Global settings singleton
_settings: Optional[RolloutSettings] = None


def get_settings() -> RolloutSettings:
    """Get the global settings singleton.

    Loads settings from environment variables on first call.

    Returns:
        The global RolloutSettings instance.

    Example:
        settings = get_settings()
        timeout = settings.client.timeout_seconds
    """
    global _settings
    if _settings is None:
        _settings = RolloutSettings()
    return _settings


def configure(settings: RolloutSettings) -> None:
    """Set the global settings.

    Allows programmatic configuration to override environment variables.

    Args:
        settings: The settings to use globally.

    Example:
        from osmosis_ai.rollout.config import configure, RolloutSettings

        configure(RolloutSettings(
            client=RolloutClientSettings(timeout_seconds=120),
        ))
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset global settings to None.

    Primarily used for testing to ensure clean state between tests.
    """
    global _settings
    _settings = None


__all__ = [
    # Settings classes
    "RolloutClientSettings",
    "RolloutServerSettings",
    "RolloutSettings",
    # Functions
    "get_settings",
    "configure",
    "reset_settings",
]

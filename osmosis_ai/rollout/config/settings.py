"""Configuration settings for Osmosis rollout SDK.

This module provides Pydantic-based settings classes that can be configured
via environment variables, .env files, or programmatically.

Environment Variable Prefixes:
    - OSMOSIS_ROLLOUT_CLIENT_* - Client settings
    - OSMOSIS_ROLLOUT_SERVER_* - Server settings
    - OSMOSIS_ROLLOUT_LOG_* - Logging settings
    - OSMOSIS_ROLLOUT_TRACE_* - Tracing settings
    - OSMOSIS_ROLLOUT_METRICS_* - Metrics settings

Example:
    # Set via environment variables
    export OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120
    export OSMOSIS_ROLLOUT_LOG_LEVEL=DEBUG

    # Use in code
    from osmosis_ai.rollout.config import get_settings
    settings = get_settings()
    print(settings.client.timeout_seconds)  # 120.0
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

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


class LoggingSettings(_BaseSettings):
    """Logging configuration.

    Loaded from environment variables with prefix: OSMOSIS_ROLLOUT_LOG_

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: Log format (json, console, plain).
        include_timestamp: Whether to include timestamp in logs.
        include_caller: Whether to include caller information.

    Example:
        export OSMOSIS_ROLLOUT_LOG_LEVEL=DEBUG
        export OSMOSIS_ROLLOUT_LOG_FORMAT=json
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_LOG_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level",
    )
    format: Literal["json", "console", "plain"] = Field(
        default="json",
        description="Log format",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Whether to include timestamp in logs",
    )
    include_caller: bool = Field(
        default=False,
        description="Whether to include caller information",
    )


class TracingSettings(_BaseSettings):
    """OpenTelemetry tracing configuration.

    Loaded from environment variables with prefix: OSMOSIS_ROLLOUT_TRACE_

    Attributes:
        enabled: Whether tracing is enabled.
        service_name: Service name for traces.
        exporter: Trace exporter type (otlp, jaeger, zipkin, console, none).
        endpoint: OTLP endpoint URL.
        sample_rate: Sampling rate (0.0 to 1.0).
        propagators: Trace context propagators to use.

    Example:
        export OSMOSIS_ROLLOUT_TRACE_ENABLED=true
        export OSMOSIS_ROLLOUT_TRACE_EXPORTER=otlp
        export OSMOSIS_ROLLOUT_TRACE_ENDPOINT=http://jaeger:4317
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_TRACE_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    enabled: bool = Field(
        default=False,
        description="Whether tracing is enabled",
    )
    service_name: str = Field(
        default="osmosis-rollout",
        description="Service name for traces",
    )
    exporter: Literal["otlp", "jaeger", "zipkin", "console", "none"] = Field(
        default="otlp",
        description="Trace exporter type",
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="OTLP endpoint URL (e.g., http://localhost:4317)",
    )
    sample_rate: float = Field(
        default=1.0,
        description="Sampling rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    propagators: str = Field(
        default="tracecontext,b3",
        description="Comma-separated list of propagators",
    )


class MetricsSettings(_BaseSettings):
    """Prometheus metrics configuration.

    Loaded from environment variables with prefix: OSMOSIS_ROLLOUT_METRICS_

    Attributes:
        enabled: Whether metrics collection is enabled.
        prefix: Prefix for metric names.
        include_default_labels: Whether to include default labels.
        expose_endpoint: Whether to expose /metrics endpoint.

    Example:
        export OSMOSIS_ROLLOUT_METRICS_ENABLED=true
        export OSMOSIS_ROLLOUT_METRICS_PREFIX=my_agent
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_prefix="OSMOSIS_ROLLOUT_METRICS_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    enabled: bool = Field(
        default=False,
        description="Whether metrics collection is enabled",
    )
    prefix: str = Field(
        default="osmosis_rollout",
        description="Prefix for metric names",
    )
    include_default_labels: bool = Field(
        default=True,
        description="Whether to include default labels (service, version)",
    )
    expose_endpoint: bool = Field(
        default=True,
        description="Whether to expose /metrics endpoint when enabled",
    )


class RolloutSettings(_BaseSettings):
    """Main configuration for Osmosis rollout SDK.

    Aggregates all sub-configurations and supports environment variable overrides.

    Attributes:
        client: HTTP client settings.
        server: Server settings.
        logging: Logging settings.
        tracing: OpenTelemetry tracing settings.
        metrics: Prometheus metrics settings.
        max_metadata_size_bytes: Maximum size for metadata in bytes.

    Example:
        # Use defaults (from environment variables)
        settings = RolloutSettings()

        # Override programmatically
        settings = RolloutSettings(
            client=RolloutClientSettings(timeout_seconds=120),
            logging=LoggingSettings(level="DEBUG"),
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
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

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
    "LoggingSettings",
    "TracingSettings",
    "MetricsSettings",
    "RolloutSettings",
    # Functions
    "get_settings",
    "configure",
    "reset_settings",
]

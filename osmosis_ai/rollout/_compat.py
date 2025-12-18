"""Optional dependency compatibility layer.

This module provides a unified way to check and import optional dependencies,
with graceful fallbacks when dependencies are not available.

Example:
    from osmosis_ai.rollout._compat import (
        STRUCTLOG_AVAILABLE,
        PROMETHEUS_AVAILABLE,
        require_optional,
    )

    if STRUCTLOG_AVAILABLE:
        import structlog
        logger = structlog.get_logger()
    else:
        import logging
        logger = logging.getLogger(__name__)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def import_optional(
    module_name: str,
    package_name: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Attempt to import an optional module.

    Args:
        module_name: The module to import.
        package_name: The pip package name (for error messages).

    Returns:
        Tuple of (module, available) where module is None if not available.

    Example:
        structlog, available = import_optional("structlog")
        if available:
            logger = structlog.get_logger()
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        return module, True
    except ImportError:
        return None, False


def require_optional(
    module_name: str,
    package_name: Optional[str] = None,
    feature_name: str = "",
    install_extra: Optional[str] = None,
) -> Any:
    """Import an optional module, raising a friendly error if unavailable.

    Args:
        module_name: The module to import.
        package_name: The pip package name.
        feature_name: Human-readable feature name for error messages.
        install_extra: The extras_require key for installation hint.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not available.

    Example:
        structlog = require_optional(
            "structlog",
            feature_name="structured logging",
            install_extra="logging",
        )
    """
    module, available = import_optional(module_name)
    if not available:
        pkg = package_name or module_name
        feature = feature_name or module_name
        extra = install_extra or pkg

        raise ImportError(
            f"{feature} requires '{pkg}'. "
            f"Install it with: pip install osmosis-ai[{extra}]"
        )
    return module


# Pre-defined availability checks for common optional dependencies

# Structured logging
structlog, STRUCTLOG_AVAILABLE = import_optional("structlog")

# Prometheus metrics
prometheus_client, PROMETHEUS_AVAILABLE = import_optional("prometheus_client")

# OpenTelemetry tracing
_otel_api, OTEL_API_AVAILABLE = import_optional("opentelemetry.trace")
_otel_sdk, OTEL_SDK_AVAILABLE = import_optional("opentelemetry.sdk.trace")
OTEL_AVAILABLE = OTEL_API_AVAILABLE and OTEL_SDK_AVAILABLE

# OpenTelemetry OTLP exporter
_otlp_exporter, OTLP_EXPORTER_AVAILABLE = import_optional(
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
)

# Pydantic settings
pydantic_settings, PYDANTIC_SETTINGS_AVAILABLE = import_optional("pydantic_settings")

# FastAPI (for server)
fastapi, FASTAPI_AVAILABLE = import_optional("fastapi")


__all__ = [
    # Functions
    "import_optional",
    "require_optional",
    # Availability flags
    "STRUCTLOG_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "OTEL_AVAILABLE",
    "OTEL_API_AVAILABLE",
    "OTEL_SDK_AVAILABLE",
    "OTLP_EXPORTER_AVAILABLE",
    "PYDANTIC_SETTINGS_AVAILABLE",
    "FASTAPI_AVAILABLE",
    # Pre-imported modules (may be None)
    "structlog",
    "prometheus_client",
    "pydantic_settings",
    "fastapi",
]

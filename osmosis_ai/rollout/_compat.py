"""Optional dependency compatibility layer.

This module provides a unified way to check and import optional dependencies,
with graceful fallbacks when dependencies are not available.

Example:
    from osmosis_ai.rollout._compat import (
        PYDANTIC_SETTINGS_AVAILABLE,
        require_optional,
    )

    if PYDANTIC_SETTINGS_AVAILABLE:
        from pydantic_settings import BaseSettings
    else:
        from pydantic import BaseModel as BaseSettings
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
        pydantic_settings, available = import_optional("pydantic_settings")
        if available:
            from pydantic_settings import BaseSettings
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
        pydantic_settings = require_optional(
            "pydantic_settings",
            feature_name="configuration management",
            install_extra="config",
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

# Pydantic settings
pydantic_settings, PYDANTIC_SETTINGS_AVAILABLE = import_optional("pydantic_settings")

# FastAPI (for server)
fastapi, FASTAPI_AVAILABLE = import_optional("fastapi")

# Uvicorn (for server)
uvicorn, UVICORN_AVAILABLE = import_optional("uvicorn")


__all__ = [
    # Functions
    "import_optional",
    "require_optional",
    # Availability flags
    "PYDANTIC_SETTINGS_AVAILABLE",
    "FASTAPI_AVAILABLE",
    "UVICORN_AVAILABLE",
    # Pre-imported modules (may be None)
    "pydantic_settings",
    "fastapi",
    "uvicorn",
]

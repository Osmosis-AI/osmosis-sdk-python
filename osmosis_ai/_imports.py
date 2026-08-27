"""Helpers for lazy public exports and optional dependency errors."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Final, NoReturn

LazyExports = Mapping[str, tuple[str, str]]

# Top-level import names installed by each optional extra's direct
# dependencies. Each set is deliberately complete: the install hint is correct
# advice for any module the extra provides, so a boundary never has to guess
# which subset it imports. tests/unit/test_public_api_imports.py asserts each
# set matches the extra declared in pyproject.toml.
EXTRA_MODULES: Final[Mapping[str, frozenset[str]]] = {
    "server": frozenset({"click", "fastapi", "uvicorn"}),
    "strands": frozenset({"aiohttp", "click", "litellm", "mcp", "orjson", "strands"}),
    "openai-agents": frozenset(
        {"agents", "aiohttp", "click", "litellm", "mcp", "orjson"}
    ),
    "harbor": frozenset(
        {
            "aiohttp",
            "click",
            "dockerfile_parse",
            "harbor",
            "litellm",
            "orjson",
        }
    ),
    "rubric": frozenset({"aiohttp", "click", "litellm", "orjson"}),
    "parquet": frozenset({"pyarrow"}),
    "eval": frozenset({"click", "fastapi", "litellm", "pyarrow", "uvicorn"}),
}


def import_attribute(module_name: str, attribute_name: str) -> object:
    """Import one attribute without making its leaf module part of a facade."""
    return getattr(import_module(module_name), attribute_name)


def resolve_lazy_export(
    name: str,
    *,
    module_name: str,
    namespace: dict[str, object],
    exports: LazyExports,
) -> object:
    """Resolve and cache a name from a symbol-to-leaf export map."""
    try:
        target_module, target_name = exports[name]
    except KeyError:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from None

    value = import_attribute(target_module, target_name)
    namespace[name] = value
    return value


def public_module_dir(
    namespace: Mapping[str, object], exports: LazyExports
) -> list[str]:
    """Return module attributes plus unresolved lazy public exports."""
    return sorted(namespace.keys() | exports.keys())


def raise_optional_dependency_error(
    error: ModuleNotFoundError,
    *,
    extra: str,
    feature: str,
) -> NoReturn:
    """Add an install hint only when a module owned by ``extra`` is absent.

    A missing SDK module or a missing submodule inside an installed dependency
    is left untouched. Those indicate a packaging bug or an incompatible
    dependency rather than an omitted extra and should retain their traceback.
    """
    if error.name not in EXTRA_MODULES[extra]:
        raise error

    message = (
        f"{feature} requires optional dependencies. Install them with "
        f'`pip install "osmosis-ai[{extra}]"`.'
    )
    raise ModuleNotFoundError(message, name=error.name, path=error.path) from error

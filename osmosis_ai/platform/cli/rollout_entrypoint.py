"""Import a workspace rollout entrypoint for submit preflight."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

from osmosis_ai.cli.errors import CLIError


def _synthetic_package_name(rollout_dir: Path) -> str:
    digest = hashlib.sha256(str(rollout_dir).encode("utf-8")).hexdigest()[:16]
    return f"_osmosis_rollout_{digest}"


def _clear_package_cache(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def _load_package(package_name: str, package_dir: Path) -> types.ModuleType:
    init_py = package_dir / "__init__.py"
    if init_py.is_file():
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_py,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise CLIError(f"Failed to load rollout package: {package_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return module

    module = types.ModuleType(package_name)
    module.__file__ = str(init_py)
    module.__package__ = package_name
    module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[package_name] = module
    return module


def _load_parent_packages(
    package_name: str,
    rollout_dir: Path,
    entrypoint_path: Path,
) -> None:
    parts = entrypoint_path.relative_to(rollout_dir).with_suffix("").parts[:-1]
    current_dir = rollout_dir
    current_package = package_name
    for part in parts:
        current_dir = current_dir / part
        current_package = f"{current_package}.{part}"
        _load_package(current_package, current_dir)


def load_rollout_entrypoint(
    rollout_dir: Path,
    entrypoint: str,
) -> types.ModuleType:
    """Import one rollout entrypoint without inspecting its module namespace."""
    rollout_dir = rollout_dir.resolve()
    entrypoint_path = (rollout_dir / entrypoint).resolve()
    try:
        entrypoint_path.relative_to(rollout_dir)
    except ValueError as exc:
        raise CLIError(
            f"Entrypoint must stay within {rollout_dir}: {entrypoint}"
        ) from exc

    if not entrypoint_path.is_file():
        raise CLIError(f"Entrypoint file not found: {entrypoint_path}")

    rollout_dir_str = str(rollout_dir)
    if rollout_dir_str not in sys.path:
        # Rollouts commonly use absolute imports for packages beside the entrypoint.
        sys.path.insert(0, rollout_dir_str)

    package_name = _synthetic_package_name(rollout_dir)
    _clear_package_cache(package_name)
    _load_package(package_name, rollout_dir)
    _load_parent_packages(package_name, rollout_dir, entrypoint_path)

    relative_parts = entrypoint_path.relative_to(rollout_dir).with_suffix("").parts
    module_name = ".".join((package_name, *relative_parts))
    spec = importlib.util.spec_from_file_location(module_name, entrypoint_path)
    if spec is None or spec.loader is None:
        raise CLIError(f"Failed to load entrypoint module: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_rollout_entrypoint"]

"""Import path resolution utilities.

Supports the `module.submodule:attr` notation for referencing Python objects
across process boundaries (e.g., LocalBackend ↔ HarborBackend).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


def resolve_object(ref: str | Any) -> Any:
    """Resolve a reference to a Python object.

    If ref is a string like 'my_rollout.workflow:MultiplyWorkflow', imports
    the module and returns the attribute. If already a Python object, returns as-is.
    """
    if not isinstance(ref, str):
        return ref

    if ":" not in ref:
        raise ValueError(f"Invalid import path '{ref}' — expected 'module:attr' format")

    module_path, attr_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    obj = getattr(mod, attr_name, None)
    if obj is None:
        raise ImportError(f"{module_path} does not export '{attr_name}'")
    return obj


def to_import_path(obj: Any) -> str:
    """Convert a Python object to its 'module:attr' import path string.

    Works for classes (via __module__/__qualname__) and module-level
    instances (found by identity in the defining module's namespace).
    """
    if isinstance(obj, type):
        return f"{obj.__module__}:{obj.__qualname__}"

    # Module-level instance — search by identity
    obj_module = getattr(type(obj), "__module__", None)
    if obj_module:
        mod = sys.modules.get(obj_module)
        if mod:
            for name, val in vars(mod).items():
                if val is obj and not name.startswith("_"):
                    return f"{mod.__name__}:{name}"

    raise ValueError(
        f"Cannot resolve import path for {obj!r}. "
        "Ensure it is a class or a module-level variable."
    )

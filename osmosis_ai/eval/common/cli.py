"""Shared helpers for local-execution CLI commands (`test` and `eval`)."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.common.dataset import DatasetReader
from osmosis_ai.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
)

if TYPE_CHECKING:
    from osmosis_ai.eval.common.dataset import DatasetRow


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}m{seconds:.1f}s"


def format_tokens(tokens: int) -> str:
    """Format token count with comma separators."""
    return f"{tokens:,}"


def truncate_error(text: str, max_len: int = 50) -> str:
    """Truncate a single-line error string with ellipsis if too long."""
    flat = text.replace("\n", " ")
    return flat[: max_len - 3] + "..." if len(flat) > max_len else flat


def _resolve_rollout_entrypoint(
    rollout: str,
    entrypoint: str,
    *,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the rollout root and entrypoint file path."""
    project_root = (project_root or Path.cwd()).resolve()
    rollout_dir = (project_root / "rollouts" / rollout).resolve()
    if not rollout_dir.is_dir():
        raise CLIError(
            f"Rollout directory not found: rollouts/{rollout}/\n"
            f"  Expected at: {rollout_dir}"
        )

    entrypoint_rel = Path(entrypoint)
    if entrypoint_rel.is_absolute():
        raise CLIError(
            f"Entrypoint must be a path relative to rollouts/{rollout}/, got: {entrypoint}"
        )
    if entrypoint_rel.suffix != ".py":
        raise CLIError(
            f"Entrypoint must point to a Python file ending in .py, got: {entrypoint}"
        )

    entrypoint_path = (rollout_dir / entrypoint_rel).resolve()
    try:
        entrypoint_path.relative_to(rollout_dir)
    except ValueError as exc:
        raise CLIError(
            f"Entrypoint must stay within rollouts/{rollout}/, got: {entrypoint}"
        ) from exc

    if not entrypoint_path.is_file():
        raise CLIError(
            f"Entrypoint file not found in rollouts/{rollout}/: {entrypoint}\n"
            f"  Expected at: {entrypoint_path}"
        )

    return rollout_dir, entrypoint_path


def _synthetic_rollout_package_name(rollout_dir: Path) -> str:
    digest = hashlib.sha256(str(rollout_dir).encode("utf-8")).hexdigest()[:16]
    return f"_osmosis_rollout_{digest}"


def _clear_rollout_module_cache(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def _load_package_module(package_name: str, package_dir: Path) -> types.ModuleType:
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


def _ensure_parent_packages(
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
        _load_package_module(current_package, current_dir)


def _ensure_rollout_dir_on_path(rollout_dir: Path) -> None:
    """Add the rollout directory to ``sys.path`` so sibling packages resolve.

    The synthetic-package wrapper isolates the entrypoint module itself, but
    real-world entrypoints commonly do absolute imports of sibling packages
    that live next to them (e.g. ``from multiply_openai_agents.grader import
    ...`` next to ``local_rollout_server_openai_agents_example.py``). Those
    are top-level imports, so the rollout directory must be searchable via
    ``sys.path`` for them to resolve.
    """
    rollout_dir_str = str(rollout_dir)
    if rollout_dir_str not in sys.path:
        sys.path.insert(0, rollout_dir_str)


def _load_rollout_module(
    rollout: str,
    entrypoint: str,
    *,
    project_root: Path | None = None,
) -> types.ModuleType:
    """Load an entrypoint as an isolated synthetic package subtree."""
    rollout_dir, entrypoint_path = _resolve_rollout_entrypoint(
        rollout,
        entrypoint,
        project_root=project_root,
    )
    _ensure_rollout_dir_on_path(rollout_dir)
    package_name = _synthetic_rollout_package_name(rollout_dir)
    _clear_rollout_module_cache(package_name)
    _load_package_module(package_name, rollout_dir)
    _ensure_parent_packages(package_name, rollout_dir, entrypoint_path)

    relative_parts = entrypoint_path.relative_to(rollout_dir).with_suffix("").parts
    module_name = ".".join((package_name, *relative_parts))
    spec = importlib.util.spec_from_file_location(module_name, entrypoint_path)
    if spec is None or spec.loader is None:
        raise CLIError(f"Failed to load entrypoint module: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _group_by_object_id(
    pairs: list[tuple[str, Any]],
) -> dict[int, list[tuple[str, Any]]]:
    """Group (binding name, object) pairs by object identity."""
    groups: dict[int, list[tuple[str, Any]]] = {}
    for name, obj in pairs:
        groups.setdefault(id(obj), []).append((name, obj))
    return groups


def _format_ambiguous_binding_names(pairs: list[tuple[str, Any]]) -> str:
    """Sorted unique binding names for an ambiguous candidate set."""
    return ", ".join(sorted({n for n, _ in pairs}))


def _pick_representative(pairs_for_one_object: list[tuple[str, Any]]) -> Any:
    """Deterministic pick: object referred to by the lexicographically smallest name."""
    _name, obj = min(pairs_for_one_object, key=lambda x: x[0])
    return obj


def _resolve_workflow(
    rollout: str,
    entrypoint: str,
    *,
    project_root: Path | None = None,
) -> tuple[type, Any, str]:
    """Resolve an AgentWorkflow subclass and its config.

    Converts the entrypoint file path to a module, imports it,
    and auto-discovers an AgentWorkflow subclass and optional config.

    Returns (workflow_cls, config, entrypoint_module_name) where config
    may be None.  *entrypoint_module_name* is the ``__name__`` of the
    loaded entrypoint module — callers should use this (not
    ``workflow_cls.__module__``) when discovering a Grader, because the
    workflow class may have been defined in a different file and merely
    imported into the entrypoint.
    """
    from osmosis_ai.rollout.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout.types import AgentWorkflowConfig

    mod = _load_rollout_module(
        rollout,
        entrypoint,
        project_root=project_root,
    )

    workflow_pairs = [
        (n, v)
        for n, v in vars(mod).items()
        if isinstance(v, type)
        and issubclass(v, AgentWorkflow)
        and v is not AgentWorkflow
    ]
    wf_groups = _group_by_object_id(workflow_pairs)
    if len(wf_groups) == 0:
        raise CLIError(f"No AgentWorkflow subclass found in '{entrypoint}'")
    if len(wf_groups) > 1:
        raise CLIError(
            f"Multiple AgentWorkflow subclasses found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(workflow_pairs)}. "
            "Keep only one AgentWorkflow subclass in the entrypoint module, or move "
            "extra classes to a separate file."
        )
    workflow_cls = _pick_representative(next(iter(wf_groups.values())))

    config_pairs = [
        (n, v) for n, v in vars(mod).items() if isinstance(v, AgentWorkflowConfig)
    ]
    cfg_groups = _group_by_object_id(config_pairs)
    if len(cfg_groups) > 1:
        raise CLIError(
            f"Multiple AgentWorkflowConfig instances found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(config_pairs)}. "
            "Keep only one AgentWorkflowConfig instance in the entrypoint module, or move "
            "extras to a separate file."
        )
    config = (
        _pick_representative(next(iter(cfg_groups.values())))
        if len(cfg_groups) == 1
        else None
    )

    return workflow_cls, config, mod.__name__


def load_workflow(
    rollout: str,
    entrypoint: str,
    quiet: bool = False,
    console: Console | None = None,
    project_root: Path | None = None,
) -> tuple[type | None, Any, str | None, str | None]:
    """Load an AgentWorkflow class and its config.

    Returns (workflow_cls, workflow_config, entrypoint_module_name, error).
    """
    if console and not quiet:
        console.print(f"Loading workflow: {entrypoint}")

    try:
        workflow_cls, workflow_config, entrypoint_module = _resolve_workflow(
            rollout=rollout,
            entrypoint=entrypoint,
            project_root=project_root,
        )
    except Exception as e:
        detail = str(e)
        if not isinstance(e, (CLIError, ImportError, ValueError, TypeError)):
            detail = f"{type(e).__name__}: {detail}"
        return None, None, None, detail

    if console and not quiet:
        console.print(f"  Workflow: {workflow_cls.__name__}")

    return workflow_cls, workflow_config, entrypoint_module, None


def auto_discover_grader(
    module_name: str,
    *,
    entrypoint_label: str | None = None,
) -> tuple[type | None, Any]:
    """Discover a Grader subclass and its config from the entrypoint module.

    The entrypoint file (e.g., ``local_rollout_server_example.py``) typically
    imports the Grader alongside the Workflow, so scanning its namespace is
    sufficient — no need to walk the entire package.

    Returns (grader_cls, grader_config) or (None, None) if not found.
    """
    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None

    return _discover_grader_from_module(mod, entrypoint_label or module_name)


def load_dataset_rows(
    dataset_path: str,
    limit: int | None,
    offset: int,
    quiet: bool,
    console: Console,
    empty_error: str,
    action_label: str,
) -> tuple[list[DatasetRow] | None, str | None]:
    """Load rows from dataset and provide command-specific messaging."""
    if not quiet:
        console.print(f"Loading dataset: {dataset_path}")

    try:
        reader = DatasetReader(dataset_path)
        total_rows = len(reader)
        rows = reader.read(limit=limit, offset=offset)
    except FileNotFoundError as e:
        return None, str(e)
    except (DatasetParseError, DatasetValidationError) as e:
        return None, str(e)

    if not rows:
        return None, empty_error

    if not quiet:
        if limit:
            console.print(f"  Total rows: {total_rows} ({action_label} {len(rows)})")
        else:
            console.print(f"  Total rows: {len(rows)}")

    return rows, None


def build_completion_params(
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build completion params dict from CLI options."""
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return params


def _discover_grader_from_module(
    mod: Any,
    entrypoint: str,
) -> tuple[type | None, Any]:
    """Pick Grader subclass and GraderConfig from a loaded module namespace."""
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

    grader_pairs = [
        (n, v)
        for n, v in vars(mod).items()
        if isinstance(v, type) and issubclass(v, Grader) and v is not Grader
    ]
    grader_groups = _group_by_object_id(grader_pairs)
    if len(grader_groups) > 1:
        raise CLIError(
            f"Multiple Grader subclasses found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(grader_pairs)}. "
            "Keep only one Grader subclass in the entrypoint module, or move extra "
            "classes to a separate file."
        )
    grader_cls: type | None = None
    if len(grader_groups) == 1:
        grader_cls = _pick_representative(next(iter(grader_groups.values())))

    config_pairs = [(n, v) for n, v in vars(mod).items() if isinstance(v, GraderConfig)]
    cfg_groups = _group_by_object_id(config_pairs)
    if len(cfg_groups) > 1:
        raise CLIError(
            f"Multiple GraderConfig instances found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(config_pairs)}. "
            "Keep only one GraderConfig instance in the entrypoint module, or move "
            "extras to a separate file."
        )
    grader_config = None
    if len(cfg_groups) == 1:
        grader_config = _pick_representative(next(iter(cfg_groups.values())))

    return grader_cls, grader_config


def _resolve_grader(
    module_name: str,
    explicit_grader: str | None = None,
    explicit_config: str | None = None,
) -> tuple[type | None, Any]:
    """Resolve Grader from explicit path or auto-discover from workflow module.

    Only called when [grader] is present in TOML. Returns (None, None) when
    no grader is found.
    """
    import sys

    from osmosis_ai.rollout.utils.imports import resolve_object

    if explicit_grader:
        from osmosis_ai.rollout.grader import Grader
        from osmosis_ai.rollout.types import GraderConfig

        grader_cls = resolve_object(explicit_grader)
        if (
            not isinstance(grader_cls, type)
            or not issubclass(grader_cls, Grader)
            or grader_cls is Grader
        ):
            raise CLIError(
                f"[grader].module must point to a concrete Grader subclass, "
                f"but '{explicit_grader}' resolved to {grader_cls!r}"
            )

        grader_config = resolve_object(explicit_config) if explicit_config else None
        if grader_config is not None and not isinstance(grader_config, GraderConfig):
            raise CLIError(
                f"[grader].config must point to a GraderConfig instance, "
                f"but '{explicit_config}' resolved to {type(grader_config).__name__}"
            )
        return grader_cls, grader_config

    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None

    return _discover_grader_from_module(mod, module_name)


__all__ = [
    "_resolve_grader",
    "auto_discover_grader",
    "build_completion_params",
    "format_duration",
    "format_tokens",
    "load_dataset_rows",
    "load_workflow",
    "truncate_error",
]

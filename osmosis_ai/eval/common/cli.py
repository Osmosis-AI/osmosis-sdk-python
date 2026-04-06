"""Shared helpers for local-execution CLI commands (`test` and `eval`)."""

from __future__ import annotations

import os
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


def _ensure_rollout_on_path(rollout: str | None) -> str | None:
    """Add rollouts/<name>/ to sys.path if rollout is specified.

    Returns the rollout directory path, or None.
    """
    if not rollout:
        return None

    import sys

    cwd = os.getcwd()
    rollout_dir = os.path.join(cwd, "rollouts", rollout)
    if not os.path.isdir(rollout_dir):
        raise CLIError(
            f"Rollout directory not found: rollouts/{rollout}/\n"
            f"  Expected at: {rollout_dir}"
        )
    if rollout_dir not in sys.path:
        sys.path.insert(0, rollout_dir)
    return rollout_dir


def _entrypoint_to_module(entrypoint: str) -> str:
    """Convert a file path to a Python module path.

    "multiply_rollout/workflow.py" → "multiply_rollout.workflow"
    "main.py"                      → "main"
    """
    return entrypoint.replace("/", ".").removesuffix(".py")


def _resolve_workflow(
    rollout: str,
    entrypoint: str,
) -> tuple[type, Any]:
    """Resolve an AgentWorkflow subclass and its config.

    Converts the entrypoint file path to a module, imports it,
    and auto-discovers an AgentWorkflow subclass and optional config.

    Returns (workflow_cls, config) where config may be None.
    """
    import importlib
    import sys

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    # Ensure cwd is on sys.path so local modules can be imported from CLI.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Add rollout directory to sys.path
    _ensure_rollout_on_path(rollout)

    module_name = _entrypoint_to_module(entrypoint)
    mod = importlib.import_module(module_name)

    # Auto-discover AgentWorkflow subclass
    workflow_cls = None
    for val in vars(mod).values():
        if (
            isinstance(val, type)
            and issubclass(val, AgentWorkflow)
            and val is not AgentWorkflow
        ):
            workflow_cls = val
            break

    if workflow_cls is None:
        raise CLIError(f"No AgentWorkflow subclass found in '{entrypoint}'")

    # Auto-discover config
    config = None
    for val in vars(mod).values():
        if isinstance(val, AgentWorkflowConfig):
            config = val
            break

    return workflow_cls, config


def load_workflow(
    rollout: str,
    entrypoint: str,
    quiet: bool = False,
    console: Console | None = None,
) -> tuple[type | None, Any, str | None]:
    """Load an AgentWorkflow class and its config.

    Returns (workflow_cls, workflow_config, error).
    """
    if console and not quiet:
        console.print(f"Loading workflow: {entrypoint}")

    try:
        workflow_cls, workflow_config = _resolve_workflow(
            rollout=rollout, entrypoint=entrypoint
        )
    except (CLIError, ImportError, ValueError, TypeError) as e:
        return None, None, str(e)

    if console and not quiet:
        console.print(f"  Workflow: {workflow_cls.__name__}")

    return workflow_cls, workflow_config, None


def auto_discover_grader(
    entrypoint: str,
) -> tuple[type | None, Any]:
    """Discover a Grader subclass and its config from the entrypoint module.

    The entrypoint file (e.g., ``local_rollout_server_example.py``) typically
    imports the Grader alongside the Workflow, so scanning its namespace is
    sufficient — no need to walk the entire package.

    Returns (grader_cls, grader_config) or (None, None) if not found.
    """
    import sys

    from osmosis_ai.rollout_v2.grader import Grader
    from osmosis_ai.rollout_v2.types import GraderConfig

    module_name = _entrypoint_to_module(entrypoint)
    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None

    grader_cls = None
    grader_config = None
    for val in vars(mod).values():
        if isinstance(val, type) and issubclass(val, Grader) and val is not Grader:
            grader_cls = val
        if isinstance(val, GraderConfig):
            grader_config = val

    return grader_cls, grader_config


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

    from osmosis_ai.rollout_v2.grader import Grader
    from osmosis_ai.rollout_v2.types import GraderConfig
    from osmosis_ai.rollout_v2.utils.imports import resolve_object

    if explicit_grader:
        grader_cls = resolve_object(explicit_grader)
        grader_config = resolve_object(explicit_config) if explicit_config else None
        return grader_cls, grader_config

    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None

    grader_cls = None
    grader_config = None
    for val in vars(mod).values():
        if isinstance(val, type) and issubclass(val, Grader) and val is not Grader:
            grader_cls = val
        if isinstance(val, GraderConfig):
            grader_config = val
    return grader_cls, grader_config


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

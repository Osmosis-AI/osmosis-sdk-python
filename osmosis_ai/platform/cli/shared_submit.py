"""Shared orchestration for ``osmosis train submit`` and ``osmosis eval submit``.

Both submit flows follow the same script:

1. resolve the git/workspace context and validate the config path,
2. load + validate the TOML config,
3. render a confirmation table and prompt the user,
4. POST to the platform API,
5. return an ``OperationResult`` describing the new run.

The only meaningful differences are the literal strings, the loader, and the
shape of the API call / next-step suggestions. ``CloudSubmitSpec`` parametrises
those, and ``run_cloud_submit`` is the single implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import SubmitRunResult
from osmosis_ai.platform.cli.shared_config import (
    BaseSubmitConfig,
    build_submit_summary_rows,
)
from osmosis_ai.platform.cli.utils import (
    print_remote_fetch_notice,
    require_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_rollout_backend,
    validate_workspace_directory_contract,
)


@dataclass(frozen=True)
class CloudSubmitSpec[ConfigT: BaseSubmitConfig]:
    """Describes the per-command differences between train / eval submit.

    Strings are plain literals. The four callables capture the rest:
    ``load_config`` builds ``ConfigT`` from the TOML path; ``validate_context``
    cross-checks the parsed config against the workspace directory layout;
    ``submit`` makes the API call (so each spec can pass its own set of
    config-section kwargs); ``build_next_steps`` produces the user-facing and
    structured suggestions shown alongside the success result.
    """

    config_dir: str
    """Workspace-relative directory the config must live under (e.g.
    ``configs/training``)."""

    command_label: str
    """Backticked command name used in path-related errors (e.g.
    ``"`osmosis train submit`"``)."""

    table_title: str
    """Title shown above the confirmation summary table."""

    confirm_prompt: str
    """Yes/no prompt shown before submitting."""

    status_message: str
    """Spinner status shown while the API call is in flight."""

    operation: str
    """``OperationResult.operation`` value (e.g. ``"train.submit"``)."""

    success_message_format: str
    """``str.format`` template with a ``{name}`` placeholder."""

    load_config: Callable[[Path], ConfigT]
    validate_context: Callable[[ConfigT, Path], None]
    submit: Callable[[OsmosisClient, ConfigT, Any, str], SubmitRunResult]
    build_next_steps: Callable[
        [SubmitRunResult, ConfigT],
        tuple[list[str], list[dict[str, Any]]],
    ]


def run_cloud_submit[ConfigT: BaseSubmitConfig](
    config_path: Path,
    *,
    yes: bool,
    spec: CloudSubmitSpec[ConfigT],
) -> OperationResult:
    """Run the shared submit flow for ``spec``."""
    context = require_git_workspace_directory_context()
    workspace_directory = context.workspace_directory
    validate_workspace_directory_contract(workspace_directory)

    config_path = Path(config_path)
    resolved_config_path = (
        config_path if config_path.is_absolute() else workspace_directory / config_path
    )
    ensure_workspace_directory_config_path(
        resolved_config_path,
        workspace_directory,
        config_dir=spec.config_dir,
        command_label=spec.command_label,
    )

    config = spec.load_config(resolved_config_path)
    spec.validate_context(config, workspace_directory)
    validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=spec.command_label,
    )

    summary_rows = build_submit_summary_rows(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        model=config.experiment_model_path,
        dataset=config.experiment_dataset,
        commit_sha=config.experiment_commit_sha,
        env=config.env,
        secrets=config.secrets,
    )
    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title=spec.table_title,
    )

    notes, warnings = print_remote_fetch_notice(
        workspace_directory,
        pinned_commit_sha=config.experiment_commit_sha,
    )

    require_confirmation(
        spec.confirm_prompt,
        yes=yes,
        summary=summary_rows,
        notes=notes,
        warnings=warnings,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status(spec.status_message):
        result = spec.submit(client, config, context.credentials, context.git_identity)

    display_next_steps, next_steps_structured = spec.build_next_steps(result, config)

    return OperationResult(
        operation=spec.operation,
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "model_name": config.experiment_model_path,
            "dataset_name": config.experiment_dataset,
            "created_at": result.created_at,
            **({"url": result.platform_url} if result.platform_url else {}),
            **git_result_context(context),
            "config": {
                "rollout": config.experiment_rollout,
                "entrypoint": config.experiment_entrypoint,
                "model": config.experiment_model_path,
                "dataset": config.experiment_dataset,
                "commit_sha": config.experiment_commit_sha,
            },
        },
        message=spec.success_message_format.format(name=result.name),
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
    )


__all__ = [
    "CloudSubmitSpec",
    "run_cloud_submit",
]

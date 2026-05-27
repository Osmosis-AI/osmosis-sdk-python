"""Handler for `osmosis train submit` (mirror of platform/cli/eval.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console


def submit(config_path: Path, *, yes: bool) -> Any:
    """Submit a new training run."""
    from osmosis_ai.cli.output import OperationResult, get_output_context
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.training_config import (
        load_training_config,
        validate_training_context_paths,
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

    command_label = "`osmosis train submit`"

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
        config_dir="configs/training",
        command_label=command_label,
    )
    config = load_training_config(resolved_config_path)
    validate_training_context_paths(config, workspace_directory)
    validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=command_label,
    )
    credentials = context.credentials

    summary_rows: list[tuple[str, str]] = [
        ("Rollout", config.experiment_rollout),
        ("Entrypoint", config.experiment_entrypoint),
        ("Model", config.experiment_model_path),
        ("Dataset", config.experiment_dataset),
    ]
    if config.experiment_commit_sha:
        summary_rows.append(("Commit", config.experiment_commit_sha))
    if config.env:
        env_keys = ", ".join(sorted(config.env))
        summary_rows.append((f"Rollout env ({len(config.env)})", env_keys))
    if config.secrets:
        secret_summary = ", ".join(
            f"{env_name}={secret_name}"
            for env_name, secret_name in sorted(config.secrets.items())
        )
        summary_rows.append(
            (
                f"Rollout secrets ({len(config.secrets)})",
                secret_summary,
            )
        )
    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title="Training Run",
    )

    notes, warnings = print_remote_fetch_notice(
        workspace_directory,
        pinned_commit_sha=config.experiment_commit_sha,
    )

    require_confirmation(
        "Submit this training run?",
        yes=yes,
        summary=summary_rows,
        notes=notes,
        warnings=warnings,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status("Submitting training run..."):
        result = client.submit_training_run(
            experiment_config=config.experiment_config,
            training_config=config.training_config or None,
            sampling_config=config.sampling_config or None,
            checkpoints_config=config.checkpoints_config or None,
            advanced_config=config.advanced_config or None,
            env_config=config.env or None,
            secret_refs_config=config.secrets or None,
            credentials=credentials,
            git_identity=context.git_identity,
        )

    return OperationResult(
        operation="train.submit",
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
        message=f"Training run submitted: {result.name}",
        display_next_steps=[
            f"Status: {result.status}",
            f"Model: {config.experiment_model_path}",
            f"Dataset: {config.experiment_dataset}",
            (
                f"View: {result.platform_url}"
                if result.platform_url
                else f"Check status with: osmosis train info {result.name}"
            ),
        ],
        next_steps_structured=[
            {"action": "train_info", "name": result.name},
            *(
                [{"action": "open_url", "url": result.platform_url}]
                if result.platform_url
                else []
            ),
        ],
    )


__all__ = ["submit"]

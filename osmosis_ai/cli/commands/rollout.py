"""Rollout commands: validate, list."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from osmosis_ai.cli.console import Console

app: typer.Typer = typer.Typer(
    help="Manage rollouts (validate, list).",
    no_args_is_help=True,
)


def _resolve_validation_target(
    config_path: Path,
) -> tuple[str, Path, str, str, str | None, str | None]:
    from osmosis_ai.eval.config import load_eval_config
    from osmosis_ai.platform.cli.training_config import load_training_config
    from osmosis_ai.platform.cli.workspace_contract import (
        ensure_workspace_config_path,
        resolve_workspace_root,
        validate_workspace_contract,
    )

    workspace_root = resolve_workspace_root(config_path)
    validate_workspace_contract(workspace_root)

    resolved_path = config_path.resolve()
    training_dir = (workspace_root / "configs" / "training").resolve()
    eval_dir = (workspace_root / "configs" / "eval").resolve()

    try:
        resolved_path.relative_to(training_dir)
    except ValueError:
        pass
    else:
        ensure_workspace_config_path(
            config_path,
            workspace_root,
            config_dir="configs/training",
            command_label="`osmosis rollout validate`",
        )
        config = load_training_config(config_path)
        return (
            "training",
            workspace_root,
            config.experiment_rollout,
            config.experiment_entrypoint,
            None,
            None,
        )

    try:
        resolved_path.relative_to(eval_dir)
    except ValueError:
        pass
    else:
        ensure_workspace_config_path(
            config_path,
            workspace_root,
            config_dir="configs/eval",
            command_label="`osmosis rollout validate`",
        )
        config = load_eval_config(config_path)
        return (
            "eval",
            workspace_root,
            config.eval_rollout,
            config.eval_entrypoint,
            config.grader_module,
            config.grader_config,
        )

    raise CLIError(
        "`osmosis rollout validate` only accepts configs under "
        "`configs/training/` or `configs/eval/`."
    )


def _validate_rollout_config(
    *, config_path: Path, console: Console | None = None
) -> Any:
    from osmosis_ai.cli.output import DetailField, DetailResult
    from osmosis_ai.platform.cli.workspace_contract import validate_rollout_backend

    (
        config_kind,
        workspace_root,
        rollout,
        entrypoint,
        grader_module,
        grader_config_ref,
    ) = _resolve_validation_target(config_path)

    validate_rollout_backend(
        workspace_root=workspace_root,
        rollout=rollout,
        entrypoint=entrypoint,
        command_label="`osmosis rollout validate`",
        grader_module=grader_module,
        grader_config_ref=grader_config_ref,
    )

    rows: list[tuple[str, Any]] = [
        ("Config", str(config_path.resolve())),
        ("Kind", config_kind),
        ("Rollout", rollout),
        ("Entrypoint", entrypoint),
    ]
    if grader_module:
        rows.append(("Grader override", grader_module))
    rows.append(("Status", "Validation passed."))

    return DetailResult(
        title="Rollout Validation",
        data={
            "config": str(config_path.resolve()),
            "kind": config_kind,
            "rollout": rollout,
            "entrypoint": entrypoint,
            "grader_module": grader_module,
            "grader_config": grader_config_ref,
            "valid": True,
        },
        fields=[
            DetailField(label=str(label), value=str(value)) for label, value in rows
        ],
    )


@app.command("validate")
def validate(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to training or eval TOML config file.",
    ),
) -> Any:
    """Validate a rollout entrypoint referenced by a config file."""
    return _validate_rollout_config(config_path=config_path)


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> Any:
    """List rollouts in the current workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_rollout,
    )
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        fetch_all_pages,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching rollouts..."):
        if fetch_all:
            rollouts, total_count = fetch_all_pages(
                lambda lim, off: client.list_rollouts(
                    limit=lim, offset=off, credentials=credentials
                ),
                items_attr="rollouts",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_rollouts(
                limit=effective_limit, offset=0, credentials=credentials
            )
            rollouts = page.rollouts
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Rollouts",
        items=[serialize_rollout(rollout) for rollout in rollouts],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        columns=[
            ListColumn(key="name", label="Name"),
            ListColumn(key="is_active", label="Active"),
            ListColumn(key="repo_full_name", label="Repository"),
            ListColumn(key="last_synced_commit_sha", label="Commit"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )

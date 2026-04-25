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


def _validate_rollout_config(*, config_path: Path, console: Console) -> None:
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
        ("Config", console.format_text(config_path.resolve())),
        ("Kind", console.format_text(config_kind)),
        ("Rollout", console.format_text(rollout)),
        ("Entrypoint", console.format_text(entrypoint)),
    ]
    if grader_module:
        rows.append(("Grader override", console.format_text(grader_module)))

    console.table(rows, title="Rollout Validation")
    console.print("Validation passed.", style="green")


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
) -> None:
    """Validate a rollout entrypoint referenced by a config file."""
    from osmosis_ai.cli.console import Console

    _validate_rollout_config(config_path=config_path, console=Console())


def _format_commit(r: Any, console: Console) -> Any:
    """Format commit SHA as a clickable terminal hyperlink (OSC 8 via Rich)."""
    sha = r.last_synced_commit_sha
    if not sha:
        return console.format_text("—", style="dim")
    short = sha[:7]
    if r.repo_full_name:
        url = f"https://github.com/{r.repo_full_name}/tree/{sha}"
        return console.format_url(url, label=short, style="dim")
    return console.format_text(short, style="dim")


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> None:
    """List rollouts in the current workspace."""
    from osmosis_ai.cli.console import Console
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        format_date,
        paginated_fetch,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    console = Console()
    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    with console.spinner("Fetching rollouts..."):
        client = OsmosisClient()
        rollouts, total_count, _has_more = paginated_fetch(
            lambda lim, off: client.list_rollouts(
                limit=lim, offset=off, credentials=credentials
            ),
            items_attr="rollouts",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    if not rollouts:
        console.print("No rollouts found.")
        return

    console.print(f"Rollouts ({total_count}):", style="bold")
    for r in rollouts:
        name = console.format_text(r.name)
        active = (
            console.format_text("[active]", style="green")
            if r.is_active
            else console.format_text("[inactive]", style="dim")
        )
        commit = _format_commit(r, console)
        date = console.format_text(format_date(r.created_at), style="dim")
        console.print(
            console.format_text("  ")
            + name
            + console.format_text("  ")
            + active
            + console.format_text("  ")
            + commit
            + console.format_text("  ")
            + date
        )

    print_pagination_footer(len(rollouts), total_count, "rollouts")

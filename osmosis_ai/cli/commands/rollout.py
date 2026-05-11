"""Rollout commands: validate, list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage rollouts (validate, list).",
    no_args_is_help=True,
)


def _workspace_result_context(workspace: Any) -> dict[str, Any]:
    return {
        "workspace": {"id": workspace.workspace_id, "name": workspace.workspace_name},
        "project_root": str(workspace.project_root),
    }


def _resolve_validation_target(
    config_path: Path,
) -> tuple[str, Path, str, str]:
    from osmosis_ai.eval.config import load_eval_config
    from osmosis_ai.platform.cli.project_contract import (
        ensure_project_config_path,
        resolve_project_root_from_cwd,
        validate_project_contract,
    )
    from osmosis_ai.platform.cli.training_config import load_training_config

    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)

    resolved_path = config_path.resolve()
    training_dir = (project_root / "configs" / "training").resolve()
    eval_dir = (project_root / "configs" / "eval").resolve()

    try:
        resolved_path.relative_to(training_dir)
    except ValueError:
        pass
    else:
        ensure_project_config_path(
            config_path,
            project_root,
            config_dir="configs/training",
            command_label="`osmosis rollout validate`",
        )
        config = load_training_config(config_path)
        return (
            "training",
            project_root,
            config.experiment_rollout,
            config.experiment_entrypoint,
        )

    try:
        resolved_path.relative_to(eval_dir)
    except ValueError:
        pass
    else:
        ensure_project_config_path(
            config_path,
            project_root,
            config_dir="configs/eval",
            command_label="`osmosis rollout validate`",
        )
        config = load_eval_config(config_path)
        return (
            "eval",
            project_root,
            config.eval_rollout,
            config.eval_entrypoint,
        )

    raise CLIError(
        "`osmosis rollout validate` only accepts configs under "
        "`configs/training/` or `configs/eval/`."
    )


def _validate_eval_static_entrypoint(
    *, project_root: Path, rollout: str, entrypoint: str
) -> None:
    from osmosis_ai.eval.common.cli import _resolve_rollout_entrypoint

    rollout_dir, entrypoint_path = _resolve_rollout_entrypoint(
        rollout,
        entrypoint,
        project_root=project_root,
    )
    pyproject = rollout_dir / "pyproject.toml"
    if not pyproject.is_file():
        raise CLIError(
            f"rollouts/{rollout}/pyproject.toml is required for eval server validation."
        )
    if entrypoint_path.suffix != ".py":
        raise CLIError("Eval entrypoint must be a Python script.")


async def _validate_eval_server_entrypoint(
    *,
    project_root: Path,
    rollout: str,
    entrypoint: str,
) -> None:
    import uuid

    from osmosis_ai.eval.controller.locks import (
        FixedPortLock,
        assert_user_server_port_free,
    )
    from osmosis_ai.eval.controller.process import (
        start_user_server_process,
        wait_for_user_server_health,
    )

    fixed_port_lock = FixedPortLock()
    try:
        fixed_port_lock.acquire()
    except Exception as exc:
        raise CLIError(f"Eval server validation failed: {exc}") from exc
    try:
        assert_user_server_port_free()
        rollout_dir = project_root / "rollouts" / rollout
        try:
            process = await start_user_server_process(
                project_root=project_root,
                rollout_dir=rollout_dir,
                rollout_name=rollout,
                entrypoint=entrypoint,
                invocation_id=f"validate-{uuid.uuid4().hex[:8]}",
                log_dir=project_root / ".osmosis" / "logs" / "rollout-validate",
            )
        except Exception as exc:
            raise CLIError(f"Eval server validation failed: {exc}") from exc
        try:
            try:
                await wait_for_user_server_health(timeout_sec=30.0, process=process)
            except Exception as exc:
                raise CLIError(f"Eval server health check failed: {exc}") from exc
        finally:
            await process.terminate()
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(f"Eval server validation failed: {exc}") from exc
    finally:
        fixed_port_lock.release()


def _validate_rollout_config(*, config_path: Path, server: bool = False) -> Any:
    from osmosis_ai.cli.output import DetailField, DetailResult
    from osmosis_ai.platform.cli.project_contract import validate_rollout_backend

    config_kind, project_root, rollout, entrypoint = _resolve_validation_target(
        config_path
    )

    if config_kind == "training":
        if server:
            raise CLIError("--server is only supported for eval configs.")
        validate_rollout_backend(
            project_root=project_root,
            rollout=rollout,
            entrypoint=entrypoint,
            command_label="`osmosis rollout validate`",
        )
        status = "Validation passed."
    else:
        _validate_eval_static_entrypoint(
            project_root=project_root,
            rollout=rollout,
            entrypoint=entrypoint,
        )
        if server:
            import asyncio

            asyncio.run(
                _validate_eval_server_entrypoint(
                    project_root=project_root,
                    rollout=rollout,
                    entrypoint=entrypoint,
                )
            )
            status = (
                "Validation passed. Dynamic eval run server startup and /health "
                "checks passed."
            )
        else:
            status = (
                "Validation passed. Static server entrypoint checks passed; use "
                "`osmosis rollout validate --server ...` to verify startup and /health."
            )

    rows: list[tuple[str, Any]] = [
        ("Config", str(config_path.resolve())),
        ("Kind", config_kind),
        ("Rollout", rollout),
        ("Entrypoint", entrypoint),
        ("Status", status),
    ]

    return DetailResult(
        title="Rollout Validation",
        data={
            "config": str(config_path.resolve()),
            "kind": config_kind,
            "rollout": rollout,
            "entrypoint": entrypoint,
            "server": server,
            "status": status,
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
    server: bool = typer.Option(
        False,
        "--server",
        help="Start the eval server entrypoint and require GET /health.",
    ),
) -> Any:
    """Validate a rollout entrypoint referenced by a config file."""
    return _validate_rollout_config(config_path=config_path, server=server)


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> Any:
    """List rollouts in the linked project workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_rollout,
    )
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        require_workspace_context,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching rollouts..."):
        if fetch_all:
            rollouts, total_count = fetch_all_pages(
                lambda lim, off: client.list_rollouts(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    workspace_id=workspace_id,
                ),
                items_attr="rollouts",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_rollouts(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                workspace_id=workspace_id,
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
        extra=_workspace_result_context(workspace),
        columns=[
            ListColumn(key="name", label="Name"),
            ListColumn(key="is_active", label="Active"),
            ListColumn(key="repo_full_name", label="Repository"),
            ListColumn(key="last_synced_commit_sha", label="Commit"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )

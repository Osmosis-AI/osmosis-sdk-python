"""Deployment management commands (LoRA adapters for inference)."""

from __future__ import annotations

import re
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage inference deployments (create, list, status, delete, rename).",
    no_args_is_help=True,
)


def _print_deployment_section(
    deployments: list[Any],
    total_count: int,
) -> None:
    """Print a list of deployments with consistent formatting."""
    from osmosis_ai.platform.cli.utils import entity_status_style, format_dim_date

    if not deployments:
        return
    console.print(f"Deployments ({total_count}):", style="bold")
    for d in deployments:
        style = entity_status_style(d.status) or "dim"
        status_str = console.format_styled(f"[{d.status}]", style)
        name = console.escape(d.lora_name)
        base = console.format_styled(d.base_model, "dim") if d.base_model else ""
        run = (
            console.format_styled(f"from {d.training_run_name}", "dim")
            if d.training_run_name
            else ""
        )
        step = console.format_styled(f"step:{d.checkpoint_step}", "dim")
        date = format_dim_date(d.created_at)
        console.print(
            f"  {name}  {status_str}  {base}  {run}  {step}  {date}",
            highlight=False,
        )
    console.print()


def _fetch_all_deployments(client: Any, credentials: Any) -> list[Any]:
    """Fetch all deployments via exhaustive pagination."""
    from osmosis_ai.platform.cli.utils import fetch_all_pages

    deployments, _ = fetch_all_pages(
        lambda lim, off: client.list_deployments(
            limit=lim, offset=off, credentials=credentials
        ),
        items_attr="deployments",
    )
    return deployments


def _default_lora_name(run_name: str, step: int) -> str:
    """Mirror of monolith src/lib/slug-generator.ts::generateLoraName.

    Used for the confirmation preview only. Server computes the
    authoritative name if none is passed; in the rare event of drift,
    the preview may disagree with what the server stores, but the
    server response is what the CLI ultimately renders.
    """
    raw = f"{run_name}-step-{step}-lora"
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", raw)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug[:255]


@app.command("create")
def create(
    training_run: str = typer.Argument(..., help="Training run name (or UUID)."),
    step: int | None = typer.Option(
        None,
        "--step",
        "-s",
        help="Checkpoint step. Defaults to the latest uploaded checkpoint.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help=(
            "LoRA adapter name. 1-255 chars, [a-zA-Z0-9_-]+. "
            "Defaults to '<run>-step-<N>-lora'."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Deploy a LoRA checkpoint as an inference adapter."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        platform_entity_url,
    )

    ws_name, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner("Fetching checkpoints..."):
        ckpts = client.list_training_run_checkpoints(
            training_run, credentials=credentials
        )

    if not ckpts.checkpoints:
        raise CLIError(
            f"No uploaded checkpoints found for '{training_run}'.\n"
            "Wait for training to finish, or check "
            "'osmosis train status'."
        )

    # Defensive sort so auto-pick always returns the highest step
    sorted_cps = sorted(
        ckpts.checkpoints, key=lambda c: c.checkpoint_step, reverse=True
    )

    if step is None:
        selected = sorted_cps[0]
    else:
        selected = next((c for c in sorted_cps if c.checkpoint_step == step), None)
        if selected is None:
            available = ", ".join(str(c.checkpoint_step) for c in sorted_cps)
            raise CLIError(
                f"No uploaded checkpoint at step {step}. Available steps: {available}."
            )

    run_display = ckpts.training_run_name or training_run
    preview_name = name or _default_lora_name(run_display, selected.checkpoint_step)

    rows: list[tuple[str, str]] = [
        ("Training run", console.escape(run_display)),
        ("Checkpoint", f"step {selected.checkpoint_step}"),
        ("LoRA name", console.escape(preview_name)),
    ]
    console.table(rows, title="Deploy")
    require_confirmation("Deploy this checkpoint?", yes=yes)

    with console.spinner("Deploying..."):
        result = client.create_deployment(
            training_run=training_run,
            checkpoint_step=selected.checkpoint_step,
            lora_name=name,
            credentials=credentials,
        )

    url = platform_entity_url(ws_name, "deployments")
    console.print(
        f"Deployment created: {console.escape(result.lora_name)}", style="green"
    )
    console.print(f"  Status: {result.status}")
    console.print(f"  View: {url}")


@app.command("list")
def list_deployments(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of deployments to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all deployments."),
) -> None:
    """List deployments in the current workspace."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner("Fetching deployments..."):
        if fetch_all:
            deployments = _fetch_all_deployments(client, credentials)
            total = len(deployments)
        else:
            result = client.list_deployments(
                limit=effective_limit, credentials=credentials
            )
            deployments = result.deployments
            total = result.total_count

    if not deployments:
        console.print("No deployments found.")
        return

    _print_deployment_section(deployments, total)

    if not fetch_all:
        print_pagination_footer(len(deployments), total, "deployments")


@app.command("status")
def status(
    name: str = typer.Argument(..., help="Deployment name (LoRA name) or UUID."),
) -> None:
    """Show deployment details."""
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        format_date,
    )

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()

    with console.spinner("Fetching deployment..."):
        d = client.get_deployment(name, credentials=credentials)

    rows: list[tuple[str, str]] = [
        ("LoRA name", console.escape(d.lora_name)),
        ("ID", d.id),
        ("Status", d.status),
        ("Base model", console.escape(d.base_model) if d.base_model else "—"),
        ("Checkpoint", f"step {d.checkpoint_step}"),
    ]
    if d.training_run_name:
        rows.append(("Training run", console.escape(d.training_run_name)))
    elif d.training_run_id:
        rows.append(("Training run", d.training_run_id))
    if d.creator_name:
        rows.append(("Creator", console.escape(d.creator_name)))
    if d.created_at:
        rows.append(("Created", format_date(d.created_at)))

    console.table(rows, title="Deployment")


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="LoRA name or UUID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a deployment (remove the LoRA adapter from inference)."""
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()

    require_confirmation(
        f'Delete deployment "{name}"? The LoRA adapter will be removed from inference.',
        yes=yes,
    )
    client.delete_deployment(name, credentials=credentials)
    console.print(f'Deployment "{name}" deleted.', style="green")


@app.command("rename")
def rename(
    name: str = typer.Argument(..., help="Current LoRA name or UUID."),
    new_name: str = typer.Argument(..., help="New LoRA name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Rename a deployment's LoRA adapter name."""
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.cli.utils import _require_auth

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()

    require_confirmation(f'Rename "{name}" to "{new_name}"?', yes=yes)

    with console.spinner("Renaming..."):
        result = client.rename_deployment(name, new_name, credentials=credentials)

    console.print(
        f'Renamed: "{name}" → "{console.escape(result.lora_name)}"',
        style="green",
    )

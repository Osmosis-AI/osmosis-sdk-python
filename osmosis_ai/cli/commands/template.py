"""``osmosis template`` command shell.

Thin Typer wrapper that delegates to :mod:`osmosis_ai.templates.cli`. Heavy
imports stay inside the command bodies per the CLI lazy-loading contract.
"""

from __future__ import annotations

from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Add starter templates into your Osmosis workspace directory.",
    no_args_is_help=True,
)


def _complete_template_names(ctx: Any, args: list[str], incomplete: str) -> list[str]:
    del ctx, args
    from osmosis_ai.templates.registry import list_templates as registry_list_templates

    return [name for name in registry_list_templates() if name.startswith(incomplete)]


@app.command("list")
def list_templates() -> Any:
    """List starter templates."""
    from osmosis_ai.templates.cli import list_command

    return list_command()


@app.command("apply")
def apply(
    name: str = typer.Argument(
        ...,
        help="Template name (see 'osmosis template list').",
        autocompletion=_complete_template_names,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help=(
            "Overwrite existing rollouts/<name>/ directory and config files. "
            "Without --force, the command refuses to clobber existing files."
        ),
    ),
) -> Any:
    """Copy a starter template into the workspace directory's canonical layout.

    Files land at ``rollouts/<name>/`` and ``configs/{training,eval}/<name>.toml``
    so the rollout is immediately runnable.
    """
    from osmosis_ai.templates.cli import apply_command

    return apply_command(name=name, force=force)

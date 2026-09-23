"""Build the images declared by a connected job repository."""

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Build Harbor task images from a Git repository.", no_args_is_help=True
)


@app.command("build")
def build(
    repository: str | None = typer.Option(
        None, "--repo", help="Legacy bundle build from a connected GitHub repository."
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="GitHub task source; publish deterministic images without a task bundle.",
    ),
    ref: str = typer.Option(
        "HEAD",
        "--ref",
        help="Branch, tag, or commit; defaults to the repository's default branch.",
    ),
    tasks_dir: str = typer.Option(
        "tasks",
        "--path",
        "--tasks-dir",
        help="Task directory or dataset.toml relative to the repository root.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write verified task-to-image JSON here after the build completes.",
    ),
    output_dir: Path = typer.Option(
        Path(".osmosis/images"),
        "--output-dir",
        help="Request state and verified build artifacts; reuse to resume.",
    ),
    request_id: str | None = typer.Option(
        None,
        "--request-id",
        help="Optional idempotency UUID; generated and saved automatically.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for completion and collect the verified build artifacts.",
    ),
    timeout: float = typer.Option(
        10800,
        "--timeout",
        min=1,
        help="Local wait limit in seconds; remote builds continue afterward.",
    ),
) -> CommandResult:
    """Build Harbor environments remotely and return their task-to-image mapping."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.cli.images import build as run

    if bool(url) == bool(repository):
        raise CLIError("Specify exactly one of --url or --repo", code="VALIDATION")
    options: dict[str, Any] = {}
    if url:
        options["image_layout"] = "source-v1"
    if output:
        options["output"] = output
    return run(
        repository=url or repository or "",
        ref=ref,
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        request_id=request_id,
        wait=wait,
        timeout=timeout,
        **options,
    )


@app.command("info")
def info(
    request_id: str = typer.Argument(..., help="Image build request UUID."),
    repository: str = typer.Option(
        ..., "--url", "--repo", help="Connected GitHub repository URL."
    ),
) -> CommandResult:
    """Show the pinned commit, task count, and image progress."""
    from osmosis_ai.platform.cli.images import info as run

    return run(repository=repository, request_id=request_id)

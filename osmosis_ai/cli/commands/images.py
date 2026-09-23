"""Build the images declared by a connected job repository."""

from pathlib import Path

import typer

from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Build Harbor task images from a Git repository.", no_args_is_help=True
)


@app.command("build")
def build(
    repository: str = typer.Option(
        ..., "--repo", help="Connected GitHub repository URL (HTTPS or SSH)."
    ),
    ref: str = typer.Option(
        "HEAD",
        "--ref",
        help="Branch, tag, or commit; defaults to the repository's default branch.",
    ),
    tasks_dir: str = typer.Option(
        "tasks", "--tasks-dir", help="Task directory relative to the repository root."
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
        help="Wait for completion and download the prepared tasks.",
    ),
    timeout: float = typer.Option(
        10800,
        "--timeout",
        min=1,
        help="Local wait limit in seconds; remote builds continue afterward.",
    ),
) -> CommandResult:
    """Discover all tasks, build their images remotely, and download the task bundle."""
    from osmosis_ai.platform.cli.images import build as run

    return run(
        repository=repository,
        ref=ref,
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        request_id=request_id,
        wait=wait,
        timeout=timeout,
    )


@app.command("info")
def info(
    request_id: str = typer.Argument(..., help="Image build request UUID."),
    repository: str = typer.Option(
        ..., "--repo", help="Connected GitHub repository URL."
    ),
) -> CommandResult:
    """Show the pinned commit, task count, and image progress."""
    from osmosis_ai.platform.cli.images import info as run

    return run(repository=repository, request_id=request_id)

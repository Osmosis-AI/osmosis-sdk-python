"""Harbor dataset and image commands."""

from __future__ import annotations

from asyncio import run as run_async

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import CommandResult, OperationResult

app: typer.Typer = typer.Typer(
    help="Build and manage Harbor task environments.",
    no_args_is_help=True,
)


@app.command("prebuild")
def prebuild(
    dataset: str = typer.Argument(
        ...,
        help="Local folder, Harbor dataset reference, or Git dataset URL.",
        metavar="DATASET",
    ),
    image_repository: str = typer.Option(
        ...,
        "--image-repository",
        help=(
            "Image repository used by Harbor serve. Each distinct environment "
            "is published as <repository>:<Harbor environment ID>."
        ),
    ),
    build_system: str = typer.Option(
        "buildx",
        "--build-system",
        help="Build executor: buildx or google-cloud-build.",
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        help="Target container platform; it must match the Harbor serve runtime.",
    ),
    builder: str | None = typer.Option(
        None,
        "--builder",
        help="Buildx builder name; only valid with --build-system buildx.",
    ),
    gcp_project: str | None = typer.Option(
        None,
        "--gcp-project",
        help="Google Cloud project; required for google-cloud-build.",
    ),
    gcp_region: str | None = typer.Option(
        None,
        "--gcp-region",
        help="Google Cloud Build region; required for google-cloud-build.",
    ),
) -> CommandResult:
    """Build each distinct task environment and publish it to a registry."""
    try:
        from osmosis_ai.rollout.backend.harbor.images import build_and_publish
    except ModuleNotFoundError as exc:
        raise CLIError(
            "Harbor prebuild requires the Harbor dependencies. Install "
            "`osmosis-ai[harbor]`.",
            code="VALIDATION",
        ) from exc

    try:
        result = run_async(
            build_and_publish(
                dataset,
                image_repository=image_repository,
                build_system=build_system,
                platform=platform,
                buildx_builder=builder,
                gcp_project=gcp_project,
                gcp_region=gcp_region,
            )
        )
    except CLIError:
        raise
    except ValueError as exc:
        raise CLIError(str(exc), code="VALIDATION") from exc
    except RuntimeError as exc:
        raise CLIError(str(exc), code="PLATFORM_ERROR") from exc

    images = [
        {
            "image": environment.image.image,
            "immutable_image": environment.image.immutable_image,
            "digest": environment.image.digest,
            "content_hash": environment.content_hash,
        }
        for environment in result.environments
    ]
    image_label = "image" if len(images) == 1 else "images"
    task_label = "task" if result.task_count == 1 else "tasks"
    return OperationResult(
        operation="harbor.prebuild",
        status="success",
        resource={
            "dataset": result.dataset,
            "dataset_path": str(result.dataset_path),
            "task_count": result.task_count,
            "environment_count": len(result.environments),
            "image_count": len(images),
            "images": images,
        },
        message=(
            f"Published {len(images)} Harbor {image_label} for "
            f"{result.task_count} {task_label}."
        ),
        display_next_steps=[image["image"] for image in images],
    )


__all__ = ["app", "prebuild"]

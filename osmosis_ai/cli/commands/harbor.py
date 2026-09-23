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


def _parse_build_args(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        key, separator, argument = value.partition("=")
        if not separator or not key:
            raise CLIError(
                f"Invalid --build-arg {value!r}; expected KEY=VALUE.",
                code="VALIDATION",
            )
        if key in parsed:
            raise CLIError(
                f"Duplicate --build-arg key {key!r}.",
                code="VALIDATION",
            )
        parsed[key] = argument
    return parsed


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
        help="Registry repository that receives the content-addressed images.",
    ),
    build_system: str = typer.Option(
        "buildx",
        "--build-system",
        help="Build executor: buildx or google-cloud-build.",
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        help="Target container platform included in the image hash.",
    ),
    build_args: list[str] = typer.Option(
        None,
        "--build-arg",
        help="Docker build argument as KEY=VALUE; repeat for multiple arguments.",
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
                build_args=_parse_build_args(build_args),
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
            "task_count": len(environment.task_names),
            "tasks": list(environment.task_names),
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
            "environment_count": len(images),
            "images": images,
        },
        message=(
            f"Published {len(images)} Harbor {image_label} for "
            f"{result.task_count} {task_label}."
        ),
        display_next_steps=[image["immutable_image"] for image in images],
    )


__all__ = ["app", "prebuild"]

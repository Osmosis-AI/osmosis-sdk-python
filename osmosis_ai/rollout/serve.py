"""Construct and serve a rollout backend from TOML configuration."""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from osmosis_ai.cli.errors import CLIError


class _ServeConfigModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class SimpleBackendConfig(_ServeConfigModel):
    """Configuration for the in-process rollout backend."""

    workflow: str
    grader: str | None = None
    workflow_config: str | None = None
    grader_config: str | None = None


class HarborBackendConfig(_ServeConfigModel):
    """Configuration for the Harbor rollout backend."""

    tasks_dir: Path = Path("tasks")
    task_mode: Literal["template", "dataset"] = "template"
    agent: str
    grader: str | None = None
    workflow_config: str | None = None
    grader_config: str | None = None
    environment: str = "docker"
    environment_kwargs: dict[str, Any] | None = None
    native_agent_kwargs: dict[str, Any] | None = None
    concurrency: int = Field(default=4, ge=1)
    native_model_name: str = "openai/osmosis-rollout"
    trials_dir: Path | None = None
    cleanup_successful_trials: bool = True
    patch_dockerfile_with_sdk: bool | None = None
    agent_setup_timeout_sec: float | None = Field(default=None, ge=0)
    max_queue_depth: int | None = Field(default=None, ge=1)


class _SimpleServeConfig(_ServeConfigModel):
    backend: Literal["simple"]
    simple: SimpleBackendConfig


class _HarborServeConfig(_ServeConfigModel):
    backend: Literal["harbor"]
    harbor: HarborBackendConfig


RolloutServeConfig = Annotated[
    _SimpleServeConfig | _HarborServeConfig,
    Field(discriminator="backend"),
]
_CONFIG_ADAPTER = TypeAdapter(RolloutServeConfig)


def _validation_message(path: Path, exc: ValidationError) -> str:
    issues: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        label = location or "config"
        issues.append(f"  - {label}: {error['msg']}")
    return f"Invalid rollout server config {path}:\n" + "\n".join(issues)


def load_serve_config(path: Path) -> tuple[RolloutServeConfig, Path]:
    """Load a rollout server config and return it with its rollout directory."""
    resolved = path.expanduser().resolve()
    try:
        with resolved.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError:
        raise CLIError(
            f"Rollout server config not found: {resolved}", code="NOT_FOUND"
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise CLIError(
            f"Invalid TOML in rollout server config {resolved}: {exc}",
            code="VALIDATION",
        ) from exc
    except OSError as exc:
        raise CLIError(
            f"Cannot read rollout server config {resolved}: {exc}",
            code="VALIDATION",
        ) from exc

    try:
        config = _CONFIG_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise CLIError(_validation_message(resolved, exc), code="VALIDATION") from exc
    return config, resolved.parent


def _resolve_path(path: Path | None, rollout_dir: Path) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (rollout_dir / path).resolve()


def _required_directory(path: Path, *, label: str) -> Path:
    if not path.is_dir():
        raise CLIError(f"{label} does not exist: {path}", code="NOT_FOUND")
    return path


def _server_port(port: int | None) -> int:
    if port is not None:
        return port
    raw = os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise CLIError(
            f"_OSMOSIS_ROLLOUT_PORT must be an integer, got {raw!r}.",
            code="VALIDATION",
        ) from exc
    if not 1 <= resolved <= 65535:
        raise CLIError(
            f"_OSMOSIS_ROLLOUT_PORT must be between 1 and 65535, got {resolved}.",
            code="VALIDATION",
        )
    return resolved


@contextmanager
def _rollout_context(rollout_dir: Path) -> Iterator[None]:
    """Resolve relative paths and imports from the selected rollout directory."""
    previous_cwd = Path.cwd()
    previous_sys_path = sys.path.copy()
    rollout_path = str(rollout_dir)
    if not sys.path or sys.path[0] != rollout_path:
        sys.path.insert(0, rollout_path)
    os.chdir(rollout_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_sys_path


def _run_server(*, backend: Any, host: str, port: int | None) -> None:
    if not host.strip():
        raise CLIError("--host must be non-empty.", code="VALIDATION")

    try:
        import uvicorn

        from osmosis_ai.rollout.server import create_rollout_server
    except ModuleNotFoundError as exc:
        raise CLIError(
            "Serving rollouts requires the server dependencies. Install "
            "`osmosis-ai[server]`.",
            code="VALIDATION",
        ) from exc

    app = create_rollout_server(backend=backend)
    uvicorn.run(app, host=host, port=_server_port(port))


def _serve_simple(
    config: SimpleBackendConfig,
    *,
    rollout_dir: Path,
    host: str,
    port: int | None,
) -> None:
    with _rollout_context(rollout_dir):
        from osmosis_ai.rollout.backend.local import LocalBackend

        try:
            backend = LocalBackend(
                workflow=config.workflow,
                grader=config.grader,
                workflow_config=config.workflow_config,
                grader_config=config.grader_config,
            )
        except (ImportError, ValueError) as exc:
            raise CLIError(
                f"Could not configure the simple rollout backend: {exc}",
                code="VALIDATION",
            ) from exc
        _run_server(backend=backend, host=host, port=port)


def _serve_harbor(
    config: HarborBackendConfig,
    *,
    rollout_dir: Path,
    host: str,
    port: int | None,
) -> None:
    resolved_tasks_dir = _resolve_path(config.tasks_dir, rollout_dir)
    assert resolved_tasks_dir is not None
    _required_directory(resolved_tasks_dir, label="Harbor tasks directory")
    resolved_trials_dir = _resolve_path(config.trials_dir, rollout_dir)

    with _rollout_context(rollout_dir):
        try:
            from harbor.models.environment_type import EnvironmentType
            from harbor.models.trial.config import EnvironmentConfig
            from harbor.trial.queue import TrialQueue

            from osmosis_ai.rollout.backend.harbor import HarborBackend
        except ModuleNotFoundError as exc:
            raise CLIError(
                "Serving with Harbor requires the Harbor dependencies. Install "
                "`osmosis-ai[server,harbor]`.",
                code="VALIDATION",
            ) from exc

        try:
            environment_type = EnvironmentType(config.environment)
        except ValueError as exc:
            choices = ", ".join(item.value for item in EnvironmentType)
            raise CLIError(
                f"Unknown Harbor environment {config.environment!r}. Choose one of: "
                f"{choices}.",
                code="VALIDATION",
            ) from exc

        try:
            backend = HarborBackend(
                orchestrator=TrialQueue(n_concurrent=config.concurrency),
                tasks_dir=resolved_tasks_dir,
                task_mode=config.task_mode,
                agent=config.agent,
                native_agent_kwargs=config.native_agent_kwargs,
                model_name=config.native_model_name,
                grader=config.grader,
                workflow_config=config.workflow_config,
                grader_config=config.grader_config,
                code_dir=rollout_dir,
                environment_config=EnvironmentConfig(
                    type=environment_type,
                    kwargs=config.environment_kwargs or {},
                ),
                trials_dir=resolved_trials_dir,
                cleanup_successful_trials=config.cleanup_successful_trials,
                patch_dockerfile_with_sdk=config.patch_dockerfile_with_sdk,
                agent_setup_timeout_sec=config.agent_setup_timeout_sec,
                max_queue_depth=config.max_queue_depth,
            )
        except (ImportError, ValueError) as exc:
            raise CLIError(
                f"Could not configure the Harbor rollout backend: {exc}",
                code="VALIDATION",
            ) from exc
        _run_server(backend=backend, host=host, port=port)


def serve(config_path: Path, *, host: str, port: int | None) -> None:
    """Run the rollout server declared by ``config_path``."""
    config, rollout_dir = load_serve_config(config_path)
    if config.backend == "simple":
        _serve_simple(config.simple, rollout_dir=rollout_dir, host=host, port=port)
    else:
        _serve_harbor(config.harbor, rollout_dir=rollout_dir, host=host, port=port)


__all__ = [
    "HarborBackendConfig",
    "RolloutServeConfig",
    "SimpleBackendConfig",
    "load_serve_config",
    "serve",
]

"""TOML config loading and validation for eval runs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.workspace_directory_contract import ensure_context_path


class _EvalSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    rollout: str
    entrypoint: str
    limit: Annotated[int, Field(ge=1)] | None = None
    offset: Annotated[int, Field(ge=0)] = 0
    fresh: bool = False
    retry_failed: bool = False


class _LLMSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str | None = None
    api_key_env: str | None = None


class _RunsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: Annotated[int, Field(ge=1)] = 1
    batch_size: Annotated[int, Field(ge=1)] = 1
    pass_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class _OutputSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_samples: bool = False
    output_path: str | None = None
    quiet: bool = False
    debug: bool = False


class _TimeoutsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_workflow_timeout_s: Annotated[float, Field(gt=0)] = 450.0
    grader_timeout_s: Annotated[float, Field(gt=0)] = 150.0


class EvalConfig(BaseModel):
    """Parsed eval TOML configuration."""

    # [eval]
    eval_dataset: str
    eval_rollout: str
    eval_entrypoint: str
    eval_limit: Annotated[int, Field(ge=1)] | None = None
    eval_offset: Annotated[int, Field(ge=0)] = 0
    eval_fresh: bool = False
    eval_retry_failed: bool = False

    # [llm]
    llm_model: str
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None

    # [runs]
    runs_n: Annotated[int, Field(ge=1)] = 1
    runs_batch_size: Annotated[int, Field(ge=1)] = 1
    runs_pass_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    # [output]
    output_log_samples: bool = False
    output_path: str | None = None
    output_quiet: bool = False
    output_debug: bool = False

    # [timeouts]
    timeout_agent_sec: Annotated[float, Field(gt=0)] = 450.0
    timeout_grader_sec: Annotated[float, Field(gt=0)] = 150.0


def load_eval_config(path: Path) -> EvalConfig:
    """Load and validate TOML config. Raises CLIError on any problem."""
    if not path.exists():
        raise CLIError(f"Config file not found: {path}")

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise CLIError(f"Cannot read config file {path}: {e}") from e

    if "eval" not in raw:
        raise CLIError(f"Missing [eval] section in {path}")

    eval_section = raw["eval"]
    if not isinstance(eval_section, dict):
        raise CLIError(f"[eval] must be a TOML table in {path}")

    for required_key in ("rollout", "entrypoint", "dataset"):
        if required_key not in eval_section:
            raise CLIError(f"Missing '{required_key}' in [eval] section of {path}")

    llm_section = raw.get("llm", {})
    if not isinstance(llm_section, dict):
        raise CLIError(f"[llm] must be a TOML table in {path}")
    if "model" not in llm_section:
        raise CLIError(f"Missing [llm].model in {path}")

    if raw.get("grader") is not None:
        raise CLIError(
            "[grader] is no longer supported in eval configs. Grading must be "
            "performed by the rollout server and reported through the grader callback."
        )
    if raw.get("baseline") is not None:
        raise CLIError(
            "[baseline] is no longer supported in eval configs. Run separate eval "
            "configs when comparing models."
        )

    runs_section = raw.get("runs", {})
    output_section = raw.get("output", {})
    timeouts_section = raw.get("timeouts", {})

    try:
        eval_parsed = _EvalSection(**eval_section)
        _LLMSection(**llm_section)
        runs = _RunsSection(**runs_section)
        output = _OutputSection(**output_section)
        timeouts = _TimeoutsSection(**timeouts_section)
    except Exception as e:
        raise CLIError(f"Invalid config in {path}: {e}") from e

    return EvalConfig(
        eval_dataset=eval_section["dataset"],
        eval_rollout=eval_section["rollout"],
        eval_entrypoint=eval_section["entrypoint"],
        eval_limit=eval_parsed.limit,
        eval_offset=eval_parsed.offset,
        eval_fresh=eval_parsed.fresh,
        eval_retry_failed=eval_parsed.retry_failed,
        llm_model=llm_section["model"],
        llm_base_url=llm_section.get("base_url"),
        llm_api_key_env=llm_section.get("api_key_env"),
        runs_n=runs.n,
        runs_batch_size=runs.batch_size,
        runs_pass_threshold=runs.pass_threshold,
        output_log_samples=output.log_samples,
        output_path=output.output_path,
        output_quiet=output.quiet,
        output_debug=output.debug,
        timeout_agent_sec=timeouts.agent_workflow_timeout_s,
        timeout_grader_sec=timeouts.grader_timeout_s,
    )


def resolve_eval_context_paths(
    config: EvalConfig, workspace_directory: Path
) -> EvalConfig:
    """Resolve eval filesystem paths against the active workspace directory."""
    dataset = ensure_context_path(
        Path(config.eval_dataset),
        workspace_directory,
        required_dir="data",
        label="[eval].dataset",
    )
    entrypoint = ensure_context_path(
        Path("rollouts") / config.eval_rollout / config.eval_entrypoint,
        workspace_directory,
        required_dir=f"rollouts/{config.eval_rollout}",
        label="[eval].entrypoint",
    )
    rollout_dir = workspace_directory / "rollouts" / config.eval_rollout
    pyproject_path = rollout_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise CLIError(
            f"rollouts/{config.eval_rollout}/pyproject.toml is required for "
            "`osmosis eval run`. Local eval starts the server with "
            f"`uv run python {config.eval_entrypoint}` from {rollout_dir}."
        )
    return config.model_copy(
        update={
            "eval_dataset": str(dataset),
            "eval_entrypoint": str(
                entrypoint.relative_to(
                    workspace_directory / "rollouts" / config.eval_rollout
                )
            ),
        }
    )


__all__ = ["EvalConfig", "load_eval_config", "resolve_eval_context_paths"]

"""TOML config loading and validation for eval runs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.project_contract import ensure_context_path


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


class _GraderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str | None = None
    config: str | None = None


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


class _BaselineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str | None = None
    api_key_env: str | None = None


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

    # [grader] (auto-discovered when using rollout+entrypoint)
    grader_module: str | None = None
    grader_config: str | None = None

    # [runs]
    runs_n: Annotated[int, Field(ge=1)] = 1
    runs_batch_size: Annotated[int, Field(ge=1)] = 1
    runs_pass_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    # [output]
    output_log_samples: bool = False
    output_path: str | None = None
    output_quiet: bool = False
    output_debug: bool = False

    # [baseline]
    baseline_model: str | None = None
    baseline_base_url: str | None = None
    baseline_api_key_env: str | None = None


def _is_filesystem_grader_config(value: str) -> bool:
    """Return True when [grader].config looks like a local filesystem path."""
    if "://" in value:
        return False
    candidate = Path(value)
    return (
        candidate.is_absolute()
        or "/" in value
        or "\\" in value
        or bool(candidate.suffix)
    )


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

    grader_section = raw.get("grader")
    runs_section = raw.get("runs", {})
    output_section = raw.get("output", {})
    baseline_section = raw.get("baseline", {})

    try:
        eval_parsed = _EvalSection(**eval_section)
        _LLMSection(**llm_section)
        if grader_section:
            _GraderSection(**grader_section)
        runs = _RunsSection(**runs_section)
        output = _OutputSection(**output_section)
        if baseline_section:
            _BaselineSection(**baseline_section)
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
        grader_module=grader_section.get("module") if grader_section else None,
        grader_config=grader_section.get("config") if grader_section else None,
        runs_n=runs.n,
        runs_batch_size=runs.batch_size,
        runs_pass_threshold=runs.pass_threshold,
        output_log_samples=output.log_samples,
        output_path=output.output_path,
        output_quiet=output.quiet,
        output_debug=output.debug,
        baseline_model=baseline_section.get("model") if baseline_section else None,
        baseline_base_url=baseline_section.get("base_url")
        if baseline_section
        else None,
        baseline_api_key_env=baseline_section.get("api_key_env")
        if baseline_section
        else None,
    )


def resolve_eval_context_paths(config: EvalConfig, project_root: Path) -> EvalConfig:
    """Resolve eval filesystem paths against the active project root."""
    dataset = ensure_context_path(
        Path(config.eval_dataset),
        project_root,
        required_dir="data",
        label="[eval].dataset",
    )
    entrypoint = ensure_context_path(
        Path("rollouts") / config.eval_rollout / config.eval_entrypoint,
        project_root,
        required_dir=f"rollouts/{config.eval_rollout}",
        label="[eval].entrypoint",
    )
    grader_config = config.grader_config
    if grader_config:
        candidate = Path(grader_config)
        if _is_filesystem_grader_config(grader_config):
            grader_config = str(
                ensure_context_path(
                    candidate,
                    project_root,
                    required_dir=".",
                    label="[grader].config",
                )
            )

    return config.model_copy(
        update={
            "eval_dataset": str(dataset),
            "eval_entrypoint": str(
                entrypoint.relative_to(project_root / "rollouts" / config.eval_rollout)
            ),
            "grader_config": grader_config,
        }
    )


__all__ = ["EvalConfig", "load_eval_config", "resolve_eval_context_paths"]

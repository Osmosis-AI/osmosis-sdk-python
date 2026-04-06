"""TOML config loading and validation for eval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import tomllib
from pydantic import BaseModel, Field

from osmosis_ai.cli.errors import CLIError


class _EvalSection(BaseModel):
    module: str
    dataset: str


class _LLMSection(BaseModel):
    model: str
    base_url: str | None = None
    api_key_env: str | None = None


class _GraderSection(BaseModel):
    module: str | None = None
    config: str | None = None


class _RunsSection(BaseModel):
    n: Annotated[int, Field(ge=1)] = 1
    batch_size: Annotated[int, Field(ge=1)] = 1
    pass_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class _OutputSection(BaseModel):
    log_samples: bool = False
    output_path: str | None = None


class _BaselineSection(BaseModel):
    model: str
    base_url: str | None = None
    api_key_env: str | None = None


class EvalConfig(BaseModel):
    """Parsed eval TOML configuration."""

    # [eval]
    eval_module: str
    eval_dataset: str

    # [llm]
    llm_model: str
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None

    # [grader]
    has_grader: bool = False
    grader_module: str | None = None
    grader_config: str | None = None

    # [runs]
    runs_n: Annotated[int, Field(ge=1)] = 1
    runs_batch_size: Annotated[int, Field(ge=1)] = 1
    runs_pass_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    # [output]
    output_log_samples: bool = False
    output_path: str | None = None

    # [baseline]
    baseline_model: str | None = None
    baseline_base_url: str | None = None
    baseline_api_key_env: str | None = None


def load_eval_config(path: Path) -> EvalConfig:
    """Load and validate TOML config. Raises CLIError on any problem."""
    if not path.exists():
        raise CLIError(f"Config file not found: {path}")

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {path}: {e}") from e

    if "eval" not in raw:
        raise CLIError(f"Missing [eval] section in {path}")

    eval_section = raw["eval"]
    if "module" not in eval_section:
        raise CLIError(f"Missing 'module' in [eval] section of {path}")
    if "dataset" not in eval_section:
        raise CLIError(f"Missing 'dataset' in [eval] section of {path}")

    llm_section = raw.get("llm", {})
    if "model" not in llm_section:
        raise CLIError(f"Missing [llm].model in {path}")

    grader_section = raw.get("grader")
    has_grader = grader_section is not None

    runs_section = raw.get("runs", {})
    output_section = raw.get("output", {})
    baseline_section = raw.get("baseline", {})

    try:
        _EvalSection(**eval_section)
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
        eval_module=eval_section["module"],
        eval_dataset=eval_section["dataset"],
        llm_model=llm_section["model"],
        llm_base_url=llm_section.get("base_url"),
        llm_api_key_env=llm_section.get("api_key_env"),
        has_grader=has_grader,
        grader_module=grader_section.get("module") if grader_section else None,
        grader_config=grader_section.get("config") if grader_section else None,
        runs_n=runs.n,
        runs_batch_size=runs.batch_size,
        runs_pass_threshold=runs.pass_threshold,
        output_log_samples=output.log_samples,
        output_path=output.output_path,
        baseline_model=baseline_section.get("model") if baseline_section else None,
        baseline_base_url=baseline_section.get("base_url")
        if baseline_section
        else None,
        baseline_api_key_env=baseline_section.get("api_key_env")
        if baseline_section
        else None,
    )


__all__ = ["EvalConfig", "load_eval_config"]

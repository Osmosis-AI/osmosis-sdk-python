"""TOML config loading and validation for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, model_validator

from osmosis_ai.platform.cli.shared_config import (
    BackendValidatedParamSection,
    BaseSubmitConfig,
    load_submit_config,
)

_EVAL_CONFIG_LABEL = "eval"


class _EvaluationSection(BackendValidatedParamSection):
    limit: Any = None
    n: Any = None
    max_concurrent_attempts: Any = None
    pass_threshold: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None

    @model_validator(mode="before")
    @classmethod
    def _accept_batch_size(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "batch_size" not in data:
            return data
        if "max_concurrent_attempts" in data:
            raise ValueError(
                "Set only `max_concurrent_attempts`; `batch_size` is its old name."
            )
        renamed = dict(data)
        renamed["max_concurrent_attempts"] = renamed.pop("batch_size")
        return renamed


class EvalSubmitConfig(BaseSubmitConfig):
    """Parsed evaluation run TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    evaluation: _EvaluationSection = Field(default_factory=_EvaluationSection)

    @property
    def evaluation_config(self) -> dict[str, Any]:
        return self.evaluation.model_dump(exclude_none=True)


def load_eval_submit_config(path: Path) -> EvalSubmitConfig:
    """Load and validate TOML config for evaluation run submit."""
    return load_submit_config(
        path,
        config_class=EvalSubmitConfig,
        extra_sections=[("evaluation", _EvaluationSection)],
        config_label=_EVAL_CONFIG_LABEL,
        secrets_section_required=True,
    )


__all__ = [
    "EvalSubmitConfig",
    "load_eval_submit_config",
]

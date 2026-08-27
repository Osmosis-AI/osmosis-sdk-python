"""TOML config loading and validation for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from osmosis_ai.platform.cli.shared_config import (
    BackendValidatedParamSection,
    BaseSubmitConfig,
    load_submit_config,
)

_EVAL_CONFIG_LABEL = "eval"


class _EvaluationSection(BackendValidatedParamSection):
    limit: Any = None
    n: Any = None
    batch_size: Any = None
    pass_threshold: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None


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

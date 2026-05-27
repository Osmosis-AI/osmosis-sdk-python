"""TOML config loading and validation for `osmosis eval submit` runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.shared_config import (
    AdvancedPassthroughSection,
    BackendValidatedParamSection,
    collect_section_validation_issues,
    collect_top_level_validation_issues,
    config_issues_error,
    parse_section,
    read_toml_file,
    read_toml_table,
    validate_env_values,
    validate_env_var_keys,
    validate_workspace_rollout_paths,
)

_EVAL_CONFIG_LABEL = "eval"
_EVAL_CONFIG_SECTIONS: frozenset[str] = frozenset(
    {"experiment", "evaluation", "advanced", "env", "secrets"}
)


class _ExperimentSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    model_path: str
    dataset: str
    commit_sha: str | None = None


class _EvaluationSection(BackendValidatedParamSection):
    limit: Any = None
    n: Any = None
    batch_size: Any = None
    pass_threshold: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None


class EvalSubmitConfig(BaseModel):
    """Parsed cloud eval TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    experiment: _ExperimentSection
    evaluation: _EvaluationSection
    advanced: AdvancedPassthroughSection = Field(
        default_factory=AdvancedPassthroughSection
    )
    env: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)

    @property
    def experiment_rollout(self) -> str:
        return self.experiment.rollout

    @property
    def experiment_entrypoint(self) -> str:
        return self.experiment.entrypoint

    @property
    def experiment_dataset(self) -> str:
        return self.experiment.dataset

    @property
    def experiment_commit_sha(self) -> str | None:
        return self.experiment.commit_sha

    @property
    def experiment_model_path(self) -> str:
        return self.experiment.model_path

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.experiment.model_dump(exclude_none=True)

    @property
    def evaluation_config(self) -> dict[str, Any]:
        return self.evaluation.model_dump(exclude_none=True)

    @property
    def advanced_config(self) -> dict[str, Any]:
        return self.advanced.model_dump(exclude_none=True)


def load_eval_submit_config(path: Path) -> EvalSubmitConfig:
    """Load and validate TOML config for cloud eval submit."""
    raw = read_toml_file(path)

    experiment_section = read_toml_table(raw, "experiment", path, required=True)
    for required_key in ("rollout", "entrypoint", "model_path", "dataset"):
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    evaluation_section = read_toml_table(raw, "evaluation", path)
    advanced_section = read_toml_table(raw, "advanced", path)
    env_section = read_toml_table(raw, "env", path)
    secrets_section = read_toml_table(raw, "secrets", path)

    issues = [
        *collect_top_level_validation_issues(
            raw, allowed_sections=_EVAL_CONFIG_SECTIONS
        ),
        *(
            issue
            for section_name, model_type, data in (
                ("experiment", _ExperimentSection, experiment_section),
                ("evaluation", _EvaluationSection, evaluation_section),
                ("advanced", AdvancedPassthroughSection, advanced_section),
            )
            for issue in collect_section_validation_issues(
                section_name=section_name,
                model_type=model_type,
                data=data,
            )
        ),
        *validate_env_values(env=env_section, secrets=secrets_section),
    ]
    if issues:
        raise config_issues_error(issues=issues, config_label=_EVAL_CONFIG_LABEL)

    experiment = parse_section(
        section_name="experiment",
        model_type=_ExperimentSection,
        data=experiment_section,
        config_label=_EVAL_CONFIG_LABEL,
    )
    evaluation = parse_section(
        section_name="evaluation",
        model_type=_EvaluationSection,
        data=evaluation_section,
        config_label=_EVAL_CONFIG_LABEL,
    )
    advanced = parse_section(
        section_name="advanced",
        model_type=AdvancedPassthroughSection,
        data=advanced_section,
        config_label=_EVAL_CONFIG_LABEL,
    )
    assert isinstance(experiment, _ExperimentSection)
    assert isinstance(evaluation, _EvaluationSection)
    assert isinstance(advanced, AdvancedPassthroughSection)

    env = {key: value for key, value in env_section.items() if isinstance(value, str)}
    secrets = {
        key: value for key, value in secrets_section.items() if isinstance(value, str)
    }
    validate_env_var_keys(env=env, secrets=secrets, path=path)

    return EvalSubmitConfig(
        experiment=experiment,
        evaluation=evaluation,
        advanced=advanced,
        env=env,
        secrets=secrets,
    )


def validate_eval_submit_context_paths(
    config: EvalSubmitConfig,
    workspace_directory: Path,
) -> None:
    """Validate rollout and entrypoint paths against the workspace directory."""
    validate_workspace_rollout_paths(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        workspace_directory=workspace_directory,
        command_label="Eval",
    )


__all__ = [
    "EvalSubmitConfig",
    "load_eval_submit_config",
    "validate_eval_submit_context_paths",
]

"""TOML config loading and validation for `osmosis train submit` runs."""

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

_TRAINING_CONFIG_LABEL = "training"
_TRAINING_CONFIG_SECTIONS: frozenset[str] = frozenset(
    {
        "experiment",
        "training",
        "sampling",
        "checkpoints",
        "advanced",
        "env",
        "secrets",
    }
)


class _ExperimentSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    model_path: str
    dataset: str
    commit_sha: str | None = None


class _TrainingSection(BackendValidatedParamSection):
    lr: Any = None
    total_epochs: Any = None
    n_samples_per_prompt: Any = None
    rollout_batch_size: Any = None
    max_prompt_length: Any = None
    max_response_length: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None


class _SamplingSection(BackendValidatedParamSection):
    rollout_temperature: Any = None
    rollout_top_p: Any = None


class _CheckpointsSection(BackendValidatedParamSection):
    eval_interval: Any = None
    checkpoint_save_freq: Any = None


class TrainingConfig(BaseModel):
    """Parsed training TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    experiment: _ExperimentSection
    training: _TrainingSection = Field(default_factory=_TrainingSection)
    sampling: _SamplingSection = Field(default_factory=_SamplingSection)
    checkpoints: _CheckpointsSection = Field(default_factory=_CheckpointsSection)
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
    def experiment_model_path(self) -> str:
        return self.experiment.model_path

    @property
    def experiment_dataset(self) -> str:
        return self.experiment.dataset

    @property
    def experiment_commit_sha(self) -> str | None:
        return self.experiment.commit_sha

    @property
    def training_lr(self) -> Any:
        return self.training.lr

    @property
    def training_total_epochs(self) -> Any:
        return self.training.total_epochs

    @property
    def training_n_samples_per_prompt(self) -> Any:
        return self.training.n_samples_per_prompt

    @property
    def training_rollout_batch_size(self) -> Any:
        return self.training.rollout_batch_size

    @property
    def training_max_prompt_length(self) -> Any:
        return self.training.max_prompt_length

    @property
    def training_max_response_length(self) -> Any:
        return self.training.max_response_length

    @property
    def training_agent_workflow_timeout_s(self) -> Any:
        return self.training.agent_workflow_timeout_s

    @property
    def training_grader_timeout_s(self) -> Any:
        return self.training.grader_timeout_s

    @property
    def sampling_rollout_temperature(self) -> Any:
        return self.sampling.rollout_temperature

    @property
    def sampling_rollout_top_p(self) -> Any:
        return self.sampling.rollout_top_p

    @property
    def checkpoints_eval_interval(self) -> Any:
        return self.checkpoints.eval_interval

    @property
    def checkpoints_checkpoint_save_freq(self) -> Any:
        return self.checkpoints.checkpoint_save_freq

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.experiment.model_dump(exclude_none=True)

    @property
    def training_config(self) -> dict[str, Any]:
        return self.training.model_dump(exclude_none=True)

    @property
    def sampling_config(self) -> dict[str, Any]:
        return self.sampling.model_dump(exclude_none=True)

    @property
    def checkpoints_config(self) -> dict[str, Any]:
        return self.checkpoints.model_dump(exclude_none=True)

    @property
    def advanced_config(self) -> dict[str, Any]:
        return self.advanced.model_dump(exclude_none=True)


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate TOML config for training. Raises :class:`CLIError` on any problem."""
    raw = read_toml_file(path)

    experiment_section = read_toml_table(raw, "experiment", path, required=True)
    for required_key in ("rollout", "entrypoint", "model_path", "dataset"):
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    training_section = read_toml_table(raw, "training", path)
    sampling_section = read_toml_table(raw, "sampling", path)
    checkpoints_section = read_toml_table(raw, "checkpoints", path)
    advanced_section = read_toml_table(raw, "advanced", path)
    env_section = read_toml_table(raw, "env", path)
    secrets_section = read_toml_table(raw, "secrets", path)
    issues = [
        *collect_top_level_validation_issues(
            raw, allowed_sections=_TRAINING_CONFIG_SECTIONS
        ),
        *(
            issue
            for section_name, model_type, data in (
                ("experiment", _ExperimentSection, experiment_section),
                ("training", _TrainingSection, training_section),
                ("sampling", _SamplingSection, sampling_section),
                ("checkpoints", _CheckpointsSection, checkpoints_section),
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
        raise config_issues_error(issues=issues, config_label=_TRAINING_CONFIG_LABEL)

    experiment = parse_section(
        section_name="experiment",
        model_type=_ExperimentSection,
        data=experiment_section,
        config_label=_TRAINING_CONFIG_LABEL,
    )
    training = parse_section(
        section_name="training",
        model_type=_TrainingSection,
        data=training_section,
        config_label=_TRAINING_CONFIG_LABEL,
    )
    sampling = parse_section(
        section_name="sampling",
        model_type=_SamplingSection,
        data=sampling_section,
        config_label=_TRAINING_CONFIG_LABEL,
    )
    checkpoints = parse_section(
        section_name="checkpoints",
        model_type=_CheckpointsSection,
        data=checkpoints_section,
        config_label=_TRAINING_CONFIG_LABEL,
    )
    advanced = parse_section(
        section_name="advanced",
        model_type=AdvancedPassthroughSection,
        data=advanced_section,
        config_label=_TRAINING_CONFIG_LABEL,
    )
    assert isinstance(experiment, _ExperimentSection)
    assert isinstance(training, _TrainingSection)
    assert isinstance(sampling, _SamplingSection)
    assert isinstance(checkpoints, _CheckpointsSection)
    assert isinstance(advanced, AdvancedPassthroughSection)

    env = {key: value for key, value in env_section.items() if isinstance(value, str)}
    secrets = {
        key: value for key, value in secrets_section.items() if isinstance(value, str)
    }
    validate_env_var_keys(env=env, secrets=secrets, path=path)

    return TrainingConfig(
        experiment=experiment,
        training=training,
        sampling=sampling,
        checkpoints=checkpoints,
        advanced=advanced,
        env=env,
        secrets=secrets,
    )


def validate_training_context_paths(
    config: TrainingConfig, workspace_directory: Path
) -> None:
    validate_workspace_rollout_paths(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        workspace_directory=workspace_directory,
        command_label="Training",
    )


__all__ = [
    "TrainingConfig",
    "load_training_config",
    "validate_training_context_paths",
]

"""TOML config loading and validation for `osmosis train submit` runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from osmosis_ai.platform.cli.shared_config import (
    BackendValidatedParamSection,
    BaseSubmitConfig,
    load_submit_config,
    validate_workspace_rollout_paths,
)

_TRAIN_SUBMIT_CONFIG_LABEL = "training"


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


class TrainSubmitConfig(BaseSubmitConfig):
    """Parsed training TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    training: _TrainingSection = Field(default_factory=_TrainingSection)
    sampling: _SamplingSection = Field(default_factory=_SamplingSection)
    checkpoints: _CheckpointsSection = Field(default_factory=_CheckpointsSection)

    @property
    def training_config(self) -> dict[str, Any]:
        return self.training.model_dump(exclude_none=True)

    @property
    def sampling_config(self) -> dict[str, Any]:
        return self.sampling.model_dump(exclude_none=True)

    @property
    def checkpoints_config(self) -> dict[str, Any]:
        return self.checkpoints.model_dump(exclude_none=True)


def load_train_submit_config(path: Path) -> TrainSubmitConfig:
    """Load and validate TOML config for ``osmosis train submit``."""
    return load_submit_config(
        path,
        config_class=TrainSubmitConfig,
        extra_sections=[
            ("training", _TrainingSection),
            ("sampling", _SamplingSection),
            ("checkpoints", _CheckpointsSection),
        ],
        config_label=_TRAIN_SUBMIT_CONFIG_LABEL,
    )


def validate_train_submit_context_paths(
    config: TrainSubmitConfig, workspace_directory: Path
) -> None:
    validate_workspace_rollout_paths(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        workspace_directory=workspace_directory,
        command_label="Training",
    )


__all__ = [
    "TrainSubmitConfig",
    "load_train_submit_config",
    "validate_train_submit_context_paths",
]

"""TOML config loading and validation for `osmosis train submit` runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, ConfigDict

from osmosis_ai.cli.errors import CLIError


class _ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    model_path: str
    dataset: str
    commit_sha: str | None = None


class _TrainingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float | None = None
    total_epochs: int | None = None
    n_samples_per_prompt: int | None = None
    global_batch_size: int | None = None
    max_prompt_length: int | None = None
    max_response_length: int | None = None


class _SamplingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_temperature: float | None = None
    rollout_top_p: float | None = None


class _CheckpointsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_interval: int | None = None
    checkpoint_save_freq: int | None = None


class TrainingConfig(BaseModel):
    """Parsed training TOML configuration with flattened section fields."""

    # experiment
    experiment_rollout: str
    experiment_entrypoint: str
    experiment_model_path: str
    experiment_dataset: str
    experiment_commit_sha: str | None

    # training
    training_lr: float | None
    training_total_epochs: int | None
    training_n_samples_per_prompt: int | None
    training_global_batch_size: int | None
    training_max_prompt_length: int | None
    training_max_response_length: int | None

    # sampling
    sampling_rollout_temperature: float | None
    sampling_rollout_top_p: float | None

    # checkpoints
    checkpoints_eval_interval: int | None
    checkpoints_checkpoint_save_freq: int | None

    def to_api_config(self) -> dict[str, Any]:
        """Build the ``config`` dict for the training-runs API payload.

        Merges training, sampling, and checkpoints fields, omitting None values.
        """
        fields: dict[str, Any] = {}
        dump = self.model_dump()
        for prefix in ("training_", "sampling_", "checkpoints_"):
            for key, value in dump.items():
                if key.startswith(prefix) and value is not None:
                    fields[key.removeprefix(prefix)] = value
        return fields


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate TOML config for training. Raises :class:`CLIError` on any problem."""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise CLIError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise CLIError(f"Cannot read config file {path}: {e}") from e

    # ── Validate [experiment] section ────────────────────────────
    if "experiment" not in raw:
        raise CLIError(f"Missing [experiment] section in {path}")

    experiment_section = raw["experiment"]
    if not isinstance(experiment_section, dict):
        raise CLIError(f"[experiment] must be a table in {path}")

    for required_key in ("rollout", "entrypoint", "model_path", "dataset"):
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    training_section = raw.get("training", {})
    sampling_section = raw.get("sampling", {})
    checkpoints_section = raw.get("checkpoints", {})
    try:
        experiment = _ExperimentSection(**experiment_section)
        training = _TrainingSection(**training_section)
        sampling = _SamplingSection(**sampling_section)
        checkpoints = _CheckpointsSection(**checkpoints_section)
    except Exception as e:
        raise CLIError(f"Invalid config in {path}: {e}") from e

    # ── Cross-field validation ───────────────────────────────────
    if training.n_samples_per_prompt is not None and training.n_samples_per_prompt <= 0:
        raise CLIError(
            f"n_samples_per_prompt must be a positive integer, "
            f"got {training.n_samples_per_prompt} in {path}"
        )
    if (
        training.global_batch_size is not None
        and training.n_samples_per_prompt is not None
        and training.global_batch_size % training.n_samples_per_prompt != 0
    ):
        raise CLIError(
            f"global_batch_size ({training.global_batch_size}) must be divisible "
            f"by n_samples_per_prompt ({training.n_samples_per_prompt}) in {path}"
        )

    return TrainingConfig(
        experiment_rollout=experiment.rollout,
        experiment_entrypoint=experiment.entrypoint,
        experiment_model_path=experiment.model_path,
        experiment_dataset=experiment.dataset,
        experiment_commit_sha=experiment.commit_sha,
        training_lr=training.lr,
        training_total_epochs=training.total_epochs,
        training_n_samples_per_prompt=training.n_samples_per_prompt,
        training_global_batch_size=training.global_batch_size,
        training_max_prompt_length=training.max_prompt_length,
        training_max_response_length=training.max_response_length,
        sampling_rollout_temperature=sampling.rollout_temperature,
        sampling_rollout_top_p=sampling.rollout_top_p,
        checkpoints_eval_interval=checkpoints.eval_interval,
        checkpoints_checkpoint_save_freq=checkpoints.checkpoint_save_freq,
    )


__all__ = ["TrainingConfig", "load_training_config"]

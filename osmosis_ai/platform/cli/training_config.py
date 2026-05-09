"""TOML config loading and validation for `osmosis train submit` runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from osmosis_ai.cli.errors import CLIError

# Names the platform-app reserves for built-in env vars on the rollout container.
_RESERVED_ROLLOUT_ENV_NAMES: frozenset[str] = frozenset(
    {
        "GITHUB_CLONE_URL",
        "GITHUB_TOKEN",
        "ENTRYPOINT_SCRIPT",
        "REPOSITORY_PATH",
        "TRAINING_RUN_ID",
        "ROLLOUT_NAME",
        "ROLLOUT_PORT",
    }
)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


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
    rollout_batch_size: int | None = None
    max_prompt_length: int | None = None
    max_response_length: int | None = None
    agent_workflow_timeout_s: float | None = None
    grader_timeout_s: float | None = None


class _SamplingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_temperature: float | None = None
    rollout_top_p: float | None = None


class _CheckpointsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_interval: int | None = None
    checkpoint_save_freq: int | None = None


class _RolloutSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)


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
    training_rollout_batch_size: int | None
    training_max_prompt_length: int | None
    training_max_response_length: int | None
    training_agent_workflow_timeout_s: float | None = None
    training_grader_timeout_s: float | None = None

    # sampling
    sampling_rollout_temperature: float | None
    sampling_rollout_top_p: float | None

    # checkpoints
    checkpoints_eval_interval: int | None
    checkpoints_checkpoint_save_freq: int | None

    # rollout container launch arguments
    rollout_env: dict[str, str]
    rollout_secret_refs: dict[str, str]

    def to_api_config(self) -> dict[str, Any]:
        """Build the ``config`` dict for the training-runs API payload.

        Merges training, sampling, and checkpoints fields, omitting None values.
        Rollout env/secrets are submitted as separate top-level fields, not here.
        """
        fields: dict[str, Any] = {}
        dump = self.model_dump()
        for prefix in ("training_", "sampling_", "checkpoints_"):
            for key, value in dump.items():
                if key.startswith(prefix) and value is not None:
                    fields[key.removeprefix(prefix)] = value
        return fields


def _validate_rollout_env_keys(
    *,
    env: dict[str, str],
    secrets: dict[str, str],
    path: Path,
) -> None:
    """Reject invalid env-var names, reserved names, and overlap between sections."""
    for section_name, section in (("rollout.env", env), ("rollout.secrets", secrets)):
        for key in section:
            if not _ENV_VAR_NAME_RE.match(key):
                raise CLIError(
                    f"Invalid env var name '{key}' in [{section_name}] of {path}: "
                    "must match ^[A-Z_][A-Z0-9_]*$"
                )
            if key in _RESERVED_ROLLOUT_ENV_NAMES:
                raise CLIError(
                    f"'{key}' in [{section_name}] of {path} is reserved by "
                    "the rollout container runtime; choose a different name."
                )

    overlap = sorted(set(env) & set(secrets))
    if overlap:
        names = ", ".join(overlap)
        raise CLIError(
            f"Key(s) appear in both [rollout.env] and [rollout.secrets] of {path}: "
            f"{names}. Each env var name must come from exactly one section."
        )


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
    rollout_section = raw.get("rollout", {})

    try:
        experiment = _ExperimentSection(**experiment_section)
        training = _TrainingSection(**training_section)
        sampling = _SamplingSection(**sampling_section)
        checkpoints = _CheckpointsSection(**checkpoints_section)
        rollout = _RolloutSection(**rollout_section)
    except Exception as e:
        raise CLIError(f"Invalid config in {path}: {e}") from e

    _validate_rollout_env_keys(env=rollout.env, secrets=rollout.secrets, path=path)

    # ── Cross-field validation ───────────────────────────────────
    if training.n_samples_per_prompt is not None and training.n_samples_per_prompt <= 0:
        raise CLIError(
            f"n_samples_per_prompt must be a positive integer, "
            f"got {training.n_samples_per_prompt} in {path}"
        )
    if training.rollout_batch_size is None or training.n_samples_per_prompt is None:
        raise CLIError("rollout_batch_size and n_samples_per_prompt must both be set")

    return TrainingConfig(
        experiment_rollout=experiment.rollout,
        experiment_entrypoint=experiment.entrypoint,
        experiment_model_path=experiment.model_path,
        experiment_dataset=experiment.dataset,
        experiment_commit_sha=experiment.commit_sha,
        training_lr=training.lr,
        training_total_epochs=training.total_epochs,
        training_n_samples_per_prompt=training.n_samples_per_prompt,
        training_rollout_batch_size=training.rollout_batch_size,
        training_max_prompt_length=training.max_prompt_length,
        training_max_response_length=training.max_response_length,
        training_agent_workflow_timeout_s=training.agent_workflow_timeout_s,
        training_grader_timeout_s=training.grader_timeout_s,
        sampling_rollout_temperature=sampling.rollout_temperature,
        sampling_rollout_top_p=sampling.rollout_top_p,
        checkpoints_eval_interval=checkpoints.eval_interval,
        checkpoints_checkpoint_save_freq=checkpoints.checkpoint_save_freq,
        rollout_env=dict(rollout.env),
        rollout_secret_refs=dict(rollout.secrets),
    )


def validate_training_context_paths(config: TrainingConfig, project_root: Path) -> None:
    if Path(config.experiment_rollout).is_absolute():
        raise CLIError("Training rollout must be a logical rollout name.")

    rollouts_root = (project_root / "rollouts").resolve()
    rollout_root = (project_root / "rollouts" / config.experiment_rollout).resolve()
    try:
        rollout_root.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(
            "Training rollout must resolve under the current project's rollouts directory."
        ) from exc

    rollout_path = (rollout_root / config.experiment_entrypoint).resolve()
    try:
        rollout_path.relative_to(rollout_root)
    except ValueError as exc:
        raise CLIError(
            "Training entrypoint must resolve under rollouts/<rollout>/ within the current project."
        ) from exc


__all__ = [
    "TrainingConfig",
    "load_training_config",
    "validate_training_context_paths",
]

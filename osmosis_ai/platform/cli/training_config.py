"""TOML config loading and validation for `osmosis train submit` runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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

    lr: Annotated[float, Field(gt=0.0, le=1.0)] = 1e-6
    total_epochs: Annotated[int, Field(ge=1, le=10_000)] = 1
    n_samples_per_prompt: Annotated[int, Field(ge=1, le=1_024)] = 8
    rollout_batch_size: Annotated[int, Field(ge=1, le=1_000_000)] = 64
    max_prompt_length: Annotated[int, Field(ge=1, le=262_144)] = 8_192
    max_response_length: Annotated[int, Field(ge=1, le=262_144)] = 8_192
    agent_workflow_timeout_s: Annotated[float, Field(ge=1, le=86_400)] = 450.0
    grader_timeout_s: Annotated[float, Field(ge=1, le=86_400)] = 150.0


class _SamplingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    rollout_top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class _CheckpointsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_interval: Annotated[int | None, Field(ge=1, le=1_000_000)] = None
    checkpoint_save_freq: Annotated[int, Field(ge=1, le=1_000_000)] = 20


class _RolloutSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)


class _TrainingRunParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training: _TrainingSection = Field(default_factory=_TrainingSection)
    sampling: _SamplingSection = Field(default_factory=_SamplingSection)
    checkpoints: _CheckpointsSection = Field(default_factory=_CheckpointsSection)

    def to_api_config(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for section in (self.training, self.sampling, self.checkpoints):
            fields.update(section.model_dump(exclude_none=True))
        return fields


def _format_bound(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_input(value: Any) -> str:
    return repr(value)


def _format_field_path(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _format_field_bounds(model_type: type[BaseModel], field_name: str) -> str | None:
    field = model_type.model_fields.get(field_name)
    if field is None:
        return None

    lower: list[str] = []
    upper: list[str] = []
    for metadata in field.metadata:
        gt = getattr(metadata, "gt", None)
        ge = getattr(metadata, "ge", None)
        lt = getattr(metadata, "lt", None)
        le = getattr(metadata, "le", None)
        if gt is not None:
            lower.append(f"> {_format_bound(gt)}")
        if ge is not None:
            lower.append(f">= {_format_bound(ge)}")
        if lt is not None:
            upper.append(f"< {_format_bound(lt)}")
        if le is not None:
            upper.append(f"<= {_format_bound(le)}")

    bounds = [*lower, *upper]
    if not bounds:
        return None
    return " and ".join(bounds)


def _format_validation_issue(
    *,
    error: dict[str, Any],
    section_name: str,
    model_type: type[BaseModel],
    path: Path,
) -> str:
    loc = tuple(error.get("loc") or ())
    field_path = _format_field_path(loc)
    field_name = str(loc[0]) if loc else section_name
    error_type = str(error.get("type"))

    if error_type == "extra_forbidden":
        return f"Unknown key '{field_name}' in [{section_name}] of {path}"

    subject = f"{field_path} in [{section_name}] of {path}"
    value = _format_input(error.get("input"))

    bounds = _format_field_bounds(model_type, field_name)
    if (
        error_type
        in {
            "greater_than",
            "greater_than_equal",
            "less_than",
            "less_than_equal",
        }
        and bounds is not None
    ):
        return f"{subject} must be {bounds}, got {value}"

    expected_types = {
        "bool_parsing": "a boolean",
        "bool_type": "a boolean",
        "dict_type": "a table",
        "float_parsing": "a number",
        "float_type": "a number",
        "int_parsing": "an integer",
        "int_type": "an integer",
        "string_type": "a string",
    }
    expected_type = expected_types.get(error_type)
    if expected_type is not None:
        return f"{subject} must be {expected_type}, got {value}"

    return f"{subject} is invalid: {error.get('msg', 'invalid value')}"


def _config_validation_error(
    *,
    error: ValidationError,
    section_name: str,
    model_type: type[BaseModel],
    path: Path,
) -> CLIError:
    issues = [
        _format_validation_issue(
            error=validation_issue,
            section_name=section_name,
            model_type=model_type,
            path=path,
        )
        for validation_issue in error.errors()
    ]
    if len(issues) == 1:
        return CLIError(issues[0])
    lines = [
        f"Invalid [{section_name}] section in {path}:",
        *[f"- {i}" for i in issues],
    ]
    return CLIError("\n".join(lines))


def _read_table(
    raw: dict[str, Any],
    section_name: str,
    path: Path,
    *,
    required: bool = False,
) -> dict[str, Any]:
    if section_name not in raw:
        if required:
            raise CLIError(f"Missing [{section_name}] section in {path}")
        return {}

    section = raw[section_name]
    if not isinstance(section, dict):
        raise CLIError(f"[{section_name}] must be a table in {path}")
    return section


def _parse_section(
    *,
    section_name: str,
    model_type: type[BaseModel],
    data: dict[str, Any],
    path: Path,
) -> BaseModel:
    try:
        return model_type(**data)
    except ValidationError as e:
        raise _config_validation_error(
            error=e,
            section_name=section_name,
            model_type=model_type,
            path=path,
        ) from e


class TrainingConfig(BaseModel):
    """Parsed training TOML configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment: _ExperimentSection
    params: _TrainingRunParams
    rollout: _RolloutSection

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
    def training_lr(self) -> float:
        return self.params.training.lr

    @property
    def training_total_epochs(self) -> int:
        return self.params.training.total_epochs

    @property
    def training_n_samples_per_prompt(self) -> int:
        return self.params.training.n_samples_per_prompt

    @property
    def training_rollout_batch_size(self) -> int:
        return self.params.training.rollout_batch_size

    @property
    def training_max_prompt_length(self) -> int:
        return self.params.training.max_prompt_length

    @property
    def training_max_response_length(self) -> int:
        return self.params.training.max_response_length

    @property
    def training_agent_workflow_timeout_s(self) -> float:
        return self.params.training.agent_workflow_timeout_s

    @property
    def training_grader_timeout_s(self) -> float:
        return self.params.training.grader_timeout_s

    @property
    def sampling_rollout_temperature(self) -> float:
        return self.params.sampling.rollout_temperature

    @property
    def sampling_rollout_top_p(self) -> float:
        return self.params.sampling.rollout_top_p

    @property
    def checkpoints_eval_interval(self) -> int | None:
        return self.params.checkpoints.eval_interval

    @property
    def checkpoints_checkpoint_save_freq(self) -> int:
        return self.params.checkpoints.checkpoint_save_freq

    @property
    def rollout_env(self) -> dict[str, str]:
        return dict(self.rollout.env)

    @property
    def rollout_secret_refs(self) -> dict[str, str]:
        return dict(self.rollout.secrets)

    def to_api_config(self) -> dict[str, Any]:
        """Build the ``config`` dict for the training-runs API payload.

        Merges training, sampling, and checkpoints fields, omitting None values.
        Rollout env/secrets are submitted as separate top-level fields, not here.
        """
        return self.params.to_api_config()


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

    experiment_section = _read_table(raw, "experiment", path, required=True)
    for required_key in ("rollout", "entrypoint", "model_path", "dataset"):
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    experiment = _parse_section(
        section_name="experiment",
        model_type=_ExperimentSection,
        data=experiment_section,
        path=path,
    )
    training = _parse_section(
        section_name="training",
        model_type=_TrainingSection,
        data=_read_table(raw, "training", path),
        path=path,
    )
    sampling = _parse_section(
        section_name="sampling",
        model_type=_SamplingSection,
        data=_read_table(raw, "sampling", path),
        path=path,
    )
    checkpoints = _parse_section(
        section_name="checkpoints",
        model_type=_CheckpointsSection,
        data=_read_table(raw, "checkpoints", path),
        path=path,
    )
    rollout = _parse_section(
        section_name="rollout",
        model_type=_RolloutSection,
        data=_read_table(raw, "rollout", path),
        path=path,
    )
    assert isinstance(experiment, _ExperimentSection)
    assert isinstance(training, _TrainingSection)
    assert isinstance(sampling, _SamplingSection)
    assert isinstance(checkpoints, _CheckpointsSection)
    assert isinstance(rollout, _RolloutSection)

    _validate_rollout_env_keys(env=rollout.env, secrets=rollout.secrets, path=path)

    return TrainingConfig(
        experiment=experiment,
        params=_TrainingRunParams(
            training=training,
            sampling=sampling,
            checkpoints=checkpoints,
        ),
        rollout=rollout,
    )


def validate_training_context_paths(
    config: TrainingConfig, workspace_directory: Path
) -> None:
    if Path(config.experiment_rollout).is_absolute():
        raise CLIError("Training rollout must be a logical rollout name.")

    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_root = (
        workspace_directory / "rollouts" / config.experiment_rollout
    ).resolve()
    try:
        rollout_root.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(
            "Training rollout must resolve under the current workspace directory's rollouts directory."
        ) from exc

    rollout_path = (rollout_root / config.experiment_entrypoint).resolve()
    try:
        rollout_path.relative_to(rollout_root)
    except ValueError as exc:
        raise CLIError(
            "Training entrypoint must resolve under rollouts/<rollout>/ within the current workspace directory."
        ) from exc


__all__ = [
    "TrainingConfig",
    "load_training_config",
    "validate_training_context_paths",
]

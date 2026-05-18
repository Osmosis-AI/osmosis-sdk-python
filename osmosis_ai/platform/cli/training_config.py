"""TOML config loading and validation for `osmosis train submit` runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

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
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    model_path: str
    dataset: str
    commit_sha: str | None = None


class _BackendValidatedParamSection(BaseModel):
    """Preserve training params for server-side schema validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class _TrainingSection(_BackendValidatedParamSection):
    lr: Any = None
    total_epochs: Any = None
    n_samples_per_prompt: Any = None
    rollout_batch_size: Any = None
    max_prompt_length: Any = None
    max_response_length: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None


class _SamplingSection(_BackendValidatedParamSection):
    rollout_temperature: Any = None
    rollout_top_p: Any = None


class _CheckpointsSection(_BackendValidatedParamSection):
    eval_interval: Any = None
    checkpoint_save_freq: Any = None


_PARAM_SECTION_MODELS = (
    ("training", _TrainingSection),
    ("sampling", _SamplingSection),
    ("checkpoints", _CheckpointsSection),
)

_KNOWN_PARAM_SECTION_BY_KEY = {
    key: section_name
    for section_name, model_type in _PARAM_SECTION_MODELS
    for key in model_type.model_fields
}
_MAX_KEY_CORRECTION_DISTANCE = 3


class _RolloutSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    env: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)


class _TrainingRunParams(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    training: _TrainingSection = Field(default_factory=_TrainingSection)
    sampling: _SamplingSection = Field(default_factory=_SamplingSection)
    checkpoints: _CheckpointsSection = Field(default_factory=_CheckpointsSection)

    def to_api_config(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for section in (self.training, self.sampling, self.checkpoints):
            fields.update(section.model_dump(exclude_none=True))
        return fields


def _format_input(value: Any) -> str:
    return repr(value)


def _format_field_path(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _validation_issue_to_training_config_issue(
    *,
    error: ErrorDetails,
    section_name: str,
) -> dict[str, str]:
    loc = tuple(error.get("loc") or ())
    field_path = _format_field_path(loc)
    error_type = str(error.get("type"))
    issue_key = f"{section_name}.{field_path}" if field_path else section_name

    if error_type == "extra_forbidden":
        return {"key": issue_key, "message": "Unrecognized key"}

    value = _format_input(error.get("input"))

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
        return {
            "key": issue_key,
            "message": f"must be {expected_type}, got {value}",
        }

    return {
        "key": issue_key,
        "message": f"is invalid: {error.get('msg', 'invalid value')}",
    }


def _config_validation_error(
    *,
    error: ValidationError,
    section_name: str,
    model_type: type[BaseModel],
    path: Path,
) -> CLIError:
    issues = [
        _validation_issue_to_training_config_issue(
            error=validation_issue,
            section_name=section_name,
        )
        for validation_issue in error.errors()
    ]
    return _training_config_issues_error(issues=issues)


def _training_config_issues_error(
    *,
    issues: list[dict[str, str]],
) -> CLIError:
    lines = [
        "Invalid training config:",
        *[
            f"  - {issue['key']}: {_format_training_config_issue(issue)}"
            for issue in issues
        ],
    ]
    return CLIError(
        "\n".join(lines),
        details={
            "error": "Invalid training config",
            "issues": issues,
        },
    )


def _format_training_config_issue(issue: dict[str, str]) -> str:
    message = issue["message"]
    correction = issue.get("key_correction")
    if correction:
        return f"{message} (did you mean '{correction}'?)"
    return message


def _levenshtein_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _find_key_correction(key: str) -> str | None:
    best_key: str | None = None
    best_distance = _MAX_KEY_CORRECTION_DISTANCE + 1
    for valid_key in _KNOWN_PARAM_SECTION_BY_KEY:
        distance = _levenshtein_distance(key, valid_key)
        if distance < best_distance:
            best_distance = distance
            best_key = valid_key
    return best_key if best_distance <= _MAX_KEY_CORRECTION_DISTANCE else None


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


def _collect_section_validation_issues(
    *,
    section_name: str,
    model_type: type[BaseModel],
    data: dict[str, Any],
) -> list[dict[str, str]]:
    try:
        model_type(**data)
    except ValidationError as e:
        return [
            _validation_issue_to_training_config_issue(
                error=validation_issue,
                section_name=section_name,
            )
            for validation_issue in e.errors()
        ]
    return []


def _collect_param_section_key_issues(
    *,
    sections: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    issues: list[dict[str, str]] = []
    for section_name, section in sections.items():
        for key in section:
            previous_section = seen.get(key)
            if previous_section is not None:
                issues.append(
                    {
                        "key": key,
                        "message": (
                            f"appears in both [{previous_section}] and [{section_name}]"
                        ),
                    }
                )
            seen[key] = section_name

    for section_name, section in sections.items():
        for key in section:
            expected_section = _KNOWN_PARAM_SECTION_BY_KEY.get(key)
            if expected_section is not None and expected_section != section_name:
                issues.append(
                    {
                        "key": key,
                        "message": (
                            f"belongs in [{expected_section}], not [{section_name}]"
                        ),
                    }
                )
            elif expected_section is None:
                issue = {
                    "key": key,
                    "message": "Unrecognized key",
                }
                correction = _find_key_correction(key)
                if correction is not None:
                    issue["key_correction"] = correction
                issues.append(issue)

    return issues


class TrainingConfig(BaseModel):
    """Parsed training TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

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
    def training_lr(self) -> Any:
        return self.params.training.lr

    @property
    def training_total_epochs(self) -> Any:
        return self.params.training.total_epochs

    @property
    def training_n_samples_per_prompt(self) -> Any:
        return self.params.training.n_samples_per_prompt

    @property
    def training_rollout_batch_size(self) -> Any:
        return self.params.training.rollout_batch_size

    @property
    def training_max_prompt_length(self) -> Any:
        return self.params.training.max_prompt_length

    @property
    def training_max_response_length(self) -> Any:
        return self.params.training.max_response_length

    @property
    def training_agent_workflow_timeout_s(self) -> Any:
        return self.params.training.agent_workflow_timeout_s

    @property
    def training_grader_timeout_s(self) -> Any:
        return self.params.training.grader_timeout_s

    @property
    def sampling_rollout_temperature(self) -> Any:
        return self.params.sampling.rollout_temperature

    @property
    def sampling_rollout_top_p(self) -> Any:
        return self.params.sampling.rollout_top_p

    @property
    def checkpoints_eval_interval(self) -> Any:
        return self.params.checkpoints.eval_interval

    @property
    def checkpoints_checkpoint_save_freq(self) -> Any:
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

    training_section = _read_table(raw, "training", path)
    sampling_section = _read_table(raw, "sampling", path)
    checkpoints_section = _read_table(raw, "checkpoints", path)
    rollout_section = _read_table(raw, "rollout", path)
    param_sections = {
        "training": training_section,
        "sampling": sampling_section,
        "checkpoints": checkpoints_section,
    }
    issues = [
        *_collect_section_validation_issues(
            section_name="experiment",
            model_type=_ExperimentSection,
            data=experiment_section,
        ),
        *_collect_param_section_key_issues(
            sections=param_sections,
        ),
        *_collect_section_validation_issues(
            section_name="rollout",
            model_type=_RolloutSection,
            data=rollout_section,
        ),
    ]
    if issues:
        raise _training_config_issues_error(issues=issues)

    experiment = _parse_section(
        section_name="experiment",
        model_type=_ExperimentSection,
        data=experiment_section,
        path=path,
    )
    training = _parse_section(
        section_name="training",
        model_type=_TrainingSection,
        data=training_section,
        path=path,
    )
    sampling = _parse_section(
        section_name="sampling",
        model_type=_SamplingSection,
        data=sampling_section,
        path=path,
    )
    checkpoints = _parse_section(
        section_name="checkpoints",
        model_type=_CheckpointsSection,
        data=checkpoints_section,
        path=path,
    )
    rollout = _parse_section(
        section_name="rollout",
        model_type=_RolloutSection,
        data=rollout_section,
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

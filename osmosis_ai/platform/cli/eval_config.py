"""TOML config loading and validation for `osmosis eval submit` runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

from osmosis_ai.cli.errors import CLIError

_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_EVAL_CONFIG_SECTIONS: frozenset[str] = frozenset(
    {"experiment", "llm", "evaluation", "env", "secrets"}
)


class _ExperimentSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    dataset: str
    commit_sha: str | None = None


class _LLMSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model_path: str
    base_url: str | None = None


class _EvaluationSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    limit: Any = None
    n: Any = None
    batch_size: Any = None
    pass_threshold: Any = None
    agent_workflow_timeout_s: Any = None
    grader_timeout_s: Any = None


def _format_input(value: Any) -> str:
    return repr(value)


def _format_field_path(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _validation_issue_to_eval_config_issue(
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


def _eval_config_issues_error(*, issues: list[dict[str, str]]) -> CLIError:
    lines = [
        "Invalid eval config:",
        *[f"  - {issue['key']}: {issue['message']}" for issue in issues],
    ]
    return CLIError(
        "\n".join(lines),
        details={
            "error": "Invalid eval config",
            "issues": issues,
        },
    )


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
            _validation_issue_to_eval_config_issue(
                error=validation_issue,
                section_name=section_name,
            )
            for validation_issue in e.errors()
        ]
    return []


def _collect_top_level_validation_issues(raw: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"key": section_name, "message": "Unrecognized section"}
        for section_name in raw
        if section_name not in _EVAL_CONFIG_SECTIONS
    ]


def _parse_section(
    *,
    section_name: str,
    model_type: type[BaseModel],
    data: dict[str, Any],
) -> BaseModel:
    try:
        return model_type(**data)
    except ValidationError as e:
        raise _eval_config_issues_error(
            issues=[
                _validation_issue_to_eval_config_issue(
                    error=validation_issue,
                    section_name=section_name,
                )
                for validation_issue in e.errors()
            ]
        ) from e


def _validate_env_var_maps(
    *,
    env: dict[str, str],
    secrets: dict[str, str],
    path: Path,
) -> None:
    """Reject invalid env-var names, reserved names, and overlap between maps."""
    for section_name, section in (("env", env), ("secrets", secrets)):
        for key in section:
            if not _ENV_VAR_NAME_RE.match(key):
                raise CLIError(
                    f"Invalid env var name '{key}' in [{section_name}] of {path}: "
                    "must match ^[A-Z_][A-Z0-9_]*$"
                )
            if key.startswith("_OSMOSIS_"):
                raise CLIError(
                    f"'{key}' in [{section_name}] of {path}: env var names starting "
                    "with _OSMOSIS_ are reserved by the platform; choose a different name."
                )

    overlap = sorted(set(env) & set(secrets))
    if overlap:
        names = ", ".join(overlap)
        raise CLIError(
            f"Key(s) appear in both [env] and [secrets] of {path}: "
            f"{names}. Each env var name must come from exactly one section."
        )


def _validate_env_values(
    *,
    env: dict[str, Any],
    secrets: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for section_name, section in (("env", env), ("secrets", secrets)):
        for key, value in section.items():
            if not isinstance(value, str):
                issues.append(
                    {
                        "key": f"{section_name}.{key}",
                        "message": f"must be a string, got {_format_input(value)}",
                    }
                )
    return issues


class EvalSubmitConfig(BaseModel):
    """Parsed cloud eval TOML configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    experiment: _ExperimentSection
    llm: _LLMSection
    evaluation: _EvaluationSection
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
    def llm_model_path(self) -> str:
        return self.llm.model_path

    @property
    def llm_base_url(self) -> str | None:
        return self.llm.base_url

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.experiment.model_dump(exclude_none=True)

    @property
    def llm_config(self) -> dict[str, Any]:
        return self.llm.model_dump(exclude_none=True)

    @property
    def evaluation_config(self) -> dict[str, Any]:
        return self.evaluation.model_dump(exclude_none=True)


def load_eval_submit_config(path: Path) -> EvalSubmitConfig:
    """Load and validate TOML config for cloud eval submit."""
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
    for required_key in ("rollout", "entrypoint", "dataset"):
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    llm_section = _read_table(raw, "llm", path, required=True)
    if "model_path" not in llm_section:
        raise CLIError(f"Missing 'model_path' in [llm] section of {path}")

    evaluation_section = _read_table(raw, "evaluation", path)
    env_section = _read_table(raw, "env", path)
    secrets_section = _read_table(raw, "secrets", path)

    issues = [
        *_collect_top_level_validation_issues(raw),
        *_collect_section_validation_issues(
            section_name="experiment",
            model_type=_ExperimentSection,
            data=experiment_section,
        ),
        *_collect_section_validation_issues(
            section_name="llm",
            model_type=_LLMSection,
            data=llm_section,
        ),
        *_collect_section_validation_issues(
            section_name="evaluation",
            model_type=_EvaluationSection,
            data=evaluation_section,
        ),
        *_validate_env_values(env=env_section, secrets=secrets_section),
    ]
    if issues:
        raise _eval_config_issues_error(issues=issues)

    experiment = _parse_section(
        section_name="experiment",
        model_type=_ExperimentSection,
        data=experiment_section,
    )
    llm = _parse_section(section_name="llm", model_type=_LLMSection, data=llm_section)
    evaluation = _parse_section(
        section_name="evaluation",
        model_type=_EvaluationSection,
        data=evaluation_section,
    )
    assert isinstance(experiment, _ExperimentSection)
    assert isinstance(llm, _LLMSection)
    assert isinstance(evaluation, _EvaluationSection)

    env = {key: value for key, value in env_section.items() if isinstance(value, str)}
    secrets = {
        key: value for key, value in secrets_section.items() if isinstance(value, str)
    }
    _validate_env_var_maps(env=env, secrets=secrets, path=path)

    return EvalSubmitConfig(
        experiment=experiment,
        llm=llm,
        evaluation=evaluation,
        env=env,
        secrets=secrets,
    )


def validate_eval_submit_context_paths(
    config: EvalSubmitConfig,
    workspace_directory: Path,
) -> None:
    """Validate rollout and entrypoint paths against the workspace directory."""
    if Path(config.experiment_rollout).is_absolute():
        raise CLIError("Eval rollout must be a logical rollout name.")

    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_root = (
        workspace_directory / "rollouts" / config.experiment_rollout
    ).resolve()
    try:
        rollout_root.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(
            "Eval rollout must resolve under the current workspace directory's rollouts directory."
        ) from exc

    entrypoint_path = (rollout_root / config.experiment_entrypoint).resolve()
    try:
        entrypoint_path.relative_to(rollout_root)
    except ValueError as exc:
        raise CLIError(
            "Eval entrypoint must resolve under rollouts/<rollout>/ within the current workspace directory."
        ) from exc


__all__ = [
    "EvalSubmitConfig",
    "load_eval_submit_config",
    "validate_eval_submit_context_paths",
]

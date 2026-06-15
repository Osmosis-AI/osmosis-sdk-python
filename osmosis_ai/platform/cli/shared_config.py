"""Shared building blocks for ``osmosis train submit`` and ``osmosis eval submit``."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import ErrorDetails

from osmosis_ai.cli.errors import CLIError

ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# A pinned commit SHA is a hex string. Git's default short form is 7 chars and a
# full SHA-1 is 40; anything outside that range is almost certainly a typo.
COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
RESERVED_ENV_PREFIX = "_OSMOSIS_"

_EXPECTED_TYPE_BY_ERROR: dict[str, str] = {
    "bool_parsing": "a boolean",
    "bool_type": "a boolean",
    "dict_type": "a table",
    "float_parsing": "a number",
    "float_type": "a number",
    "int_parsing": "an integer",
    "int_type": "an integer",
    "string_type": "a string",
}


class BackendValidatedParamSection(BaseModel):
    """Base for TOML sections whose value-level schema is owned by the backend.

    The SDK only enforces structure (rejects unknown keys via ``extra='forbid'``)
    and leaves value-level validation to the platform/backend. This keeps the
    SDK from having to track backend schema evolution.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class AdvancedPassthroughSection(BaseModel):
    """Preserve advanced backend params for server-side schema validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


def validate_workspace_rollout_paths(
    *,
    rollout: str,
    entrypoint: str,
    workspace_directory: Path,
    command_label: str,
) -> None:
    """Validate ``rollout`` is a single-segment logical name and that
    ``<rollouts>/<rollout>/<entrypoint>`` resolves under ``rollouts/<rollout>/``.

    ``command_label`` is the human-facing prefix in error messages
    (e.g. ``"Training"`` or ``"Evaluation"``).
    """
    if Path(rollout).is_absolute():
        raise CLIError(f"{command_label} rollout must be a logical rollout name.")

    if rollout in ("", ".", ".."):
        raise CLIError(
            f"{command_label} rollout {rollout!r} is not a valid rollout name."
        )

    if "/" in rollout or "\\" in rollout:
        raise CLIError(
            f"{command_label} rollout {rollout!r} must be a single-segment name "
            "(no path separators)."
        )

    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_root = (workspace_directory / "rollouts" / rollout).resolve()
    try:
        rollout_root.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(
            f"{command_label} rollout must resolve under the current workspace "
            "directory's rollouts directory."
        ) from exc

    entrypoint_path = (rollout_root / entrypoint).resolve()
    try:
        entrypoint_path.relative_to(rollout_root)
    except ValueError as exc:
        raise CLIError(
            f"{command_label} entrypoint must resolve under rollouts/<rollout>/ "
            "within the current workspace directory."
        ) from exc


def read_toml_file(path: Path) -> dict[str, Any]:
    """Read a TOML file. Raises :class:`CLIError` on any I/O or parse problem."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise CLIError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise CLIError(f"Cannot read config file {path}: {e}") from e


def read_toml_table(
    raw: dict[str, Any],
    section_name: str,
    path: Path,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Read a top-level TOML table; enforce ``required`` and ``dict`` type."""
    if section_name not in raw:
        if required:
            raise CLIError(f"Missing [{section_name}] section in {path}")
        return {}

    section = raw[section_name]
    if not isinstance(section, dict):
        raise CLIError(f"[{section_name}] must be a table in {path}")
    return section


def read_toml_secrets(
    raw: dict[str, Any],
    path: Path,
    *,
    section_required: bool = False,
) -> list[str]:
    """Read secret refs from ``[secrets].required`` when present.

    Training configs may omit the whole ``[secrets]`` section. Evaluation
    configs must include it. When the section exists, it must always include
    the ``required`` field.
    """
    if "secrets" not in raw:
        if section_required:
            raise CLIError(f"Missing [secrets] section in {path}")
        return []

    value = raw["secrets"]
    if not isinstance(value, dict):
        raise CLIError(f"[secrets] must be a table in {path}")

    extra_keys = sorted(set(value) - {"required"})
    if extra_keys:
        raise CLIError(
            f"[secrets] in {path} only supports the 'required' field; "
            f"unknown key(s): {', '.join(extra_keys)}."
        )

    if "required" not in value:
        raise CLIError(f"Missing 'required' in [secrets] section of {path}")

    required = value["required"]
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise CLIError(
            f"'required' in [secrets] of {path} must be a list of strings, "
            'e.g. required = ["OPENAI_API_KEY"].'
        )
    return required


def format_input(value: Any) -> str:
    return repr(value)


def format_field_path(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def validation_issue_to_config_issue(
    *,
    error: ErrorDetails,
    section_name: str,
) -> dict[str, str]:
    """Convert a Pydantic ``ErrorDetails`` into a CLI-friendly issue dict."""
    loc = tuple(error.get("loc") or ())
    field_path = format_field_path(loc)
    error_type = str(error.get("type"))
    issue_key = f"{section_name}.{field_path}" if field_path else section_name

    if error_type == "extra_forbidden":
        return {"key": issue_key, "message": "Unrecognized key"}

    expected_type = _EXPECTED_TYPE_BY_ERROR.get(error_type)
    if expected_type is not None:
        return {
            "key": issue_key,
            "message": f"must be {expected_type}, got {format_input(error.get('input'))}",
        }

    return {
        "key": issue_key,
        "message": f"is invalid: {error.get('msg', 'invalid value')}",
    }


def format_config_issue(issue: dict[str, str]) -> str:
    """Render a single issue's message, surfacing any backend-provided correction."""
    message = issue["message"]
    correction = issue.get("key_correction")
    if correction:
        return f"{message} (did you mean '{correction}'?)"
    return message


def config_issues_error(
    *,
    issues: list[dict[str, str]],
    config_label: str,
) -> CLIError:
    """Build a :class:`CLIError` that lists every issue for the given config kind."""
    header = f"Invalid {config_label} config"
    lines = [
        f"{header}:",
        *[f"  - {issue['key']}: {format_config_issue(issue)}" for issue in issues],
    ]
    return CLIError(
        "\n".join(lines),
        details={"error": header, "issues": issues},
    )


def collect_section_validation_issues(
    *,
    section_name: str,
    model_type: type[BaseModel],
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Return CLI-friendly issues from instantiating ``model_type`` with ``data``."""
    try:
        model_type(**data)
    except ValidationError as e:
        return [
            validation_issue_to_config_issue(
                error=validation_issue,
                section_name=section_name,
            )
            for validation_issue in e.errors()
        ]
    return []


def collect_top_level_validation_issues(
    raw: dict[str, Any],
    *,
    allowed_sections: frozenset[str],
) -> list[dict[str, str]]:
    """Flag any top-level TOML key that is not in ``allowed_sections``."""
    return [
        {"key": section_name, "message": "Unrecognized section"}
        for section_name in raw
        if section_name not in allowed_sections
    ]


def parse_section(
    *,
    section_name: str,
    model_type: type[BaseModel],
    data: dict[str, Any],
    config_label: str,
) -> BaseModel:
    """Instantiate ``model_type`` or raise a labelled :class:`CLIError`."""
    try:
        return model_type(**data)
    except ValidationError as e:
        raise config_issues_error(
            issues=[
                validation_issue_to_config_issue(
                    error=validation_issue,
                    section_name=section_name,
                )
                for validation_issue in e.errors()
            ],
            config_label=config_label,
        ) from e


def validate_env_var_keys(
    *,
    env: dict[str, str],
    path: Path,
) -> None:
    """Reject invalid or reserved [env] var names."""
    for key in env:
        if not ENV_VAR_NAME_RE.match(key):
            raise CLIError(
                f"Invalid env var name '{key}' in [env] of {path}: "
                "must match ^[A-Z_][A-Z0-9_]*$"
            )
        if key.startswith(RESERVED_ENV_PREFIX):
            raise CLIError(
                f"'{key}' in [env] of {path}: env var names starting "
                f"with {RESERVED_ENV_PREFIX} are reserved by the platform; "
                "choose a different name."
            )


def validate_secret_names(
    *,
    secrets: list[str],
    env: dict[str, str],
    path: Path,
) -> None:
    """Validate each secret name, reject duplicates and overlap with [env] keys."""
    duplicates = sorted({name for name in secrets if secrets.count(name) > 1})
    if duplicates:
        raise CLIError(
            f"Duplicate secret name(s) in [secrets].required of {path}: "
            f"{', '.join(duplicates)}. Each name must appear once."
        )

    for name in secrets:
        if not SECRET_NAME_RE.match(name):
            raise CLIError(
                f"Invalid secret name '{name}' in {path}: use uppercase "
                "letters, digits, and underscores, starting with a letter "
                "(e.g. MY_SECRET). Must match ^[A-Z][A-Z0-9_]*$."
            )

    overlap = sorted(set(env) & set(secrets))
    if overlap:
        names = ", ".join(overlap)
        raise CLIError(
            f"Name(s) appear in both [env] and [secrets] of {path}: "
            f"{names}. Each env var name must come from exactly one place."
        )


class ExperimentSection(BaseModel):
    """``[experiment]`` table — shared between train and eval submit configs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rollout: str
    entrypoint: str
    model_path: str
    dataset: str
    commit_sha: str | None = None

    @field_validator("commit_sha")
    @classmethod
    def _validate_commit_sha(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not COMMIT_SHA_RE.fullmatch(value):
            raise ValueError(
                "must be a hexadecimal Git commit SHA (7-40 hex characters), "
                f"e.g. a full 40-character SHA; got {value!r}"
            )
        return value


_EXPERIMENT_REQUIRED_KEYS: tuple[str, ...] = (
    "rollout",
    "entrypoint",
    "model_path",
    "dataset",
)


class BaseSubmitConfig(BaseModel):
    """Common skeleton for ``train submit`` / ``eval submit`` TOML configs.

    Subclasses add their own optional sections (e.g. ``training``, ``sampling``,
    ``checkpoints`` or ``evaluation``) but inherit ``[experiment]``, ``[advanced]``,
    ``[env]`` and ``[secrets]`` plus the matching properties used by the
    submit flow.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    experiment: ExperimentSection
    advanced: AdvancedPassthroughSection = Field(
        default_factory=AdvancedPassthroughSection
    )
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)

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
    def experiment_config(self) -> dict[str, Any]:
        return self.experiment.model_dump(exclude_none=True)

    @property
    def advanced_config(self) -> dict[str, Any]:
        return self.advanced.model_dump(exclude_none=True)


def load_submit_config[SubmitConfigT: BaseSubmitConfig](
    path: Path,
    *,
    config_class: type[SubmitConfigT],
    extra_sections: list[tuple[str, type[BaseModel]]],
    config_label: str,
    secrets_section_required: bool = False,
) -> SubmitConfigT:
    """Load and validate a submit TOML file into ``config_class``.

    ``extra_sections`` is a list of ``(section_name, pydantic_model)`` tuples
    describing the optional sections specific to this config (e.g. for training:
    ``[("training", _TrainingSection), ("sampling", _SamplingSection), ...]``).
    Each section name must match the corresponding field name on ``config_class``.

    ``[experiment]``, ``[advanced]``, ``[env]`` and ``[secrets]`` are handled
    here; subclasses only declare their own sections.
    """
    raw = read_toml_file(path)

    experiment_section = read_toml_table(raw, "experiment", path, required=True)
    for required_key in _EXPERIMENT_REQUIRED_KEYS:
        if required_key not in experiment_section:
            raise CLIError(
                f"Missing '{required_key}' in [experiment] section of {path}"
            )

    extra_data: dict[str, dict[str, Any]] = {
        name: read_toml_table(raw, name, path) for name, _ in extra_sections
    }
    advanced_section = read_toml_table(raw, "advanced", path)
    env_section = read_toml_table(raw, "env", path)
    secrets_list = read_toml_secrets(
        raw, path, section_required=secrets_section_required
    )

    allowed_sections = frozenset(
        {
            "experiment",
            "advanced",
            "env",
            "secrets",
            *(name for name, _ in extra_sections),
        }
    )

    section_specs: list[tuple[str, type[BaseModel], dict[str, Any]]] = [
        ("experiment", ExperimentSection, experiment_section),
        *((name, model, extra_data[name]) for name, model in extra_sections),
        ("advanced", AdvancedPassthroughSection, advanced_section),
    ]

    issues = [
        *collect_top_level_validation_issues(raw, allowed_sections=allowed_sections),
        *(
            issue
            for section_name, model_type, data in section_specs
            for issue in collect_section_validation_issues(
                section_name=section_name,
                model_type=model_type,
                data=data,
            )
        ),
        *validate_env_values(env=env_section),
    ]
    if issues:
        raise config_issues_error(issues=issues, config_label=config_label)

    parsed: dict[str, Any] = {}
    for section_name, model_type, data in section_specs:
        parsed[section_name] = parse_section(
            section_name=section_name,
            model_type=model_type,
            data=data,
            config_label=config_label,
        )

    env = {key: value for key, value in env_section.items() if isinstance(value, str)}
    validate_env_var_keys(env=env, path=path)
    validate_secret_names(secrets=secrets_list, env=env, path=path)

    return config_class(**parsed, env=env, secrets=secrets_list)


def build_submit_summary_rows(
    *,
    rollout: str,
    entrypoint: str,
    model: str,
    dataset: str,
    commit_sha: str | None,
) -> list[tuple[str, str]]:
    """Build the confirmation-table rows shared by ``train`` and ``eval`` submit."""
    rows: list[tuple[str, str]] = [
        ("Rollout", rollout),
        ("Entrypoint", entrypoint),
        ("Model", model),
        ("Dataset", dataset),
    ]
    if commit_sha:
        rows.append(("Commit", commit_sha))
    return rows


def build_env_table_rows(env: dict[str, str]) -> list[tuple[str, str]]:
    """Build (name, value) rows for the env vars table, sorted by name."""
    return [(name, value) for name, value in sorted(env.items())]


def build_secret_table_rows(
    secrets: list[str],
    *,
    user_secret_names: set[str],
    workspace_secret_names: set[str],
) -> list[tuple[str, str]]:
    """Build (name, scope) rows for the secrets table, sorted by name.

    A personal secret is labeled an override only when a workspace secret of
    the same name also exists; otherwise it is a personal-only secret.
    """
    rows: list[tuple[str, str]] = []
    for name in sorted(secrets):
        if name in user_secret_names:
            label = (
                "Personal (overrides workspace)"
                if name in workspace_secret_names
                else "Personal"
            )
        else:
            label = "Workspace"
        rows.append((name, label))
    return rows


def validate_env_values(
    *,
    env: dict[str, Any],
) -> list[dict[str, str]]:
    """Return issues for any [env] value that is not a string."""
    issues: list[dict[str, str]] = []
    for key, value in env.items():
        if not isinstance(value, str):
            issues.append(
                {
                    "key": f"env.{key}",
                    "message": f"must be a string, got {format_input(value)}",
                }
            )
    return issues


__all__ = [
    "ENV_VAR_NAME_RE",
    "RESERVED_ENV_PREFIX",
    "SECRET_NAME_RE",
    "AdvancedPassthroughSection",
    "BackendValidatedParamSection",
    "BaseSubmitConfig",
    "ExperimentSection",
    "build_env_table_rows",
    "build_secret_table_rows",
    "build_submit_summary_rows",
    "load_submit_config",
    "read_toml_file",
    "read_toml_secrets",
    "read_toml_table",
    "validate_env_values",
    "validate_env_var_keys",
    "validate_secret_names",
    "validate_workspace_rollout_paths",
]

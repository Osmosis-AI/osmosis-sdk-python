"""Shared building blocks for ``osmosis train submit`` and ``osmosis eval submit``."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import ErrorDetails

from osmosis_ai.cli.errors import CLIError

ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
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
    (e.g. ``"Training"`` or ``"Eval"``).
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
    secrets: dict[str, str],
    path: Path,
) -> None:
    """Reject invalid env-var names, reserved names, and overlap between sections."""
    for section_name, section in (("env", env), ("secrets", secrets)):
        for key in section:
            if not ENV_VAR_NAME_RE.match(key):
                raise CLIError(
                    f"Invalid env var name '{key}' in [{section_name}] of {path}: "
                    "must match ^[A-Z_][A-Z0-9_]*$"
                )
            if key.startswith(RESERVED_ENV_PREFIX):
                raise CLIError(
                    f"'{key}' in [{section_name}] of {path}: env var names starting "
                    f"with {RESERVED_ENV_PREFIX} are reserved by the platform; "
                    "choose a different name."
                )

    overlap = sorted(set(env) & set(secrets))
    if overlap:
        names = ", ".join(overlap)
        raise CLIError(
            f"Key(s) appear in both [env] and [secrets] of {path}: "
            f"{names}. Each env var name must come from exactly one section."
        )


def build_submit_summary_rows(
    *,
    rollout: str,
    entrypoint: str,
    model: str,
    dataset: str,
    commit_sha: str | None,
    env: dict[str, str],
    secrets: dict[str, str],
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
    if env:
        env_keys = ", ".join(sorted(env))
        rows.append((f"Rollout env ({len(env)})", env_keys))
    if secrets:
        secret_summary = ", ".join(
            f"{env_name}={secret_name}"
            for env_name, secret_name in sorted(secrets.items())
        )
        rows.append((f"Rollout secrets ({len(secrets)})", secret_summary))
    return rows


def validate_env_values(
    *,
    env: dict[str, Any],
    secrets: dict[str, Any],
) -> list[dict[str, str]]:
    """Return issues for any env/secret value that is not a string."""
    issues: list[dict[str, str]] = []
    for section_name, section in (("env", env), ("secrets", secrets)):
        for key, value in section.items():
            if not isinstance(value, str):
                issues.append(
                    {
                        "key": f"{section_name}.{key}",
                        "message": f"must be a string, got {format_input(value)}",
                    }
                )
    return issues


__all__ = [
    "ENV_VAR_NAME_RE",
    "RESERVED_ENV_PREFIX",
    "AdvancedPassthroughSection",
    "BackendValidatedParamSection",
    "build_submit_summary_rows",
    "collect_section_validation_issues",
    "collect_top_level_validation_issues",
    "config_issues_error",
    "format_config_issue",
    "format_field_path",
    "format_input",
    "parse_section",
    "read_toml_file",
    "read_toml_table",
    "validate_env_values",
    "validate_env_var_keys",
    "validate_workspace_rollout_paths",
    "validation_issue_to_config_issue",
]

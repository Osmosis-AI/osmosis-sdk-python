"""Shared orchestration for ``osmosis train submit`` and ``osmosis eval submit``.

Both submit flows follow the same script:

1. resolve the git/workspace context and validate the config path,
2. load + validate the TOML config,
3. render a confirmation table and prompt the user,
4. POST to the platform API,
5. return an ``OperationResult`` describing the new run.

The only meaningful differences are the literal strings, the loader, and the
shape of the API call / next-step suggestions. ``CloudSubmitSpec`` parametrises
those, and ``run_cloud_submit`` is the single implementation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import SubmitRunResult
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.shared_config import (
    BaseSubmitConfig,
    build_env_table_rows,
    build_secret_table_rows,
    build_submit_summary_rows,
)
from osmosis_ai.platform.cli.utils import (
    fetch_all_pages,
    print_remote_fetch_notice,
    require_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_rollout_backend,
    validate_workspace_directory_contract,
)

_MISSING_SECRET_RE = re.compile(r"Secret\(s\) not found: (.+)")


@dataclass(frozen=True)
class CloudSubmitSpec[ConfigT: BaseSubmitConfig]:
    """Describes the per-command differences between train / eval submit.

    Strings are plain literals. The four callables capture the rest:
    ``load_config`` builds ``ConfigT`` from the TOML path; ``validate_context``
    cross-checks the parsed config against the workspace directory layout;
    ``submit`` makes the API call (so each spec can pass its own set of
    config-section kwargs); ``build_next_steps`` produces the user-facing and
    structured suggestions shown alongside the success result.
    """

    config_dir: str
    """Workspace-relative directory the config must live under (e.g.
    ``configs/training``)."""

    command_label: str
    """Backticked command name used in path-related errors (e.g.
    ``"`osmosis train submit`"``)."""

    table_title: str
    """Title shown above the confirmation summary table."""

    confirm_prompt: str
    """Yes/no prompt shown before submitting."""

    status_message: str
    """Spinner status shown while the API call is in flight."""

    operation: str
    """``OperationResult.operation`` value (e.g. ``"train.submit"``)."""

    success_message_format: str
    """``str.format`` template with a ``{name}`` placeholder."""

    load_config: Callable[[Path], ConfigT]
    validate_context: Callable[[ConfigT, Path], None]
    submit: Callable[[OsmosisClient, ConfigT, Any, str], SubmitRunResult]
    build_next_steps: Callable[
        [SubmitRunResult, ConfigT],
        tuple[list[str], list[dict[str, Any]]],
    ]


def _fetch_secret_scopes(
    client: OsmosisClient, *, credentials: Any, git_identity: str
) -> tuple[set[str], set[str]] | None:
    """Return ``(workspace_names, personal_names)`` for the caller's workspace.

    Fetches both scopes so the submit summary can mirror the platform's
    resolution (personal preferred, override only when a workspace secret of
    the same name also exists) and detect missing secrets up front.

    Returns ``None`` on failure (network, auth) so the caller can fall back to
    a best-effort display instead of blocking the submit.
    """
    try:

        def _fetch(limit: int, offset: int) -> Any:
            return client.list_environment_secrets(
                limit=limit,
                offset=offset,
                scope="all",
                credentials=credentials,
                git_identity=git_identity,
            )

        secrets, _ = fetch_all_pages(_fetch, items_attr="environment_secrets")
        workspace = {s.name for s in secrets if s.scope == "workspace"}
        personal = {s.name for s in secrets if s.scope == "user"}
        return workspace, personal
    except Exception:
        return None


def _missing_secret_message(names: list[str]) -> str:
    """Build a fail-fast message for run-submit secrets that don't exist."""
    lines = [
        f"Could not find secret(s): {', '.join(names)}.",
        "",
        "Run the following to add them:",
    ]
    lines.extend(f"  osmosis secret set {name}" for name in names)
    lines.extend(
        [
            "",
            "Secrets default to personal scope. Use --scope workspace for secrets shared across the workspace.",
        ]
    )
    return "\n".join(lines)


def _enrich_missing_secret_error(
    exc: PlatformAPIError,
) -> PlatformAPIError | None:
    """If ``exc`` is a missing-secret 404, return a new error with actionable hints.

    Returns ``None`` when ``exc`` is unrelated so the caller can re-raise as-is.
    """
    if exc.status_code != 404:
        return None
    match = _MISSING_SECRET_RE.search(str(exc))
    if not match:
        return None

    names = [n.strip() for n in match.group(1).split(",")]
    platform_url = (exc.details or {}).get("platform_url")

    lines = [
        str(exc),
        "",
        "Run the following to add them:",
    ]
    for name in names:
        lines.append(f"  osmosis secret set {name}")
    lines.append("")
    lines.append(
        "Secrets default to personal scope. Use --scope workspace for secrets shared across the workspace."
    )
    if platform_url:
        lines.append(f"\nOr add them in the UI: {platform_url}")

    return PlatformAPIError(
        "\n".join(lines),
        exc.status_code,
        error_code=exc.error_code,
        field=exc.field,
        details=exc.details,
    )


def run_cloud_submit[ConfigT: BaseSubmitConfig](
    config_path: Path,
    *,
    yes: bool,
    spec: CloudSubmitSpec[ConfigT],
) -> OperationResult:
    """Run the shared submit flow for ``spec``."""
    context = require_git_workspace_directory_context()
    workspace_directory = context.workspace_directory
    validate_workspace_directory_contract(workspace_directory)

    config_path = Path(config_path)
    resolved_config_path = (
        config_path if config_path.is_absolute() else workspace_directory / config_path
    )
    ensure_workspace_directory_config_path(
        resolved_config_path,
        workspace_directory,
        config_dir=spec.config_dir,
        command_label=spec.command_label,
    )

    config = spec.load_config(resolved_config_path)
    spec.validate_context(config, workspace_directory)
    validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=spec.command_label,
    )

    summary_rows = build_submit_summary_rows(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        model=config.experiment_model_path,
        dataset=config.experiment_dataset,
        commit_sha=config.experiment_commit_sha,
        env=config.env,
        secrets=config.secrets,
    )

    console.table(
        [(label, console.escape(value)) for label, value in summary_rows],
        title=spec.table_title,
    )

    full_summary: list[tuple[str, str]] = list(summary_rows)

    if config.env:
        env_rows = build_env_table_rows(config.env)
        console.table(
            [(name, console.escape(value)) for name, value in env_rows],
            title=f"Env Vars ({len(env_rows)})",
            headers=("Name", "Value"),
        )
        full_summary.extend((f"env.{name}", value) for name, value in env_rows)

    if config.secrets:
        scopes = _fetch_secret_scopes(
            OsmosisClient(),
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
        if scopes is None:
            # Lookup failed — show names without a confident scope rather than
            # blocking the submit or mislabeling; the server still validates.
            secret_rows = [(name, "—") for name in sorted(config.secrets)]
        else:
            workspace_names, personal_names = scopes
            missing = sorted(
                {
                    name
                    for name in config.secrets
                    if name not in workspace_names and name not in personal_names
                }
            )
            if missing:
                raise CLIError(_missing_secret_message(missing))
            secret_rows = build_secret_table_rows(
                config.secrets,
                user_secret_names=personal_names,
                workspace_secret_names=workspace_names,
            )
        console.table(
            secret_rows,
            title=f"Secrets ({len(secret_rows)})",
            headers=("Name", "Scope"),
        )
        full_summary.extend((f"secret.{name}", scope) for name, scope in secret_rows)

    notes, warnings = print_remote_fetch_notice(
        workspace_directory,
        pinned_commit_sha=config.experiment_commit_sha,
    )

    require_confirmation(
        spec.confirm_prompt,
        yes=yes,
        summary=full_summary,
        notes=notes,
        warnings=warnings,
    )

    client = OsmosisClient()
    output = get_output_context()
    with output.status(spec.status_message):
        try:
            result = spec.submit(
                client, config, context.credentials, context.git_identity
            )
        except PlatformAPIError as exc:
            enriched = _enrich_missing_secret_error(exc)
            if enriched is not None:
                raise enriched from exc
            raise

    display_next_steps, next_steps_structured = spec.build_next_steps(result, config)

    return OperationResult(
        operation=spec.operation,
        status="success",
        resource={
            "id": result.id,
            "name": result.name,
            "status": result.status,
            "model_name": config.experiment_model_path,
            "dataset_name": config.experiment_dataset,
            "created_at": result.created_at,
            **({"url": result.platform_url} if result.platform_url else {}),
            **git_result_context(context),
            "config": {
                "rollout": config.experiment_rollout,
                "entrypoint": config.experiment_entrypoint,
                "model": config.experiment_model_path,
                "dataset": config.experiment_dataset,
                "commit_sha": config.experiment_commit_sha,
            },
        },
        message=spec.success_message_format.format(name=result.name),
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
    )


__all__ = [
    "CloudSubmitSpec",
    "run_cloud_submit",
]

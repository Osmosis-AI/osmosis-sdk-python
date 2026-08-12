"""Shared orchestration for cloud submit (train, eval, and benchmark).

Train and eval follow the same script via ``CloudSubmitSpec`` /
``run_cloud_submit``. Benchmark TOML is a different shape (no
``BaseSubmitConfig``, no ``commit_sha``), so it renders its own summary
tables and HLE warning, then joins the shared tail:
``prepare_submit_secrets`` → ``confirm_remote_fetch_and_post``.
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
from osmosis_ai.platform.api.models import WIRE_SCOPE_PERSONAL, SubmitRunResult
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.secret_resolution import resolve_run_secrets
from osmosis_ai.platform.cli.shared_config import (
    BaseSubmitConfig,
    build_env_table_rows,
    build_secret_table_rows,
    build_submit_summary_rows,
)
from osmosis_ai.platform.cli.utils import (
    fetch_environment_secrets,
    print_remote_fetch_notice,
    require_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_rollout_backend,
    validate_workspace_directory_contract,
)
from osmosis_ai.platform.cli.workspace_repo import check_pinned_commit

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
    submit: Callable[
        [OsmosisClient, ConfigT, Any, str, dict[str, str]], SubmitRunResult
    ]
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
    secrets = fetch_environment_secrets(
        client, scope="all", credentials=credentials, git_identity=git_identity
    )
    if secrets is None:
        return None
    workspace = {s.name for s in secrets if s.scope == "workspace"}
    personal = {s.name for s in secrets if s.scope == WIRE_SCOPE_PERSONAL}
    return workspace, personal


def _secret_add_hint_lines(
    names: list[str], *, platform_url: str | None = None
) -> list[str]:
    """Build the shared "add these secrets" hint tail shared by both error paths.

    Emits the "Run the following to add them:" line, one
    ``  osmosis secret set <name>`` per name, a blank-line separator, and the
    personal-scope guidance sentence. When ``platform_url`` is provided, appends
    a UI deep-link line (used only on the server-404 enrich path).
    """
    lines = ["Run the following to add them:"]
    lines.extend(f"  osmosis secret set {name}" for name in names)
    lines.extend(
        [
            "",
            "Secrets default to personal scope. Use --scope workspace for secrets shared across the workspace.",
        ]
    )
    if platform_url:
        lines.append(f"\nOr add them in the UI: {platform_url}")
    return lines


def _missing_secret_message(names: list[str]) -> str:
    """Build a fail-fast message for run-submit secrets that don't exist."""
    lines = [
        f"Could not find secret(s): {', '.join(names)}.",
        "",
    ]
    lines.extend(_secret_add_hint_lines(names))
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
    ]
    lines.extend(_secret_add_hint_lines(names, platform_url=platform_url))

    return PlatformAPIError(
        "\n".join(lines),
        exc.status_code,
        error_code=exc.error_code,
        field=exc.field,
        details=exc.details,
    )


def prepare_submit_secrets(
    *,
    prompt_names: list[str],
    required_names: list[str],
    secrets_file: str | None,
    credentials: Any,
    git_identity: str,
    full_summary: list[tuple[str, str]],
) -> dict[str, str]:
    """Resolve provided secrets, print the secrets table, extend ``full_summary``.

    ``prompt_names`` may be supplied at submit (file / env / interactive).
    ``required_names`` must exist as a stored record or a provided value.
    Train/eval pass the same list for both; benchmark prompts only
    ``[secrets].required`` while requiring every referenced secret.

    Returns ``{}`` without printing when ``required_names`` is empty.
    """
    if not required_names:
        return {}

    provided_secrets: dict[str, str] = {}
    scopes = _fetch_secret_scopes(
        OsmosisClient(),
        credentials=credentials,
        git_identity=git_identity,
    )
    if scopes is None:
        # Lookup failed — show names without a confident scope rather than
        # blocking the submit or mislabeling; the server still validates.
        secret_rows = [(name, "–") for name in sorted(required_names)]
    else:
        workspace_names, personal_names = scopes
        provided_secrets = resolve_run_secrets(
            names=list(prompt_names),
            secrets_file=secrets_file,
            stored_names=workspace_names | personal_names,
        )
        missing = sorted(
            {
                name
                for name in required_names
                if name not in workspace_names
                and name not in personal_names
                and name not in provided_secrets
            }
        )
        if missing:
            raise CLIError(_missing_secret_message(missing))
        stored_rows = build_secret_table_rows(
            [name for name in required_names if name not in provided_secrets],
            user_secret_names=personal_names,
            workspace_secret_names=workspace_names,
        )
        secret_rows = sorted(
            [*stored_rows, *((name, "Run") for name in provided_secrets)]
        )
    console.table(
        secret_rows,
        title=f"Secrets ({len(secret_rows)})",
        headers=("Name", "Scope"),
    )
    full_summary.extend((f"secret.{name}", scope) for name, scope in secret_rows)
    return provided_secrets


def confirm_remote_fetch_and_post[T](
    *,
    yes: bool,
    confirm_prompt: str,
    full_summary: list[tuple[str, str]],
    workspace_directory: Path,
    status_message: str,
    post: Callable[[], T],
    branch: str | None = None,
    pinned_commit_sha: str | None = None,
    extra_warnings: list[str] | None = None,
    provided_secrets: dict[str, str] | None = None,
    warn_on_missing_commit_sha: bool = True,
) -> T:
    """Shared tail: remote-fetch notice, confirmation, POST with secret-404 hints."""
    notes, warnings = print_remote_fetch_notice(
        workspace_directory,
        branch=branch,
        pinned_commit_sha=pinned_commit_sha,
        extra_warnings=extra_warnings,
        warn_on_missing_commit_sha=warn_on_missing_commit_sha,
    )
    require_confirmation(
        confirm_prompt,
        yes=yes,
        summary=full_summary,
        notes=notes,
        warnings=warnings,
    )
    output = get_output_context()
    with output.status(status_message):
        try:
            return post()
        except PlatformAPIError as exc:
            if provided_secrets:
                from osmosis_ai.platform.cli.secret_redact import (
                    redact_provided_secrets,
                )

                exc = redact_provided_secrets(exc, provided_secrets.values())
            enriched = _enrich_missing_secret_error(exc)
            if enriched is not None:
                raise enriched from None
            if provided_secrets:
                raise exc from None
            raise


def run_cloud_submit[ConfigT: BaseSubmitConfig](
    config_path: Path,
    *,
    yes: bool,
    spec: CloudSubmitSpec[ConfigT],
    secrets_file: str | None = None,
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
    backend_preflight_warnings = validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=spec.command_label,
    )

    # Preflight a pinned commit before doing any further work: a confirmed-bad
    # SHA would fail server-side after the platform clones the repo, so fail fast
    # with a clear message instead. Best-effort warnings are folded into the
    # remote-fetch notice below.
    commit_preflight_warnings: list[str] = []
    if config.experiment_commit_sha:
        commit_check = check_pinned_commit(
            workspace_directory=workspace_directory,
            git_identity=context.git_identity,
            commit_sha=config.experiment_commit_sha,
        )
        if commit_check.error:
            raise CLIError(commit_check.error)
        commit_preflight_warnings = list(commit_check.warnings)

    summary_rows = build_submit_summary_rows(
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        model=config.experiment_model_path,
        dataset=config.experiment_dataset,
        branch=config.experiment_branch,
        commit_sha=config.experiment_commit_sha,
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

    provided_secrets = prepare_submit_secrets(
        prompt_names=list(config.secrets),
        required_names=list(config.secrets),
        secrets_file=secrets_file,
        credentials=context.credentials,
        git_identity=context.git_identity,
        full_summary=full_summary,
    )

    def _post() -> SubmitRunResult:
        return spec.submit(
            OsmosisClient(),
            config,
            context.credentials,
            context.git_identity,
            provided_secrets,
        )

    result = confirm_remote_fetch_and_post(
        yes=yes,
        confirm_prompt=spec.confirm_prompt,
        full_summary=full_summary,
        workspace_directory=workspace_directory,
        status_message=spec.status_message,
        post=_post,
        branch=config.experiment_branch,
        pinned_commit_sha=config.experiment_commit_sha,
        extra_warnings=[*backend_preflight_warnings, *commit_preflight_warnings],
        provided_secrets=provided_secrets,
    )

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
                "branch": config.experiment_branch,
                "commit_sha": config.experiment_commit_sha,
            },
        },
        message=spec.success_message_format.format(name=result.name),
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
    )


__all__ = [
    "CloudSubmitSpec",
    "confirm_remote_fetch_and_post",
    "prepare_submit_secrets",
    "run_cloud_submit",
]

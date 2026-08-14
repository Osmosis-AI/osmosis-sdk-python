"""`osmosis eval run` -- local evaluation, driven by the shared eval TOML.

This layer owns everything CLI-shaped: workspace and login context, config
parsing, dataset resolution, confirmation, secret resolution, the progress
display, and the result envelope. Scheduling and durable state live in
``osmosis_ai.eval.local``, which holds no CLI dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError, CLIErrorCode
from osmosis_ai.cli.output import CommandResult, OperationResult
from osmosis_ai.cli.output.context import get_output_context
from osmosis_ai.cli.paths import parse_cli_path
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.cli.eval_config import (
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)
from osmosis_ai.platform.cli.secret_resolution import resolve_run_secrets
from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_workspace_directory_config_path,
    validate_rollout_backend,
    validate_workspace_directory_contract,
)

COMMAND_LABEL = "`osmosis eval run`"
EVAL_CONFIG_DIR = "configs/eval"
DEFAULT_OUTPUT_SUBPATH = (".osmosis", "evals")

#: Advanced flag default: the platform-hosted eval-proxy (D6).
DEFAULT_EVAL_PROXY_PATH = "/api/eval-proxy"


@dataclass(frozen=True)
class _Hooks:
    """Bridges the supervisor's callbacks onto CLI output primitives."""

    yes: bool
    secrets_file: str | None
    model_path: str

    def note(self, message: str) -> None:
        console.print(message, style="dim")

    def confirm_dispatch(self, *, pending: int, model_path: str) -> None:
        require_confirmation(
            f"{pending} rollouts x model {model_path} — continue?",
            yes=self.yes,
            default=False,
        )

    def resolve_secrets(self, names: Sequence[str]) -> dict[str, str]:
        if not names:
            return {}
        # Workflow secrets must resolve locally: a name existing in the platform
        # store does not satisfy them here (§7.3). Provider keys are the
        # opposite -- the eval-proxy injects those server-side.
        return resolve_run_secrets(
            names=list(names),
            secrets_file=self.secrets_file,
            stored_names=set(),
        )

    def progress(self, snapshot: Any) -> None:
        console.print(
            f"{snapshot.completed}/{snapshot.total} "
            f"pass_rate={snapshot.pass_rate:.2f} failed={snapshot.failed}",
            style="dim",
        )


def _missing_extra_error(exc: ModuleNotFoundError) -> CLIError:
    return CLIError(
        "Local evaluation requires optional dependencies. Install them with "
        '`pip install "osmosis-ai[eval-run]"` (Harbor backends: '
        '`pip install "osmosis-ai[eval-run,harbor]"`).',
        code=CLIErrorCode.VALIDATION,
        details={"missing_module": exc.name, "extra": "eval-run"},
    )


def _resolve_output_root(output: str | None, workspace_directory: Path) -> Path:
    if output is None:
        return workspace_directory.joinpath(*DEFAULT_OUTPUT_SUBPATH)
    return parse_cli_path(output, expand_user=True).path


def _numeric(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CLIError(f"evaluation.{field} must be a number, got {value!r}")
    return float(value)


def _integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CLIError(f"evaluation.{field} must be an integer, got {value!r}")
    return value


def run(
    config_path: Path,
    *,
    name: str | None = None,
    output: str | None = None,
    dataset_file: str | None = None,
    secrets_file: str | None = None,
    rows: str | None = None,
    fresh: bool = False,
    retry_failed: bool = False,
    max_in_flight: int | None = None,
    yes: bool = False,
    rollout_port: int | None = None,
    skip_llm_preflight: bool = False,
    verbose: bool = False,
    eval_proxy_url: str | None = None,
) -> CommandResult:
    """Run an evaluation locally against the workspace's rollout server."""
    import asyncio

    try:
        from osmosis_ai.eval.local.dataset import (
            DatasetCache,
            DatasetResolutionError,
            PlatformDatasetFetcher,
            default_dataset_cache_root,
            parse_row_selector,
            resolve_explicit_dataset_file,
            resolve_platform_dataset,
            select_rows,
        )
        from osmosis_ai.eval.local.runner import (
            EvalRunSpec,
            LocalEvalError,
            LocalEvalOptions,
            LocalEvalRunner,
        )
        from osmosis_ai.eval.local.state import LocalEvalStateError
    except ModuleNotFoundError as exc:
        raise _missing_extra_error(exc) from exc

    context = require_git_workspace_directory_context()
    workspace_directory = context.workspace_directory
    validate_workspace_directory_contract(workspace_directory)

    resolved_config_path = config_path.expanduser().resolve()
    ensure_workspace_directory_config_path(
        resolved_config_path,
        workspace_directory,
        config_dir=EVAL_CONFIG_DIR,
        command_label=COMMAND_LABEL,
    )
    config = load_eval_submit_config(resolved_config_path)
    validate_eval_submit_context_paths(config, workspace_directory)
    for warning in validate_rollout_backend(
        workspace_directory=workspace_directory,
        rollout=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        command_label=COMMAND_LABEL,
    ):
        console.print_warning(warning)

    evaluation = config.evaluation_config
    spec = EvalRunSpec(
        rollout_name=config.experiment_rollout,
        entrypoint=config.experiment_entrypoint,
        model_path=config.experiment_model_path,
        dataset_name=config.experiment_dataset,
        n=_integer(evaluation.get("n"), field="n") or 1,
        batch_size=_integer(evaluation.get("batch_size"), field="batch_size"),
        pass_threshold=_numeric(
            evaluation.get("pass_threshold"), field="pass_threshold"
        )
        or 1.0,
        agent_timeout_sec=_numeric(
            evaluation.get("agent_workflow_timeout_s"), field="agent_workflow_timeout_s"
        ),
        grader_timeout_sec=_numeric(
            evaluation.get("grader_timeout_s"), field="grader_timeout_s"
        ),
        env=dict(config.env),
        secret_names=tuple(config.secrets),
        branch=config.experiment_branch,
        commit_sha=config.experiment_commit_sha,
    )

    output_root = _resolve_output_root(output, workspace_directory)
    rollout_dir = workspace_directory / "rollouts" / spec.rollout_name

    try:
        if dataset_file is not None:
            dataset = resolve_explicit_dataset_file(
                parse_cli_path(dataset_file, expand_user=True).path
            )
        else:
            cache = DatasetCache(
                default_dataset_cache_root(), git_identity=context.git_identity
            )
            fetcher = PlatformDatasetFetcher(
                credentials=context.credentials, git_identity=context.git_identity
            )
            with get_output_context().status("Resolving dataset..."):
                dataset = resolve_platform_dataset(
                    spec.dataset_name,
                    cache=cache,
                    fetcher=fetcher,
                    on_event=lambda message: console.print(message, style="dim"),
                )
        row_selector = parse_row_selector(rows) if rows else None
        selection = select_rows(
            dataset.path,
            extension=dataset.extension,
            limit=_integer(evaluation.get("limit"), field="limit"),
            row_selector=row_selector,
        )
    except DatasetResolutionError as exc:
        raise CLIError(str(exc)) from exc

    runner = LocalEvalRunner(
        spec=spec,
        options=LocalEvalOptions(
            name=name,
            output=output_root,
            rows=rows,
            fresh=fresh,
            retry_failed=retry_failed,
            max_in_flight=max_in_flight,
            yes=yes,
            rollout_port=rollout_port,
            skip_llm_preflight=skip_llm_preflight,
            verbose=verbose,
        ),
        dataset=dataset,
        selection=selection,
        rollout_dir=rollout_dir,
        output_root=output_root,
        proxy_base_url=_resolve_proxy_base_url(eval_proxy_url),
        proxy_auth_token=_platform_bearer(context.credentials),
        hooks=_Hooks(yes=yes, secrets_file=secrets_file, model_path=spec.model_path),
        provenance=_provenance(workspace_directory, spec),
        config_stem=resolved_config_path.stem,
    )

    try:
        summary = asyncio.run(runner.run())
    except (LocalEvalError, LocalEvalStateError) as exc:
        raise CLIError(str(exc)) from exc
    except KeyboardInterrupt:
        raise CLIError(
            "Interrupted. Re-run the same command to resume the pending work items.",
            code=CLIErrorCode.VALIDATION,
        ) from None

    _print_failures(summary)
    return _result(summary, context=context, dataset_source=dataset.source)


def _resolve_proxy_base_url(override: str | None) -> str:
    """Advanced flag or the platform-derived hosted eval-proxy base (D6)."""
    if override:
        return override.rstrip("/")
    from osmosis_ai.platform.auth.config import get_platform_url

    return f"{get_platform_url().rstrip('/')}{DEFAULT_EVAL_PROXY_PATH}"


def _platform_bearer(credentials: Any) -> str:
    token = getattr(credentials, "access_token", None)
    if not token:
        raise CLIError(
            "Local evaluation requires an Osmosis login: the eval-proxy session "
            "is org-authenticated. Run `osmosis auth login`.",
            code=CLIErrorCode.AUTH_REQUIRED,
        )
    return str(token)


def _provenance(workspace_directory: Path, spec: Any) -> dict[str, Any]:
    """Record what actually executed. Never mutates the workspace (§5)."""
    from osmosis_ai.platform.cli.workspace_repo import summarize_local_git_state

    state = summarize_local_git_state(workspace_directory)
    provenance: dict[str, Any] = {
        "config_branch": spec.branch,
        "config_commit_sha": spec.commit_sha,
    }
    if state is not None:
        provenance.update(
            {
                "git_head": state.head_sha,
                "git_branch": state.branch,
                "git_dirty": state.is_dirty,
            }
        )
        if spec.commit_sha and state.head_sha != spec.commit_sha:
            console.print_warning(
                f"config pins commit {spec.commit_sha} but the workspace is at "
                f"{state.head_sha}; the local run executes the workspace as it is."
            )
        elif spec.branch and state.branch and state.branch != spec.branch:
            console.print_warning(
                f"config names branch {spec.branch!r} but the workspace is on "
                f"{state.branch!r}; the local run executes the workspace as it is."
            )
    return {key: value for key, value in provenance.items() if value is not None}


def _print_failures(summary: Any) -> None:
    """Print failed rows with zero-friction paths (§4.3)."""
    if not summary.failures:
        return
    console.separator("Failed rows")
    for failure in summary.failures:
        source = (
            ""
            if failure.source_row_index == failure.row_index
            else f" (source {failure.source_row_index})"
        )
        error = failure.error_type or "unknown"
        console.print(
            f"row {failure.row_index}{source} run {failure.run_index}: "
            f"error_type={error} -> {failure.rollout_dir}"
        )


def _result(summary: Any, *, context: Any, dataset_source: str) -> OperationResult:
    resource: dict[str, Any] = {
        "run_name": summary.run_name,
        "local_run_id": summary.local_run_id,
        "output_path": str(summary.run_dir),
        "dataset_source": dataset_source,
        "total_work_items": summary.total_work_items,
        "dispatched": summary.dispatched,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "resumed": summary.resumed,
        "cancelled": summary.cancelled,
        "metrics": summary.metrics,
        "failed_rows": [
            {
                "row_index": failure.row_index,
                "source_row_index": failure.source_row_index,
                "run_index": failure.run_index,
                "rollout_id": failure.rollout_id,
                "error_type": failure.error_type,
                "rollout_dir": str(failure.rollout_dir),
            }
            for failure in summary.failures
        ],
    }
    resource.update(git_result_context(context))
    incomplete = summary.cancelled or (
        summary.succeeded + summary.failed + summary.skipped < summary.total_work_items
    )
    next_steps = []
    if incomplete:
        next_steps.append(
            f"Resume: osmosis eval run <config> --name {summary.run_name}"
        )
    if summary.failed:
        next_steps.append(
            "Retry failures under the same name: "
            f"osmosis eval run <config> --name {summary.run_name} --retry-failed"
        )
    return OperationResult(
        operation="eval.run",
        status="partial" if incomplete else "success",
        resource=resource,
        message=(
            f"Evaluation run {'interrupted' if incomplete else 'finished'}: "
            f"{summary.run_dir}"
        ),
        display_next_steps=next_steps,
        exit_code=1 if incomplete else 0,
    )

"""Handler for the `osmosis quickstart` setup wizard.

    osmosis quickstart -> run_quickstart()

Every step is a check-then-fix against live state — CLI auth, workspace,
workspace repository, billing — so the command is idempotent and Ctrl+C is
always safe: re-running resumes from reality rather than from saved progress.
The wizard ends by asking what the user wants to do and handing them an agent
prompt for it.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from urllib.parse import quote

import typer

from osmosis_ai.cli.clipboard import copy_to_clipboard
from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
from osmosis_ai.cli.prompts import Choice, confirm, pause, select_list, text_input
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import QuickstartStatus, WorkspaceSummary
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
    load_credentials,
    save_credentials,
)
from osmosis_ai.platform.auth.config import get_platform_url
from osmosis_ai.platform.auth.credentials import ensure_keyring_available
from osmosis_ai.platform.auth.flow import device_login
from osmosis_ai.platform.auth.platform_client import revoke_cli_token
from osmosis_ai.platform.cli.workspace_directories import (
    forget_workspace_directory,
    recall_workspace_directory,
    remember_workspace_directory,
)
from osmosis_ai.platform.cli.workspace_directory_contract import (
    find_workspace_directory,
)
from osmosis_ai.platform.cli.workspace_repo import (
    get_local_git_remote_url,
    git_worktree_top_level,
    normalize_git_identity,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials

_DOCS_URL = "https://docs.osmosis.ai"
_MANUAL_SETUP_URL = f"{_DOCS_URL}/platform/onboarding#manual-setup"
_PLATFORM_QUICKSTART_DOCS_URL = f"{_DOCS_URL}/platform/quickstart"
_SDK_OVERVIEW_DOCS_URL = f"{_DOCS_URL}/sdk/overview"
_BENCHMARKS_DOCS_URL = f"{_DOCS_URL}/platform/benchmarks"

# The agent is told to use this skill by name, so it must match the skill as
# shipped in workspace-template.
_BENCHMARK_SKILL = "submit-benchmarks"

_REPO_POLL_SECONDS = 5.0
_REPO_WAIT_TIMEOUT_SECONDS = 900.0
_WAITING_FOR_REPO = (
    "waiting for repository... (Ctrl+C to exit; re-run 'osmosis quickstart' to resume)"
)
_CLONE_PATH_HINT = "(this folder will be created and become your workspace directory)"
_BILLING_WARNING = (
    "Paid features (training, evaluations, benchmarks, deployments, ...) "
    "need billing set up: "
)
_BILLING_PAUSE = "Press Enter to continue..."
_CHECK_WIDTH = 20

_INTENTS: tuple[tuple[str, str], ...] = (
    ("train", "Train a model for a task"),
    ("eval", "Run an evaluation on a dataset"),
    ("benchmark", "Run a benchmark"),
    ("explore", "Just exploring for now"),
)

_TASK_PROMPTS = {
    "train": (
        "I want to train a model for {task} in this Osmosis workspace. Start with "
        "the plan-training skill: read the workspace instructions, help me settle "
        "the dataset plan, and propose the next step before creating rollouts, "
        "running evaluation runs, or submitting a training run."
    ),
    "eval": (
        "I want to evaluate a model on {task} in this Osmosis workspace. Start with "
        "the plan-eval skill: read the workspace instructions, help me settle the "
        "rollout and evaluation config, and propose the next step before submitting "
        "an evaluation run."
    ),
}
_BENCHMARK_PROMPT = (
    "I want to run a benchmark in this Osmosis workspace. Start with the "
    f"{_BENCHMARK_SKILL} skill: read the workspace instructions, help me settle "
    "which benchmark, which tasks, and which agents to compare, and confirm the "
    "run size with me before submitting a benchmark run."
)


def _require_interactive_session() -> None:
    output = get_output_context()
    if output.format is OutputFormat.rich and output.interactive:
        return
    if output.format is not OutputFormat.rich:
        reason = (
            "osmosis quickstart asks questions and prints an agent prompt, so it "
            "requires interactive rich output and cannot run with --json or --plain"
        )
    else:
        reason = "osmosis quickstart needs an interactive terminal"
    raise CLIError(
        f"{reason}; set your workspace up manually instead: {_MANUAL_SETUP_URL}",
        code="INTERACTIVE_REQUIRED",
    )


def _cancel() -> NoReturn:
    console.print("Cancelled.", style="dim")
    raise typer.Exit(130)


def _check(label: str, detail: str) -> None:
    dots = "." * max(_CHECK_WIDTH - len(label), 1)
    console.print(f"* {label} {dots}  {console.escape(detail)}")


def _step(detail: str) -> None:
    console.print(f"  -> {console.escape(detail)}", style="dim")


def _platform_url(organization_name: str, path: str) -> str:
    return f"{get_platform_url()}/{quote(organization_name, safe='')}/{path}"


def _login() -> Credentials:
    ensure_keyring_available()
    result, credentials = device_login()
    try:
        save_credentials(credentials, recover_invalid_metadata=True)
    except Exception:
        try:
            revoke_cli_token(credentials)
        except Exception:
            console.print_warning(
                "The new device-login token could not be saved or revoked.",
                code="TOKEN_REVOKE_FAILED",
            )
        raise
    _step(f"authenticated as {result.user.email}")
    return credentials


def _resolve_credentials() -> Credentials:
    credentials = load_credentials()
    if credentials is not None and not credentials.is_expired():
        _check("CLI auth", f"ok ({credentials.user.email})")
        return credentials
    _check("CLI auth", "not logged in")
    return _login()


def _fetch_workspaces(
    client: OsmosisClient, credentials: Credentials
) -> tuple[list[WorkspaceSummary], Credentials]:
    """List workspaces, re-authenticating once if the saved session is stale."""
    output = get_output_context()
    try:
        with output.status("Loading workspaces..."):
            workspaces = client.list_workspaces(credentials=credentials)
    except AuthenticationExpiredError:
        _step("saved session is no longer valid")
        credentials = _login()
        with output.status("Loading workspaces..."):
            workspaces = client.list_workspaces(credentials=credentials)

    if not workspaces:
        raise CLIError(
            "This account has no workspaces. Create one at "
            f"{get_platform_url()}, then re-run 'osmosis quickstart'.",
            code="NOT_FOUND",
        )
    return workspaces, credentials


def _clone_identity(path: Path) -> str | None:
    """Return the normalized ``owner/repo`` of the clone at ``path``, if any."""
    remote_url = get_local_git_remote_url(path)
    if remote_url is None:
        return None
    try:
        return normalize_git_identity(remote_url).identity
    except CLIError:
        return None


def _workspace_by_name(
    workspaces: list[WorkspaceSummary], requested: str
) -> WorkspaceSummary:
    wanted = requested.strip().lower()
    for workspace in workspaces:
        if workspace.name.lower() == wanted:
            return workspace
    available = ", ".join(sorted(workspace.name for workspace in workspaces))
    raise CLIError(
        f"This account has no workspace named '{requested}'. Available: {available}.",
        code="NOT_FOUND",
    )


def _resolve_workspace(
    workspaces: list[WorkspaceSummary],
    start: Path,
    requested: str | None,
) -> tuple[WorkspaceSummary, Path | None]:
    """Pick the workspace, preferring the clone the wizard was run from."""
    local_clone = find_workspace_directory(start)
    identity = _clone_identity(local_clone) if local_clone is not None else None
    detected: WorkspaceSummary | None = None
    if identity is not None:
        for workspace in workspaces:
            connected = (workspace.connected_repo_full_name or "").lower()
            if connected and connected == identity:
                detected = workspace
                break

    if requested is not None:
        chosen = _workspace_by_name(workspaces, requested)
        _check("Workspace", chosen.name)
        if detected is not None and detected.id == chosen.id:
            return chosen, local_clone
        return chosen, None

    if detected is not None:
        _check("Workspace", detected.name)
        return detected, local_clone

    if len(workspaces) == 1:
        _check("Workspace", workspaces[0].name)
        return workspaces[0], None

    _check("Workspace", "not in a workspace directory")
    choice = select_list(
        "Which workspace?",
        items=[
            Choice(title=workspace.name, value=workspace.id) for workspace in workspaces
        ],
    )
    if choice is None:
        _cancel()
    return next(workspace for workspace in workspaces if workspace.id == choice), None


def _fetch_status(
    client: OsmosisClient, workspace: WorkspaceSummary, credentials: Credentials
) -> tuple[QuickstartStatus, Credentials]:
    """Read setup state, re-authenticating once if the saved session is stale."""
    output = get_output_context()
    try:
        with output.status("Checking workspace setup..."):
            status = client.get_quickstart_status(workspace.id, credentials=credentials)
    except AuthenticationExpiredError:
        _step("saved session is no longer valid")
        credentials = _login()
        with output.status("Checking workspace setup..."):
            status = client.get_quickstart_status(workspace.id, credentials=credentials)
    return status, credentials


def _wait_for_repository(
    client: OsmosisClient, workspace: WorkspaceSummary, credentials: Credentials
) -> tuple[QuickstartStatus, Credentials]:
    """Poll until an admin connects the workspace repository.

    The wait outlives a session, so an expiry mid-poll re-authenticates and
    resumes against the original deadline.
    """
    deadline = time.monotonic() + _REPO_WAIT_TIMEOUT_SECONDS
    output = get_output_context()
    while True:
        try:
            with output.status(_WAITING_FOR_REPO):
                while True:
                    if time.monotonic() >= deadline:
                        raise CLIError(
                            "Timed out waiting for the workspace repository. Re-run "
                            "'osmosis quickstart' once an admin has connected it at "
                            f"{_platform_url(workspace.name, 'integrations/git')}."
                        )
                    time.sleep(_REPO_POLL_SECONDS)
                    status = client.get_quickstart_status(
                        workspace.id, credentials=credentials
                    )
                    if status.repo_connected:
                        _step(f"detected {status.repo_full_name}")
                        return status, credentials
        except AuthenticationExpiredError:
            _step("saved session is no longer valid")
            credentials = _login()


def _clone_url(full_name: str, transport: str) -> str:
    if transport == "ssh":
        return f"git@github.com:{full_name}.git"
    return f"https://github.com/{full_name}.git"


def _run_git_clone(url: str, target: Path) -> None:
    if shutil.which("git") is None:
        raise CLIError(
            "git is required to clone your workspace repository. Install git, "
            "then re-run 'osmosis quickstart'.",
            code="NOT_FOUND",
        )
    output = get_output_context()
    with output.status(f"cloning into {console.escape(str(target))}..."):
        result = subprocess.run(
            ["git", "clone", url, str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        detail = lines[-1] if lines else f"git exited with code {result.returncode}"
        raise CLIError(f"Could not clone the workspace repository: {detail}")


def _resolve_clone_target(entered: str, start: Path) -> Path:
    candidate = Path(entered).expanduser()
    if not candidate.is_absolute():
        candidate = start / candidate
    return candidate.resolve()


def _is_empty_directory(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def _enclosing_repository(target: Path) -> Path | None:
    """Nearest worktree that would contain ``target`` (may not exist yet).

    git only answers for existing paths, so walk up to the nearest ancestor.
    """
    existing = target
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            return None
        existing = parent
    return git_worktree_top_level(existing)


def _nesting_accepted(target: Path) -> bool | None:
    """Confirm a destination inside another repository. None means cancelled."""
    enclosing = _enclosing_repository(target)
    if enclosing is None:
        return True
    console.print_warning(
        f"{target} is inside the git repository at {enclosing}. "
        "Cloning there puts a repository inside a repository."
    )
    return confirm(f"Clone into {target} anyway?", default=False)


def _clone_workspace_repo(full_name: str, *, start: Path) -> Path:
    """Prompt for a destination and leave a usable clone of ``full_name`` there."""
    default_path = f"./{full_name.split('/')[-1]}"
    while True:
        answer = text_input(
            f"Where should we clone {full_name}?",
            default=default_path,
            instruction=_CLONE_PATH_HINT,
        )
        if answer is None:
            _cancel()
        entered = answer.strip()
        if not entered:
            continue

        target = _resolve_clone_target(entered, start)
        if target.exists():
            if _clone_identity(target) == full_name.lower():
                _step(f"continuing from {target}")
                return target
            if not _is_empty_directory(target):
                console.print_warning(
                    f"{target} already exists and is not a clone of {full_name}. "
                    "Choose a different path."
                )
                continue

        accepted = _nesting_accepted(target)
        if accepted is None:
            _cancel()
        if not accepted:
            continue

        transport = select_list(
            "Clone over HTTPS or SSH?",
            items=[
                Choice(title="HTTPS", value="https"),
                Choice(title="SSH", value="ssh"),
            ],
        )
        if transport is None:
            _cancel()
        _step(f"cloning {full_name} into {target}")
        _run_git_clone(_clone_url(full_name, transport), target)
        _step(f"continuing from {target}")
        return target


def _remembered_clone(workspace_id: str, full_name: str) -> Path | None:
    """Return the clone recorded for this workspace, if it is still one."""
    path = recall_workspace_directory(workspace_id)
    if path is None:
        return None
    if _clone_identity(path) == full_name.lower():
        return path
    forget_workspace_directory(workspace_id)
    return None


def _ensure_repository(
    client: OsmosisClient,
    workspace: WorkspaceSummary,
    credentials: Credentials,
    status: QuickstartStatus,
    local_clone: Path | None,
    start: Path,
) -> tuple[Path, QuickstartStatus, Credentials]:
    if status.repo_connected:
        _check("Workspace repo", f"{status.repo_full_name}")
    else:
        _check("Workspace repo", "none connected yet")
        console.print()
        console.print("  A workspace admin needs to connect GitHub:")
        console.print_url(
            "  ", _platform_url(workspace.name, "integrations/git"), style="yellow"
        )
        console.print()
        status, credentials = _wait_for_repository(client, workspace, credentials)

    if local_clone is not None:
        _step(f"continuing from {local_clone}")
        remember_workspace_directory(workspace.id, local_clone)
        return local_clone, status, credentials

    full_name = status.repo_full_name
    if full_name is None:
        raise CLIError(
            "The connected workspace repository has no name. Check "
            f"{_platform_url(workspace.name, 'integrations/git')} and re-run "
            "'osmosis quickstart'.",
            code="PLATFORM_ERROR",
        )

    remembered = _remembered_clone(workspace.id, full_name)
    if remembered is not None:
        _step(f"continuing from {remembered}")
        return remembered, status, credentials

    clone_dir = _clone_workspace_repo(full_name, start=start)
    remember_workspace_directory(workspace.id, clone_dir)
    return clone_dir, status, credentials


def _report_billing(
    client: OsmosisClient,
    workspace: WorkspaceSummary,
    credentials: Credentials,
    status: QuickstartStatus,
) -> tuple[QuickstartStatus, Credentials]:
    """Report billing, re-reading it once so a card added at the pause counts."""
    if status.billing_ready:
        _check("Billing", "ok")
        return status, credentials

    _check("Billing", "no payment method on file")
    console.print_url(
        f"  ! {_BILLING_WARNING}",
        _platform_url(workspace.name, "billing"),
        style="yellow",
    )
    if not pause(_BILLING_PAUSE):
        _cancel()

    status, credentials = _fetch_status(client, workspace, credentials)
    if status.billing_ready:
        _check("Billing", "ok")
    else:
        _step("continuing without billing")
    return status, credentials


def _ask_intent() -> str:
    console.print()
    choice = select_list(
        "What do you want to do?",
        items=[Choice(title=label, value=intent) for intent, label in _INTENTS],
    )
    if choice is None:
        _cancel()
    return choice


def _validate_task(value: str) -> bool | str:
    return True if value.strip() else "Describe your task in a few words."


def _ask_task(intent: str) -> str | None:
    if intent not in _TASK_PROMPTS:
        return None
    answer = text_input("Describe your task:", validate=_validate_task)
    if answer is None:
        _cancel()
    return answer.strip()


def _agent_prompt(intent: str, task: str | None) -> str | None:
    if intent == "benchmark":
        return _BENCHMARK_PROMPT
    if task is None:
        return None
    return _TASK_PROMPTS[intent].format(task=task)


def _record_completion(
    client: OsmosisClient,
    workspace: WorkspaceSummary,
    credentials: Credentials,
    intent: str,
) -> bool:
    try:
        client.complete_quickstart(workspace.id, intent, credentials=credentials)
    except (AuthenticationExpiredError, PlatformAPIError, CLIError) as exc:
        console.print_warning(
            "Setup is done, but the platform was not notified that quickstart "
            f"completed ({exc}). Re-run 'osmosis quickstart' to record it."
        )
        return False
    return True


def _print_cd_hint(clone_dir: Path, start: Path) -> None:
    if start == clone_dir or clone_dir in start.parents:
        return
    try:
        target = clone_dir.relative_to(start)
    except ValueError:
        target = clone_dir
    console.print("  Run this next:")
    console.print(
        f"    cd {console.escape(shlex.quote(str(target)))}", style="bold cyan"
    )
    console.print()


def _print_handoff(
    *,
    intent: str,
    prompt: str | None,
    clone_dir: Path,
    workspace: WorkspaceSummary,
    start: Path,
) -> None:
    console.print()
    console.separator()
    location = console.escape(str(clone_dir))

    if prompt is None:
        console.print(f"  Setup complete. Your workspace directory is at {location}.")
        console.print()
        _print_cd_hint(clone_dir, start)
        console.print_url("  Quickstart guide: ", _PLATFORM_QUICKSTART_DOCS_URL)
        console.print_url("  Building rollouts: ", _SDK_OVERVIEW_DOCS_URL)
        console.separator()
        return

    clipboard = " (copied to clipboard)" if copy_to_clipboard(prompt) else ""
    console.print(
        "  Setup complete. Open your AI agent (Claude Code, Cursor, ...) in "
        f"{location} and paste the prompt below{clipboard}:",
        soft_wrap=True,
    )
    console.print()
    _print_cd_hint(clone_dir, start)
    console.print(f"  {prompt}", style="cyan", markup=False, soft_wrap=True)
    console.print()
    if intent == "benchmark":
        console.print_url(
            "  Or browse benchmarks in the platform: ",
            _platform_url(workspace.name, "benchmarks"),
        )
        console.print_url("  Benchmarks guide: ", _BENCHMARKS_DOCS_URL)
    console.separator()


def run_quickstart(
    *, cwd: Path | None = None, workspace_name: str | None = None
) -> OperationResult:
    """Verify local setup, then hand the user an agent prompt for their goal."""
    _require_interactive_session()
    start = (cwd or Path.cwd()).resolve()

    console.print()
    client = OsmosisClient()
    credentials = _resolve_credentials()
    workspaces, credentials = _fetch_workspaces(client, credentials)
    workspace, local_clone = _resolve_workspace(workspaces, start, workspace_name)

    status, credentials = _fetch_status(client, workspace, credentials)

    clone_dir, status, credentials = _ensure_repository(
        client, workspace, credentials, status, local_clone, start
    )
    status, credentials = _report_billing(client, workspace, credentials, status)

    intent = _ask_intent()
    task = _ask_task(intent)
    prompt = _agent_prompt(intent, task)
    recorded = _record_completion(client, workspace, credentials, intent)
    _print_handoff(
        intent=intent,
        prompt=prompt,
        clone_dir=clone_dir,
        workspace=workspace,
        start=start,
    )

    return OperationResult(
        operation="quickstart",
        status="success",
        resource={
            "workspace": {"id": workspace.id, "name": workspace.name},
            "repo": {
                "connected": status.repo_connected,
                "full_name": status.repo_full_name,
            },
            "workspace_directory": str(clone_dir),
            "billing_ready": status.billing_ready,
            "previously_completed": status.completed,
            "intent": intent,
            "task": task,
            "agent_prompt": prompt,
            "completion_recorded": recorded,
        },
    )


__all__ = ["run_quickstart"]

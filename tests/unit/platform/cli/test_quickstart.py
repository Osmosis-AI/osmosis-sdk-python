"""Tests for the `osmosis quickstart` wizard handler.

The agent prompts and the wizard's user-facing questions are a cross-repo
contract (workspace-template skills, docs), so they are asserted verbatim here.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

import osmosis_ai.platform.cli.quickstart as quickstart_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    override_output_context,
)
from osmosis_ai.platform.api.models import QuickstartStatus, WorkspaceSummary
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli.workspace_directories import (
    recall_workspace_directory,
    remember_workspace_directory,
)
from tests.unit.platform.cli.conftest import strip_ansi

# Captured at import so the spinner-rendering test can restore what the fixtures
# stub out and drive the real Rich spinner.
_REAL_OUTPUT_STATUS = OutputContext.status
_REAL_RUN_GIT_CLONE = quickstart_module._run_git_clone

PLATFORM_URL = "https://platform.osmosis.ai"
FULL_NAME = "acme/acme-workspace"
BILLING_PAUSE = "Press Enter to continue..."
WORKSPACE = WorkspaceSummary(
    id="ws-1",
    name="acme",
    connected_repo_full_name=FULL_NAME,
)
OTHER_WORKSPACE = WorkspaceSummary(
    id="ws-2",
    name="globex",
    connected_repo_full_name=None,
)
CREDENTIALS = SimpleNamespace(
    user=SimpleNamespace(email="allen@osmosis.ai"),
    is_expired=lambda: False,
)

CLONE_QUESTION = f"Where should we clone {FULL_NAME}?"
TRANSPORT_QUESTION = "Clone over HTTPS or SSH?"
INTENT_QUESTION = "What do you want to do?"
TASK_QUESTION = "Describe your task:"
WORKSPACE_QUESTION = "Which workspace?"

TRAIN_PROMPT = (
    "I want to train a model for support ticket triage in this Osmosis "
    "workspace. Start with the plan-training skill: read the workspace "
    "instructions, help me settle the dataset plan, and propose the next step "
    "before creating rollouts, running evaluation runs, or submitting a "
    "training run."
)
EVAL_PROMPT = (
    "I want to evaluate a model on grader accuracy in this Osmosis workspace. "
    "Start with the plan-eval skill: read the workspace instructions, help me "
    "settle the rollout and evaluation config, and propose the next step "
    "before submitting an evaluation run."
)
BENCHMARK_PROMPT = (
    "I want to run a benchmark in this Osmosis workspace. Start with the "
    "submit-benchmarks skill: read the workspace instructions, help me settle "
    "which benchmark, which tasks, and which agents to compare, and confirm the "
    "run size with me before submitting a benchmark run."
)


def _status(
    *,
    connected: bool = True,
    billing_ready: bool = True,
    completed: bool = False,
) -> QuickstartStatus:
    return QuickstartStatus(
        repo_connected=connected,
        repo_full_name=FULL_NAME if connected else None,
        billing_ready=billing_ready,
        completed=completed,
    )


class FakeClient:
    """Client double for the two quickstart endpoints plus the workspace list."""

    def __init__(
        self,
        workspaces: list[WorkspaceSummary],
        statuses: list[QuickstartStatus],
        complete_error: Exception | None,
        list_errors: list[Exception],
        status_errors: list[Exception | None],
    ) -> None:
        self._workspaces = workspaces
        self._statuses = statuses
        self._complete_error = complete_error
        self._list_errors = list_errors
        self._status_errors = status_errors
        self.status_calls: list[str] = []
        self.status_credentials: list[Any] = []
        self.completions: list[tuple[str, str]] = []
        self.credentials: list[Any] = []

    def list_workspaces(self, *, credentials: Any = None) -> list[WorkspaceSummary]:
        self.credentials.append(credentials)
        if self._list_errors:
            raise self._list_errors.pop(0)
        return list(self._workspaces)

    def get_quickstart_status(
        self, organization_id: str, *, credentials: Any = None
    ) -> QuickstartStatus:
        self.status_calls.append(organization_id)
        self.status_credentials.append(credentials)
        if self._status_errors:
            error = self._status_errors.pop(0)
            if error is not None:
                raise error
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]

    def complete_quickstart(
        self, organization_id: str, intent: str, *, credentials: Any = None
    ) -> None:
        if self._complete_error is not None:
            raise self._complete_error
        self.completions.append((organization_id, intent))


class Prompts:
    """Scripted answers keyed by question text; records every question asked."""

    def __init__(self) -> None:
        self.selects: dict[str, list[Any]] = {}
        self.texts: dict[str, list[str | None]] = {}
        self.confirms: dict[str, list[bool | None]] = {}
        self.pauses: list[bool] = []
        self.pause_calls: list[str] = []
        self.select_calls: list[tuple[str, list[Any]]] = []
        self.select_defaults: dict[str, Any] = {}
        self.text_calls: list[tuple[str, str, str | None]] = []
        self.confirm_calls: list[tuple[str, bool]] = []
        self.rejections: list[tuple[str, str, Any]] = []

    def select_list(self, message: str, items: Any, **kwargs: Any) -> Any:
        titles = [getattr(item, "title", item) for item in items]
        self.select_calls.append((message, titles))
        self.select_defaults[message] = kwargs.get("default")
        answers = self.selects.get(message)
        if not answers:
            pytest.fail(f"unexpected select prompt: {message}")
        return answers.pop(0)

    def pause(self, message: str) -> bool:
        self.pause_calls.append(message)
        if not self.pauses:
            pytest.fail(f"unexpected pause prompt: {message}")
        return self.pauses.pop(0)

    def confirm(self, message: str, *, default: bool = True) -> bool | None:
        self.confirm_calls.append((message, default))
        answers = self.confirms.get(message)
        if not answers:
            pytest.fail(f"unexpected confirm prompt: {message}")
        return answers.pop(0)

    def text_input(
        self,
        message: str,
        *,
        default: str = "",
        validate: Any = None,
        instruction: str | None = None,
    ) -> str | None:
        """Answer the prompt, re-asking while ``validate`` rejects the answer."""
        answers = self.texts.get(message)
        while True:
            self.text_calls.append((message, default, instruction))
            if not answers:
                pytest.fail(f"unexpected text prompt: {message}")
            answer = answers.pop(0)
            if answer is None or validate is None:
                return answer
            outcome = validate(answer)
            if outcome is True:
                return answer
            self.rejections.append((message, answer, outcome))


@pytest.fixture(autouse=True)
def status_messages(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Record spinner copy and keep an interactive rich session for the wizard."""
    messages: list[str] = []

    @contextmanager
    def _status(_self: OutputContext, message: str) -> Any:
        messages.append(message)
        yield

    monkeypatch.setattr(OutputContext, "status", _status)
    with override_output_context(format=OutputFormat.rich, interactive=True):
        yield messages


@pytest.fixture
def wizard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_directories.WORKSPACE_DIRECTORIES_FILE",
        tmp_path / "config" / "workspace-directories.json",
    )
    buffer = StringIO()
    # Console binds its warning stream to sys.stderr at construction time.
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    console = Console(file=buffer, force_terminal=False, width=400)
    prompts = Prompts()
    clone_calls: list[tuple[str, Path]] = []
    remotes: dict[Path, str] = {}
    worktrees: dict[Path, Path] = {}

    monkeypatch.setattr(quickstart_module, "console", console)
    monkeypatch.setattr(quickstart_module, "select_list", prompts.select_list)
    monkeypatch.setattr(quickstart_module, "text_input", prompts.text_input)
    monkeypatch.setattr(quickstart_module, "confirm", prompts.confirm)
    monkeypatch.setattr(quickstart_module, "pause", prompts.pause)
    monkeypatch.setattr(
        quickstart_module,
        "git_worktree_top_level",
        lambda path: worktrees.get(Path(path)),
    )
    monkeypatch.setattr(quickstart_module, "copy_to_clipboard", lambda _text: False)
    monkeypatch.setattr(quickstart_module, "get_platform_url", lambda: PLATFORM_URL)
    monkeypatch.setattr(
        quickstart_module, "find_workspace_directory", lambda _start: None
    )
    monkeypatch.setattr(
        quickstart_module,
        "get_local_git_remote_url",
        lambda path: remotes.get(Path(path)),
    )
    monkeypatch.setattr(quickstart_module, "load_credentials", lambda: CREDENTIALS)
    monkeypatch.setattr(
        quickstart_module,
        "device_login",
        lambda: pytest.fail("valid credentials must not trigger a login"),
    )
    monkeypatch.setattr(quickstart_module, "ensure_keyring_available", lambda: None)
    monkeypatch.setattr(
        quickstart_module,
        "save_device_credentials_or_revoke",
        lambda _creds: "keyring",
    )
    monkeypatch.setattr(
        quickstart_module,
        "_run_git_clone",
        lambda url, target: clone_calls.append((url, target)),
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    return SimpleNamespace(
        buffer=buffer,
        stderr=stderr,
        console=console,
        prompts=prompts,
        clone_calls=clone_calls,
        remotes=remotes,
        worktrees=worktrees,
        monkeypatch=monkeypatch,
    )


def _install_client(
    wizard: Any,
    *,
    workspaces: list[WorkspaceSummary] | None = None,
    statuses: list[QuickstartStatus] | None = None,
    complete_error: Exception | None = None,
    list_errors: list[Exception] | None = None,
    status_errors: list[Exception | None] | None = None,
) -> FakeClient:
    client = FakeClient(
        workspaces if workspaces is not None else [WORKSPACE],
        statuses if statuses is not None else [_status()],
        complete_error,
        list_errors if list_errors is not None else [],
        status_errors if status_errors is not None else [],
    )
    wizard.monkeypatch.setattr(quickstart_module, "OsmosisClient", lambda: client)
    return client


def _adopt_clone(wizard: Any, path: Path) -> None:
    """Make ``path`` look like an existing clone of the workspace repo."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    wizard.remotes[path.resolve()] = f"https://github.com/{FULL_NAME}.git"


def _out(wizard: Any) -> str:
    return strip_ansi(wizard.buffer.getvalue())


def _train_answers(wizard: Any) -> None:
    wizard.prompts.selects[INTENT_QUESTION] = ["train"]
    wizard.prompts.texts[TASK_QUESTION] = ["support ticket triage"]


def _intent_labels() -> list[str]:
    return [
        "Train a model for a task",
        "Run an evaluation on a dataset",
        "Run a benchmark",
        "Just exploring for now",
    ]


def _billing_wizard(
    wizard: Any, tmp_path: Path, *, statuses: list[QuickstartStatus]
) -> FakeClient:
    """A wizard with nothing left to set up except billing."""
    clone = tmp_path / "acme-workspace"
    _adopt_clone(wizard, clone)
    wizard.monkeypatch.setattr(
        quickstart_module, "find_workspace_directory", lambda _start: clone.resolve()
    )
    return _install_client(wizard, statuses=statuses)


def _cloned_wizard(wizard: Any, tmp_path: Path) -> None:
    """Put the wizard in a workspace clone so setup has nothing left to do."""
    clone = tmp_path / "acme-workspace"
    _adopt_clone(wizard, clone)
    wizard.monkeypatch.setattr(
        quickstart_module, "find_workspace_directory", lambda _start: clone.resolve()
    )
    _install_client(wizard)


# ── TTY guard ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fmt", "interactive", "cause"),
    [
        (OutputFormat.rich, False, "needs an interactive terminal"),
        (OutputFormat.json, True, "cannot run with --json or --plain"),
        (OutputFormat.plain, True, "cannot run with --json or --plain"),
    ],
)
def test_requires_an_interactive_terminal(
    wizard: Any, fmt: OutputFormat, interactive: bool, cause: str
) -> None:
    _install_client(wizard)

    with (
        override_output_context(format=fmt, interactive=interactive),
        pytest.raises(CLIError) as excinfo,
    ):
        quickstart_module.run_quickstart()

    assert excinfo.value.code == "INTERACTIVE_REQUIRED"
    assert cause in str(excinfo.value)
    assert "https://docs.osmosis.ai/platform/onboarding" in str(excinfo.value)


# ── Idempotent all-ok rerun ──────────────────────────────────────


def test_all_green_rerun_reads_as_a_diagnostic(wizard: Any, tmp_path: Path) -> None:
    clone = tmp_path / "acme-workspace"
    _adopt_clone(wizard, clone)
    wizard.monkeypatch.setattr(
        quickstart_module, "find_workspace_directory", lambda _start: clone.resolve()
    )
    client = _install_client(wizard, statuses=[_status(completed=True)])
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.clone_calls == []
    assert [message for message, _titles in wizard.prompts.select_calls] == [
        INTENT_QUESTION
    ]
    assert client.completions == [("ws-1", "train")]
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(clone.resolve())
    assert result.resource["intent"] == "train"
    assert result.resource["previously_completed"] is True
    assert result.resource["completion_recorded"] is True

    output = _out(wizard)
    assert "allen@osmosis.ai" in output
    assert "acme" in output
    assert FULL_NAME in output
    assert "need billing set up" not in output


def test_single_workspace_is_selected_without_a_prompt(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert WORKSPACE_QUESTION not in [
        message for message, _titles in wizard.prompts.select_calls
    ]


def test_a_folder_outside_a_workspace_says_so(wizard: Any, tmp_path: Path) -> None:
    _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    wizard.prompts.selects[WORKSPACE_QUESTION] = ["ws-1"]
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert "not in a workspace directory" in _out(wizard)


def test_several_workspaces_prompt_for_one(wizard: Any, tmp_path: Path) -> None:
    client = _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    wizard.prompts.selects[WORKSPACE_QUESTION] = ["ws-2"]
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    workspace_titles = dict(wizard.prompts.select_calls)[WORKSPACE_QUESTION]
    assert workspace_titles == ["acme", "globex"]
    assert client.completions == [("ws-2", "train")]
    assert result.resource is not None
    assert result.resource["workspace"] == {"id": "ws-2", "name": "globex"}


# ── Repository connection + clone ────────────────────────────────


def test_waits_for_the_repository_then_clones(
    wizard: Any, tmp_path: Path, status_messages: list[str]
) -> None:
    client = _install_client(
        wizard,
        statuses=[_status(connected=False), _status(connected=False), _status()],
    )
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert len(client.status_calls) == 3
    assert (
        "waiting for repository... "
        "(Ctrl+C to exit; re-run 'osmosis quickstart' to resume)" in status_messages
    )
    assert wizard.clone_calls == [
        (
            f"https://github.com/{FULL_NAME}.git",
            (tmp_path / "acme-workspace").resolve(),
        )
    ]

    output = _out(wizard)
    assert f"{PLATFORM_URL}/acme/integrations/git" in output
    assert f"detected {FULL_NAME}" in output


def test_clone_prompt_echoes_the_resolved_path_and_hint(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["ssh"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    target = (tmp_path / "acme-workspace").resolve()
    assert wizard.prompts.text_calls[0] == (
        CLONE_QUESTION,
        "./acme-workspace",
        "(this folder will be created and become your workspace directory)",
    )
    assert f"cloning {FULL_NAME} into {target}" in _out(wizard)
    assert wizard.clone_calls == [(f"git@github.com:{FULL_NAME}.git", target)]


def test_adopts_an_existing_clone_at_the_requested_path(
    wizard: Any, tmp_path: Path
) -> None:
    existing = tmp_path / "somewhere-else"
    _adopt_clone(wizard, existing)
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./somewhere-else"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.clone_calls == []
    assert TRANSPORT_QUESTION not in [
        message for message, _titles in wizard.prompts.select_calls
    ]
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(existing.resolve())


def test_reprompts_when_the_path_holds_a_different_repo(
    wizard: Any, tmp_path: Path
) -> None:
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / ".git").mkdir()
    wizard.remotes[taken.resolve()] = "https://github.com/other/repo.git"
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./taken", "./fresh"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert len(wizard.prompts.text_calls) == 3  # clone path twice, then the task
    assert wizard.clone_calls == [
        (f"https://github.com/{FULL_NAME}.git", (tmp_path / "fresh").resolve())
    ]


def test_clones_into_an_existing_empty_directory(wizard: Any, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./empty"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.clone_calls == [
        (f"https://github.com/{FULL_NAME}.git", empty.resolve())
    ]
    assert len(wizard.prompts.text_calls) == 2  # clone path once, then the task
    assert "already exists" not in strip_ansi(wizard.stderr.getvalue())
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(empty.resolve())


def test_clone_spinner_survives_a_bracketed_destination(
    wizard: Any, tmp_path: Path
) -> None:
    """Render the real spinner: a path Rich would read as markup must not crash."""
    wizard.monkeypatch.setattr(OutputContext, "status", _REAL_OUTPUT_STATUS)
    wizard.monkeypatch.setattr(quickstart_module, "_run_git_clone", _REAL_RUN_GIT_CLONE)
    wizard.monkeypatch.setattr(
        quickstart_module.shutil, "which", lambda _name: "/usr/bin/git"
    )
    commands: list[list[str]] = []

    def _run(command: list[str], **_kwargs: Any) -> Any:
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    wizard.monkeypatch.setattr(quickstart_module.subprocess, "run", _run)
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./weird[/x]dir"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    target = (tmp_path / "weird[" / "x]dir").resolve()
    assert commands == [
        ["git", "clone", f"https://github.com/{FULL_NAME}.git", str(target)]
    ]
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(target)


def _nesting_question(target: Path) -> str:
    return f"Clone into {target} anyway?"


def test_a_nested_clone_target_is_questioned_and_can_be_declined(
    wizard: Any, tmp_path: Path
) -> None:
    wizard.worktrees[tmp_path] = tmp_path
    outside = tmp_path.parent / "outside-any-repo"
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace", str(outside)]
    wizard.prompts.confirms[
        _nesting_question((tmp_path / "acme-workspace").resolve())
    ] = [False]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.confirm_calls == [
        (_nesting_question((tmp_path / "acme-workspace").resolve()), False)
    ]
    assert wizard.clone_calls == [
        (f"https://github.com/{FULL_NAME}.git", outside.resolve())
    ]
    warning = strip_ansi(wizard.stderr.getvalue())
    assert f"is inside the git repository at {tmp_path}" in warning


def test_a_nested_clone_target_is_used_when_confirmed(
    wizard: Any, tmp_path: Path
) -> None:
    wizard.worktrees[tmp_path] = tmp_path
    nested = (tmp_path / "acme-workspace").resolve()
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.confirms[_nesting_question(nested)] = [True]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.clone_calls == [(f"https://github.com/{FULL_NAME}.git", nested)]


def test_cancelling_the_nesting_question_exits_130(wizard: Any, tmp_path: Path) -> None:
    wizard.worktrees[tmp_path] = tmp_path
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.confirms[
        _nesting_question((tmp_path / "acme-workspace").resolve())
    ] = [None]

    with pytest.raises(typer.Exit) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path)

    assert excinfo.value.exit_code == 130


def test_a_clone_target_outside_a_repo_is_not_questioned(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.confirm_calls == []


def test_adopting_a_clone_inside_a_repo_is_not_questioned(
    wizard: Any, tmp_path: Path
) -> None:
    """An existing clone is a resume, not a new nested checkout."""
    wizard.worktrees[tmp_path] = tmp_path
    existing = tmp_path / "acme-workspace"
    _adopt_clone(wizard, existing)
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.confirm_calls == []
    assert wizard.clone_calls == []
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(existing.resolve())


def test_a_remembered_clone_is_reused_from_anywhere(
    wizard: Any, tmp_path: Path
) -> None:
    remembered = tmp_path / "elsewhere" / "acme-workspace"
    _adopt_clone(wizard, remembered)
    remember_workspace_directory("ws-1", remembered)
    _install_client(wizard)
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.clone_calls == []
    assert [message for message, _default, _hint in wizard.prompts.text_calls] == [
        TASK_QUESTION
    ]
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(remembered.resolve())
    assert f"continuing from {remembered.resolve()}" in _out(wizard)


def test_a_stale_remembered_clone_falls_back_to_cloning(
    wizard: Any, tmp_path: Path
) -> None:
    """A clone the user has since deleted must not be offered as usable."""
    remember_workspace_directory("ws-1", tmp_path / "deleted-clone")
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    fresh = (tmp_path / "acme-workspace").resolve()
    assert wizard.clone_calls == [(f"https://github.com/{FULL_NAME}.git", fresh)]
    assert recall_workspace_directory("ws-1") == fresh


def test_the_surrounding_clone_wins_over_a_remembered_one(
    wizard: Any, tmp_path: Path
) -> None:
    stale = tmp_path / "older-clone"
    _adopt_clone(wizard, stale)
    remember_workspace_directory("ws-1", stale)
    here = tmp_path / "acme-workspace"
    _adopt_clone(wizard, here)
    wizard.monkeypatch.setattr(
        quickstart_module, "find_workspace_directory", lambda _start: here.resolve()
    )
    _install_client(wizard)
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=here)

    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(here.resolve())
    assert recall_workspace_directory("ws-1") == here.resolve()


def test_a_fresh_clone_is_remembered(wizard: Any, tmp_path: Path) -> None:
    _install_client(wizard)
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert recall_workspace_directory("ws-1") == (tmp_path / "acme-workspace").resolve()


def _inside_clone(wizard: Any, tmp_path: Path) -> Path:
    """Run the wizard from inside a clone of the first workspace's repo."""
    clone = tmp_path / "acme-workspace"
    _adopt_clone(wizard, clone)
    wizard.monkeypatch.setattr(
        quickstart_module,
        "find_workspace_directory",
        lambda _start: clone.resolve(),
    )
    return clone


def test_the_surrounding_clone_is_used_without_a_picker(
    wizard: Any, tmp_path: Path
) -> None:
    """Standing in a workspace clone answers the question, so don't ask it."""
    clone = _inside_clone(wizard, tmp_path)
    client = _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=clone / "configs")

    assert wizard.prompts.select_calls == [(INTENT_QUESTION, _intent_labels())]
    assert wizard.clone_calls == []
    assert client.status_calls == ["ws-1"]
    assert result.resource is not None
    assert result.resource["workspace"] == {"id": "ws-1", "name": "acme"}
    assert result.resource["workspace_directory"] == str(clone.resolve())
    output = _out(wizard)
    assert "Workspace" in output
    assert f"continuing from {clone.resolve()}" in output
    assert "this clone" not in output


def test_the_workspace_flag_overrides_the_surrounding_clone(
    wizard: Any, tmp_path: Path
) -> None:
    """Naming another workspace abandons the clone the wizard was run from."""
    _inside_clone(wizard, tmp_path)
    client = _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    wizard.prompts.texts[CLONE_QUESTION] = ["./globex-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path, workspace_name="globex")

    assert WORKSPACE_QUESTION not in dict(wizard.prompts.select_calls)
    assert client.status_calls == ["ws-2"]
    assert wizard.clone_calls == [
        (
            f"https://github.com/{FULL_NAME}.git",
            (tmp_path / "globex-workspace").resolve(),
        )
    ]
    assert result.resource is not None
    assert result.resource["workspace"] == {"id": "ws-2", "name": "globex"}


def test_the_workspace_flag_can_name_the_surrounding_clone(
    wizard: Any, tmp_path: Path
) -> None:
    clone = _inside_clone(wizard, tmp_path)
    _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path, workspace_name="acme")

    assert wizard.clone_calls == []
    assert result.resource is not None
    assert result.resource["workspace_directory"] == str(clone.resolve())


def test_the_workspace_flag_replaces_the_picker(wizard: Any, tmp_path: Path) -> None:
    client = _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])
    wizard.prompts.texts[CLONE_QUESTION] = ["./globex-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path, workspace_name="GLOBEX")

    assert WORKSPACE_QUESTION not in dict(wizard.prompts.select_calls)
    assert client.status_calls == ["ws-2"]


def test_an_unknown_workspace_name_lists_the_real_ones(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard, workspaces=[WORKSPACE, OTHER_WORKSPACE])

    with pytest.raises(CLIError) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path, workspace_name="nope")

    assert excinfo.value.code == "NOT_FOUND"
    assert "acme" in str(excinfo.value)
    assert "globex" in str(excinfo.value)


def test_billing_pauses_with_a_link_and_can_be_skipped(
    wizard: Any, tmp_path: Path
) -> None:
    _billing_wizard(wizard, tmp_path, statuses=[_status(billing_ready=False)])
    wizard.prompts.pauses = [True]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    output = _out(wizard)
    assert (
        "Paid features (training, evaluations, benchmarks, deployments, ...) "
        f"need billing set up: {PLATFORM_URL}/acme/billing" in output
    )
    assert wizard.prompts.pause_calls == [BILLING_PAUSE]
    assert "continuing without billing" in output
    assert TRAIN_PROMPT in output
    assert result.resource is not None
    assert result.resource["billing_ready"] is False


def test_billing_is_rechecked_after_the_pause(wizard: Any, tmp_path: Path) -> None:
    client = _billing_wizard(
        wizard, tmp_path, statuses=[_status(billing_ready=False), _status()]
    )
    wizard.prompts.pauses = [True]
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert client.status_calls == ["ws-1", "ws-1"]
    output = _out(wizard)
    assert "Billing" in output
    assert "continuing without billing" not in output
    assert result.resource is not None
    assert result.resource["billing_ready"] is True


def test_a_ready_billing_account_is_not_paused(wizard: Any, tmp_path: Path) -> None:
    _billing_wizard(wizard, tmp_path, statuses=[_status()])
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.pause_calls == []


def test_cancelling_the_billing_pause_exits_130(wizard: Any, tmp_path: Path) -> None:
    _billing_wizard(wizard, tmp_path, statuses=[_status(billing_ready=False)])
    wizard.prompts.pauses = [False]

    with pytest.raises(typer.Exit) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path)

    assert excinfo.value.exit_code == 130
    assert "Cancelled." in _out(wizard)


# ── Intent, task, and the agent prompts ──────────────────────────


def test_intent_labels_are_verbatim(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = ["explore"]

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert dict(wizard.prompts.select_calls)[INTENT_QUESTION] == _intent_labels()


def test_train_intent_prints_the_frozen_prompt(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert result.resource is not None
    assert result.resource["agent_prompt"] == TRAIN_PROMPT
    assert TRAIN_PROMPT in _out(wizard)


def test_eval_intent_prints_the_frozen_prompt(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = ["eval"]
    wizard.prompts.texts[TASK_QUESTION] = ["grader accuracy"]

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert result.resource is not None
    assert result.resource["agent_prompt"] == EVAL_PROMPT
    assert EVAL_PROMPT in _out(wizard)


def test_benchmark_intent_skips_the_task_question(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = ["benchmark"]

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.text_calls == []
    assert result.resource is not None
    assert result.resource["agent_prompt"] == BENCHMARK_PROMPT
    output = _out(wizard)
    assert BENCHMARK_PROMPT in output
    assert f"{PLATFORM_URL}/acme/benchmarks" in output
    assert "https://docs.osmosis.ai/platform/benchmarks" in output


def test_explore_intent_prints_docs_links_only(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = ["explore"]

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.text_calls == []
    assert result.resource is not None
    assert result.resource["agent_prompt"] is None
    output = _out(wizard)
    clone = tmp_path / "acme-workspace"
    assert f"Your workspace directory is at {clone.resolve()}" in output
    assert "https://docs.osmosis.ai/platform/quickstart" in output
    assert "https://docs.osmosis.ai/sdk/overview" in output
    assert "paste the prompt below" not in output
    assert "workspace clone" not in output


def test_handoff_points_the_shell_at_the_clone(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert "cd acme-workspace" in _out(wizard)


def test_handoff_omits_the_cd_from_inside_the_clone(
    wizard: Any, tmp_path: Path
) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path / "acme-workspace")

    assert "cd acme-workspace" not in _out(wizard)


def test_a_blank_task_is_rejected_until_described(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = ["train"]
    wizard.prompts.texts[TASK_QUESTION] = ["", "   ", "support ticket triage"]

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert wizard.prompts.rejections == [
        (TASK_QUESTION, "", "Describe your task in a few words."),
        (TASK_QUESTION, "   ", "Describe your task in a few words."),
    ]
    assert result.resource is not None
    assert result.resource["task"] == "support ticket triage"
    assert result.resource["agent_prompt"] == TRAIN_PROMPT


def test_cancelling_the_intent_prompt_exits_130(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    wizard.prompts.selects[INTENT_QUESTION] = [None]

    with pytest.raises(typer.Exit) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path)

    assert excinfo.value.exit_code == 130
    assert "Cancelled." in _out(wizard)


def test_prompt_is_copied_to_the_clipboard_when_available(
    wizard: Any, tmp_path: Path
) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)
    copied: list[str] = []
    wizard.monkeypatch.setattr(
        quickstart_module,
        "copy_to_clipboard",
        lambda text: bool(copied.append(text)) or True,
    )

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert copied == [TRAIN_PROMPT]
    assert "paste the prompt below (copied to clipboard):" in _out(wizard)


def test_clipboard_failure_is_silent(wizard: Any, tmp_path: Path) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)

    quickstart_module.run_quickstart(cwd=tmp_path)

    output = _out(wizard)
    assert "paste the prompt below:" in output
    assert "(copied to clipboard)" not in output


# ── Completion + auth repair ─────────────────────────────────────


def test_completion_failure_warns_and_still_prints_the_prompt(
    wizard: Any, tmp_path: Path
) -> None:
    _cloned_wizard(wizard, tmp_path)
    _install_client(wizard, complete_error=PlatformAPIError("Connection error: down"))
    _train_answers(wizard)

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert TRAIN_PROMPT in _out(wizard)
    assert result.resource is not None
    assert result.resource["completion_recorded"] is False
    warning = strip_ansi(wizard.stderr.getvalue())
    assert "was not notified" in warning
    assert "osmosis quickstart" in warning


def test_an_expired_session_relogs_in_and_retries_the_workspace_list(
    wizard: Any, tmp_path: Path
) -> None:
    _cloned_wizard(wizard, tmp_path)
    client = _install_client(
        wizard, list_errors=[AuthenticationExpiredError("session has expired")]
    )
    _train_answers(wizard)
    new_credentials = SimpleNamespace(
        user=SimpleNamespace(email="new@osmosis.ai"),
        is_expired=lambda: False,
    )
    logins: list[str] = []
    saved: list[Any] = []

    def _device_login() -> Any:
        logins.append("device")
        return (
            SimpleNamespace(user=SimpleNamespace(email="new@osmosis.ai")),
            new_credentials,
        )

    wizard.monkeypatch.setattr(quickstart_module, "device_login", _device_login)
    wizard.monkeypatch.setattr(
        quickstart_module,
        "save_device_credentials_or_revoke",
        lambda creds: saved.append(creds),
    )

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert logins == ["device"]
    assert saved == [new_credentials]
    assert client.credentials == [CREDENTIALS, new_credentials]
    assert result.resource is not None
    assert result.resource["intent"] == "train"

    output = _out(wizard)
    assert "saved session is no longer valid" in output
    assert "authenticated as new@osmosis.ai" in output


def test_a_session_expiring_mid_wait_relogs_in_and_keeps_polling(
    wizard: Any, tmp_path: Path
) -> None:
    client = _install_client(
        wizard,
        statuses=[_status(connected=False), _status()],
        status_errors=[None, AuthenticationExpiredError("session has expired")],
    )
    wizard.prompts.texts[CLONE_QUESTION] = ["./acme-workspace"]
    wizard.prompts.selects[TRANSPORT_QUESTION] = ["https"]
    _train_answers(wizard)
    new_credentials = SimpleNamespace(
        user=SimpleNamespace(email="new@osmosis.ai"),
        is_expired=lambda: False,
    )
    logins: list[str] = []

    def _device_login() -> Any:
        logins.append("device")
        return (
            SimpleNamespace(user=SimpleNamespace(email="new@osmosis.ai")),
            new_credentials,
        )

    wizard.monkeypatch.setattr(quickstart_module, "device_login", _device_login)
    wizard.monkeypatch.setattr(
        quickstart_module, "save_device_credentials_or_revoke", lambda _c: None
    )

    result = quickstart_module.run_quickstart(cwd=tmp_path)

    assert logins == ["device"]
    assert client.status_credentials == [CREDENTIALS, CREDENTIALS, new_credentials]
    assert client.completions == [(WORKSPACE.id, "train")]
    assert result.resource is not None

    output = _out(wizard)
    assert "saved session is no longer valid" in output
    assert f"detected {FULL_NAME}" in output


def test_waiting_for_the_repository_gives_up_eventually(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard, statuses=[_status(connected=False)])
    clock = iter([0.0, 600.0, 1200.0, 1800.0])
    wizard.monkeypatch.setattr("time.monotonic", lambda: next(clock))

    with pytest.raises(CLIError) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path)

    assert "Timed out waiting for the workspace repository" in str(excinfo.value)
    assert f"{PLATFORM_URL}/acme/integrations/git" in str(excinfo.value)


def test_an_account_without_workspaces_points_at_the_platform(
    wizard: Any, tmp_path: Path
) -> None:
    _install_client(wizard, workspaces=[])

    with pytest.raises(CLIError) as excinfo:
        quickstart_module.run_quickstart(cwd=tmp_path)

    assert excinfo.value.code == "NOT_FOUND"
    assert PLATFORM_URL in str(excinfo.value)
    assert wizard.prompts.select_calls == []


def test_missing_credentials_trigger_the_browser_login(
    wizard: Any, tmp_path: Path
) -> None:
    _cloned_wizard(wizard, tmp_path)
    _train_answers(wizard)
    new_credentials = SimpleNamespace(
        user=SimpleNamespace(email="new@osmosis.ai"),
        is_expired=lambda: False,
    )
    saved: list[Any] = []
    wizard.monkeypatch.setattr(quickstart_module, "load_credentials", lambda: None)
    wizard.monkeypatch.setattr(
        quickstart_module,
        "device_login",
        lambda: (
            SimpleNamespace(user=SimpleNamespace(email="new@osmosis.ai")),
            new_credentials,
        ),
    )
    wizard.monkeypatch.setattr(
        quickstart_module,
        "save_device_credentials_or_revoke",
        lambda creds: saved.append(creds),
    )

    quickstart_module.run_quickstart(cwd=tmp_path)

    assert saved == [new_credentials]
    assert "new@osmosis.ai" in _out(wizard)


def test_quickstart_login_uses_shared_save_or_revoke_helper(wizard: Any) -> None:
    credentials = SimpleNamespace(user=SimpleNamespace(email="new@osmosis.ai"))
    wizard.monkeypatch.setattr(
        quickstart_module,
        "device_login",
        lambda: (SimpleNamespace(user=credentials.user), credentials),
    )
    saved: list[Any] = []
    wizard.monkeypatch.setattr(
        quickstart_module,
        "save_device_credentials_or_revoke",
        lambda token: saved.append(token),
    )

    assert quickstart_module._login() is credentials
    assert saved == [credentials]


def test_quickstart_recovers_invalid_credential_metadata(wizard: Any) -> None:
    credentials = SimpleNamespace(user=SimpleNamespace(email="new@osmosis.ai"))
    wizard.monkeypatch.setattr(
        quickstart_module,
        "load_credentials",
        lambda: (_ for _ in ()).throw(
            CLIError("invalid metadata", code="CREDENTIALS_PARSE_FAILED")
        ),
    )
    logins: list[bool] = []
    wizard.monkeypatch.setattr(
        quickstart_module,
        "_login",
        lambda: logins.append(True) or credentials,
    )

    assert quickstart_module._resolve_credentials() is credentials
    assert logins == [True]

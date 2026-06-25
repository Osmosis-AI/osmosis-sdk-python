"""train submit must not fail preflight when rollout deps aren't installed locally.

Mirror of test_eval_submit_preflight: train and eval share `run_cloud_submit`,
so each covers the `validate_rollout_backend` seam from its own command's side.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.shared_submit as shared_submit_module
import osmosis_ai.platform.cli.train as train_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult
from osmosis_ai.platform.api.models import SubmitRunResult
from tests.unit.submit_preflight_helpers import disable_commit_preflight

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
FAKE_CREDENTIALS = object()


def _find_temp_workspace_directory(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / ".osmosis").is_dir()
            and (candidate / "configs" / "training").is_dir()
            and (candidate / "rollouts").is_dir()
        ):
            return candidate
    return None


def _make_workspace_directory(root: Path, *, main_py: str) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)], check=True, capture_output=True
    )
    for rel_path in (
        ".osmosis",
        "rollouts/calculator",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / "calculator" / "main.py").write_text(main_py, encoding="utf-8")
    return root


def _write_train_config(path: Path) -> Path:
    path.write_text(
        (
            "[experiment]\n"
            'rollout = "calculator"\n'
            'entrypoint = "main.py"\n'
            'model_path = "Qwen/Qwen3.6-35B-A3B"\n'
            'dataset = "multiply"\n\n'
            "[training]\n"
            "lr = 1e-6\n"
            "total_epochs = 2\n"
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    monkeypatch.setattr(train_module, "console", console)
    monkeypatch.setattr(shared_submit_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def _git_context() -> Any:
        workspace_directory = _find_temp_workspace_directory(Path.cwd()) or Path(
            "/repo"
        )
        return type(
            "GitContext",
            (),
            {
                "workspace_directory": workspace_directory.resolve(),
                "git_identity": GIT_IDENTITY,
                "repo_url": REPO_URL,
                "credentials": FAKE_CREDENTIALS,
            },
        )()

    monkeypatch.setattr(
        shared_submit_module,
        "require_git_workspace_directory_context",
        _git_context,
    )


@pytest.fixture(autouse=True)
def _mock_workspace_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.summarize_local_git_state",
        lambda workspace_directory: type(
            "State",
            (),
            {
                "head_sha": "localhead",
                "is_dirty": False,
                "has_upstream": True,
                "ahead": 0,
                "branch": "main",
            },
        )(),
    )
    disable_commit_preflight(monkeypatch)


def _fake_client_class(submitted: list[bool]) -> type:
    class FakeClient:
        def submit_training_run(self, **kwargs: Any) -> SubmitRunResult:
            submitted.append(True)
            return SubmitRunResult(
                id="train-1",
                name="train-run",
                status="pending",
                created_at="2026-05-27T00:00:00Z",
                platform_url="https://platform.osmosis.ai/training/train-1",
            )

    return FakeClient


def test_train_submit_skips_preflight_when_dependency_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    console_capture: StringIO,
) -> None:
    workspace_directory = _make_workspace_directory(
        tmp_path / "project", main_py="import a_package_that_is_not_installed_xyz\n"
    )
    config_path = _write_train_config(
        workspace_directory / "configs" / "training" / "default.toml"
    )
    monkeypatch.chdir(workspace_directory)
    submitted: list[bool] = []
    fake = _fake_client_class(submitted)
    monkeypatch.setattr(api_client_module, "OsmosisClient", fake)
    monkeypatch.setattr(shared_submit_module, "OsmosisClient", fake)

    result = train_module.submit(config_path=config_path, yes=True)

    assert isinstance(result, OperationResult)
    assert submitted == [True]


def test_train_submit_preflight_still_fails_on_syntax_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    console_capture: StringIO,
) -> None:
    workspace_directory = _make_workspace_directory(
        tmp_path / "project", main_py="def (oops\n"
    )
    config_path = _write_train_config(
        workspace_directory / "configs" / "training" / "default.toml"
    )
    monkeypatch.chdir(workspace_directory)
    submitted: list[bool] = []
    fake = _fake_client_class(submitted)
    monkeypatch.setattr(api_client_module, "OsmosisClient", fake)
    monkeypatch.setattr(shared_submit_module, "OsmosisClient", fake)

    with pytest.raises(CLIError):
        train_module.submit(config_path=config_path, yes=True)
    assert submitted == []

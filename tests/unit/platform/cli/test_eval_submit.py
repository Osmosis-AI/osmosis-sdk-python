from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.eval as eval_module
import osmosis_ai.platform.cli.shared_submit as shared_submit_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.output import OperationResult
from osmosis_ai.platform.api.models import SubmitRunResult

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
FAKE_CREDENTIALS = object()


def _find_temp_workspace_directory(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / ".osmosis").is_dir()
            and (candidate / "configs" / "eval").is_dir()
            and (candidate / "rollouts").is_dir()
        ):
            return candidate
    return None


def _make_workspace_directory(root: Path, *, rollout: str = "calculator") -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis",
        f"rollouts/{rollout}",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / rollout / "main.py").write_text(
        """
from osmosis_ai.rollout import AgentWorkflow, Grader


class TestWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class TestGrader(Grader):
    async def grade(self, ctx):
        return 1.0
""".strip(),
        encoding="utf-8",
    )
    return root


def _write_eval_config(path: Path, *, commit_sha: str | None = None) -> Path:
    commit_line = f'commit_sha = "{commit_sha}"\n' if commit_sha is not None else ""
    path.write_text(
        (
            'secrets = ["OPENAI_API_KEY"]\n\n'
            "[experiment]\n"
            'rollout = "calculator"\n'
            'entrypoint = "main.py"\n'
            'model_path = "openai/gpt-5-mini"\n'
            'dataset = "multiply"\n'
            f"{commit_line}\n"
            "[evaluation]\n"
            "limit = 200\n"
            "n = 2\n"
            "batch_size = 1\n"
            "pass_threshold = 1.0\n"
            "agent_workflow_timeout_s = 450\n"
            "grader_timeout_s = 150\n\n"
            "[env]\n"
            'LOG_LEVEL = "INFO"\n'
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    monkeypatch.setattr(eval_module, "console", console)
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


def test_eval_submit_passes_new_schema_to_evaluation_run_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    console_capture: StringIO,
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path / "project")
    config_path = _write_eval_config(
        workspace_directory / "configs" / "eval" / "default.toml",
        commit_sha="deadbeef",
    )
    monkeypatch.chdir(workspace_directory)
    captured_kwargs: dict[str, Any] = {}

    class FakeClient:
        def submit_evaluation_run(self, **kwargs: Any) -> SubmitRunResult:
            captured_kwargs.update(kwargs)
            return SubmitRunResult(
                id="eval-1",
                name="eval-run",
                status="pending",
                created_at="2026-05-27T00:00:00Z",
                platform_url="https://platform.osmosis.ai/evals/eval-1",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

    result = eval_module.submit(config_path=config_path, yes=True)

    assert captured_kwargs["git_identity"] == GIT_IDENTITY
    assert captured_kwargs["credentials"] is FAKE_CREDENTIALS
    assert captured_kwargs["env_config"] == {"LOG_LEVEL": "INFO"}
    assert captured_kwargs["secrets"] == ["OPENAI_API_KEY"]
    assert captured_kwargs["experiment_config"] == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "model_path": "openai/gpt-5-mini",
        "dataset": "multiply",
        "commit_sha": "deadbeef",
    }
    assert "llm_config" not in captured_kwargs
    assert captured_kwargs["evaluation_config"] == {
        "limit": 200,
        "n": 2,
        "batch_size": 1,
        "pass_threshold": 1.0,
        "agent_workflow_timeout_s": 450.0,
        "grader_timeout_s": 150.0,
    }
    assert "openai/gpt-5-mini" in console_capture.getvalue()
    assert isinstance(result, OperationResult)
    assert result.resource is not None
    assert result.resource["config"] == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "model": "openai/gpt-5-mini",
        "dataset": "multiply",
        "commit_sha": "deadbeef",
    }


def test_eval_submit_does_not_pin_local_head_without_config_commit_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path / "project")
    config_path = _write_eval_config(
        workspace_directory / "configs" / "eval" / "default.toml"
    )
    monkeypatch.chdir(workspace_directory)
    captured_kwargs: dict[str, Any] = {}

    class FakeClient:
        def submit_evaluation_run(self, **kwargs: Any) -> SubmitRunResult:
            captured_kwargs.update(kwargs)
            return SubmitRunResult(
                id="eval-1",
                name="eval-run",
                status="pending",
                created_at="2026-05-27T00:00:00Z",
                platform_url=None,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

    eval_module.submit(config_path=config_path, yes=True)

    assert "commit_sha" not in captured_kwargs["experiment_config"]
    assert captured_kwargs["experiment_config"] == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "model_path": "openai/gpt-5-mini",
        "dataset": "multiply",
    }

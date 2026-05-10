"""Unit tests for `osmosis rollout validate`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.main import main
from osmosis_ai.eval.common.cli import _resolve_rollout_entrypoint
from osmosis_ai.platform.api.models import PaginatedRollouts, RolloutInfo

WORKSPACE_ID = "ws-test"
WORKSPACE_NAME = "team-test"


def _make_project(root: Path, *, with_grader: bool = True) -> Path:
    for rel_path in (
        ".osmosis/research",
        "rollouts/demo",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)

    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n",
        encoding="utf-8",
    )
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# test\n",
        encoding="utf-8",
    )

    if with_grader:
        entrypoint = """
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.grader import Grader


class DemoWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class DemoGrader(Grader):
    async def grade(self, ctx):
        return 1.0
""".strip()
    else:
        entrypoint = """
from osmosis_ai.rollout.agent_workflow import AgentWorkflow


class DemoWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None
""".strip()

    (root / "rollouts" / "demo" / "main.py").write_text(entrypoint, encoding="utf-8")
    return root


def test_rollout_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "rollout", "list"])

    assert rc == 1
    assert "Not in an Osmosis project" in capsys.readouterr().err


def test_rollout_list_passes_workspace_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    workspace = SimpleNamespace(
        project_root=tmp_path,
        workspace_id=WORKSPACE_ID,
        workspace_name=WORKSPACE_NAME,
        credentials=object(),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_workspace_context",
        lambda: workspace,
    )

    class FakeClient:
        def list_rollouts(self, limit=30, offset=0, *, workspace_id, credentials=None):
            assert workspace_id == WORKSPACE_ID
            assert credentials is workspace.credentials
            return PaginatedRollouts(
                rollouts=[
                    RolloutInfo(
                        id="rollout_1",
                        name="demo",
                        is_active=True,
                        repo_full_name="osmosis/demo",
                        last_synced_commit_sha="abc123",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    rc = main(["--json", "rollout", "list"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["items"][0]["name"] == "demo"
    assert payload["workspace"] == {"id": WORKSPACE_ID, "name": WORKSPACE_NAME}
    assert payload["project_root"] == str(tmp_path)


def test_validate_help_shows_config_positional(capsys) -> None:
    rc = main(["rollout", "validate", "--help"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "CONFIG_PATH" in out
    assert "Path to training or eval TOML config file." in out
    assert "--host" not in out
    assert "--validate-only" not in out


def test_validate_training_config_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)
    config_path = project_root / "configs" / "training" / "demo.toml"
    config_path.write_text(
        """
[experiment]
rollout = "demo"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "demo-dataset"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
        encoding="utf-8",
    )

    rc = main(["rollout", "validate", str(config_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Validation passed." in captured.out
    assert "training" in captured.out


def test_validate_eval_config_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)
    config_path = project_root / "configs" / "eval" / "demo.toml"
    config_path.write_text(
        """
[eval]
rollout = "demo"
entrypoint = "main.py"
dataset = "data/demo.jsonl"

[llm]
model = "openai/gpt-5-mini"
""".strip(),
        encoding="utf-8",
    )

    rc = main(["rollout", "validate", str(config_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Validation passed." in captured.out
    assert "eval" in captured.out


def test_validate_rejects_noncanonical_config_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)
    config_path = project_root / "demo.toml"
    config_path.write_text(
        """
[experiment]
rollout = "demo"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "demo-dataset"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
        encoding="utf-8",
    )

    rc = main(["rollout", "validate", str(config_path)])
    captured = capsys.readouterr()

    assert rc != 0
    assert "configs/training" in captured.err
    assert "configs/eval" in captured.err


def test_validate_fails_without_grader(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project_root = _make_project(tmp_path, with_grader=False)
    monkeypatch.chdir(project_root)
    config_path = project_root / "configs" / "training" / "demo.toml"
    config_path.write_text(
        """
[experiment]
rollout = "demo"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "demo-dataset"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
        encoding="utf-8",
    )

    rc = main(["rollout", "validate", str(config_path)])
    captured = capsys.readouterr()

    assert rc != 0
    assert "Grader" in captured.err


def test_resolve_rollout_entrypoint_rejects_parent_rollout(tmp_path: Path) -> None:
    project_root = _make_project(tmp_path / "project")
    outside = project_root / "outside"
    outside.mkdir()
    (outside / "main.py").write_text("# outside\n", encoding="utf-8")

    with pytest.raises(CLIError, match=r"Rollout.*rollouts"):
        _resolve_rollout_entrypoint(
            "../outside",
            "main.py",
            project_root=project_root,
        )


def test_resolve_rollout_entrypoint_rejects_absolute_rollout(tmp_path: Path) -> None:
    project_root = _make_project(tmp_path / "project")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.py").write_text("# outside\n", encoding="utf-8")

    with pytest.raises(CLIError, match=r"Rollout.*rollouts"):
        _resolve_rollout_entrypoint(
            str(outside),
            "main.py",
            project_root=project_root,
        )

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.eval.config import EvalConfig
from osmosis_ai.eval.evaluation.orchestrator import OrchestratorResult


class _FakePlan:
    already_completed = False
    cache_path = Path("/tmp/eval-cache.json")
    samples_path = None
    completed_runs = {(0, 0, None), (1, 0, None)}
    total_expected = 2
    dataset_fingerprint_warning = None
    cache_data = {
        "runs": [
            {"row_index": 0, "run_index": 0, "success": True, "reward": 0.9},
            {
                "row_index": 1,
                "run_index": 0,
                "success": False,
                "error": "boom",
            },
        ],
        "summary": {
            "total_runs": 2,
            "passed": 1,
            "failed": 1,
            "total_tokens": 24,
            "total_duration_ms": 20.0,
            "reward_stats": {"mean": 0.9},
        },
    }

    @property
    def has_pending_work(self) -> bool:
        return False

    def release(self) -> None:
        return None


class _FakeOrchestrator:
    def __init__(self, **kwargs: Any) -> None:
        self.on_progress = kwargs["on_progress"]
        self.cache_config = kwargs["cache_config"]

    def plan(self) -> _FakePlan:
        return _FakePlan()

    async def run_prepared(self, plan: _FakePlan) -> OrchestratorResult:
        raise AssertionError("cached eval result should not execute pending work")


def _make_workspace_directory(root: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "rollouts/demo_rollout",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test\n", encoding="utf-8"
    )
    (root / "rollouts" / "demo_rollout" / "pyproject.toml").write_text(
        "[project]\nname='demo-rollout'\n",
        encoding="utf-8",
    )
    (root / "rollouts" / "demo_rollout" / "workflow.py").write_text(
        "# demo workflow\n",
        encoding="utf-8",
    )
    return root


def test_eval_run_json_returns_final_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config_path = project / "configs" / "eval" / "eval.toml"
    dataset_path = project / "data" / "data.jsonl"
    config_path.write_text("[eval]\n", encoding="utf-8")
    dataset_path.write_text('{"input": "x"}\n', encoding="utf-8")

    config = EvalConfig(
        eval_dataset="data/data.jsonl",
        eval_rollout="demo_rollout",
        eval_entrypoint="workflow.py",
        llm_model="openai/gpt-5.4",
        output_quiet=False,
    )

    monkeypatch.setattr("osmosis_ai.eval.config.load_eval_config", lambda path: config)
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_dataset_rows",
        lambda **kwargs: ([{"input": "x"}, {"input": "y"}], None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.cache.compute_dataset_fingerprint",
        lambda path: "dataset-fingerprint",
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.cache.compute_rollout_filesystem_fingerprint",
        lambda rollout_dir, *, entrypoint: "rollout-fingerprint",
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _FakeOrchestrator,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("eval run must not require credentials"),
    )
    monkeypatch.chdir(project)

    exit_code = cli.main(["--json", "eval", "run", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["data"]["total_runs"] == 2
    assert payload["data"]["completed_runs"] == 2
    assert payload["data"]["failed_runs"] == 1
    assert payload["data"]["cache_path"] == "/tmp/eval-cache.json"
    assert payload["data"]["output_path"] is None
    assert payload["data"]["partial_failures"] == [
        {
            "row_index": 1,
            "run_index": 0,
            "model_tag": None,
            "error": "boom",
        }
    ]


def test_eval_run_json_resolves_workspace_directory_before_eval_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.cli.EvalCommand.run",
        lambda self, **kwargs: pytest.fail(
            "eval run should resolve local workspace directory context before EvalCommand.run"
        ),
    )

    exit_code = cli.main(["--json", "eval", "run", "configs/eval/eval.toml"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Osmosis workspace directory" in captured.err

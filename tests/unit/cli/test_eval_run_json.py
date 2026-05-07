from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.eval.config import EvalConfig
from osmosis_ai.eval.evaluation.orchestrator import OrchestratorResult


class _FakeProxy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = False

    async def preflight_check(self) -> None:
        return None

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


class _FakeOrchestrator:
    def __init__(self, **kwargs: Any) -> None:
        self.on_progress = kwargs["on_progress"]
        self.cache_config = kwargs["cache_config"]

    async def run(self) -> OrchestratorResult:
        self.on_progress(
            1,
            2,
            {
                "success": True,
                "duration_ms": 10.0,
                "tokens": 12,
                "reward": 0.9,
            },
        )
        cache_path = Path("/tmp/eval-cache.json")
        return OrchestratorResult(
            status="completed",
            cache_path=cache_path,
            samples_path=None,
            summary={
                "total_runs": 2,
                "passed": 1,
                "failed": 1,
                "total_tokens": 24,
                "total_duration_ms": 20.0,
                "reward_stats": {"mean": 0.9},
            },
            total_completed=2,
            total_expected=2,
            cache_data={
                "runs": [
                    {"row_index": 0, "run_index": 0, "success": True, "reward": 0.9},
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": False,
                        "error": "boom",
                    },
                ]
            },
        )


def _make_project(root: Path) -> Path:
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
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test\n", encoding="utf-8"
    )
    return root


def test_eval_run_json_returns_final_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    project = _make_project(tmp_path / "project")
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

    fake_workflow = type("FakeWorkflow", (), {})
    fake_grader = type("FakeGrader", (), {})

    monkeypatch.setattr("osmosis_ai.eval.config.load_eval_config", lambda path: config)
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_dataset_rows",
        lambda **kwargs: ([{"input": "x"}, {"input": "y"}], None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **kwargs: (fake_workflow, None, "fake_entrypoint", None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli._resolve_grader",
        lambda *args, **kwargs: (fake_grader, None),
    )
    monkeypatch.setattr("osmosis_ai.eval.llm_proxy.LiteLLMProxy", _FakeProxy)
    monkeypatch.setattr(
        "osmosis_ai.rollout.backend.local.backend.LocalBackend",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "osmosis_ai.rollout.driver.InProcessDriver",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.cache.compute_module_fingerprint",
        lambda module: "module-fingerprint",
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _FakeOrchestrator,
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

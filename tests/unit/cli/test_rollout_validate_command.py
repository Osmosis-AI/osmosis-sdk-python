"""Unit tests for `osmosis rollout validate`."""

from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli.main import main


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


def test_validate_help_shows_config_positional(capsys) -> None:
    rc = main(["rollout", "validate", "--help"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "CONFIG_PATH" in out
    assert "Path to training or eval TOML config file." in out
    assert "--host" not in out
    assert "--validate-only" not in out


def test_validate_training_config_success(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)
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


def test_validate_eval_config_success(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)
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


def test_validate_rejects_noncanonical_config_path(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)
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


def test_validate_fails_without_grader(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path, with_grader=False)
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

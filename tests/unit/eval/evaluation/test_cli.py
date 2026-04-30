from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

from osmosis_ai.eval.evaluation.cli import EvalCommand


class _StopEval(Exception):
    pass


class _FakeProxy:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def preflight_check(self) -> None:
        return None

    async def start(self) -> None:
        return None


def _make_config(**overrides):
    values = {
        "eval_dataset": "dataset.jsonl",
        "eval_rollout": "demo_rollout",
        "eval_entrypoint": "workflow.py",
        "eval_limit": None,
        "eval_offset": 0,
        "eval_fresh": False,
        "eval_retry_failed": False,
        "llm_model": "openai/gpt-5-mini",
        "llm_base_url": None,
        "llm_api_key_env": None,
        "grader_module": None,
        "grader_config": None,
        "runs_batch_size": 1,
        "output_log_samples": False,
        "output_path": None,
        "output_quiet": True,
        "output_debug": False,
        "baseline_model": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_project_dotenv_sets_missing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    EvalCommand._load_project_dotenv(tmp_path)

    assert os.environ["OPENAI_API_KEY"] == "from-dotenv"


def test_load_project_dotenv_does_not_override_shell_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    EvalCommand._load_project_dotenv(tmp_path)

    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def _run_command(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace) -> None:
    fake_workflow = type("FakeWorkflow", (), {})

    monkeypatch.setattr(
        "osmosis_ai.eval.config.load_eval_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_dataset_rows",
        lambda **kwargs: ([{"id": 1}], None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **kwargs: (fake_workflow, None, "fake_entrypoint", None),
    )
    monkeypatch.setattr(EvalCommand, "_resolve_api_key", lambda self, cfg: None)
    monkeypatch.setattr("osmosis_ai.eval.llm_proxy.LiteLLMProxy", _FakeProxy)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.resolve_project_root",
        lambda path: path.parent,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.validate_project_contract",
        lambda project_root: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.ensure_project_config_path",
        lambda *args, **kwargs: None,
    )

    def _fake_local_backend(*, workflow, workflow_config, grader, grader_config):
        assert workflow is fake_workflow
        assert grader is config._expected_grader
        assert grader_config is config._expected_grader_config
        raise _StopEval

    monkeypatch.setattr(
        "osmosis_ai.rollout.backend.local.backend.LocalBackend",
        _fake_local_backend,
    )

    with pytest.raises(_StopEval):
        EvalCommand().run(
            config_path="eval.toml",
            fresh=False,
            retry_failed=False,
            limit=None,
            offset=None,
            quiet=True,
            debug=False,
            output_path=None,
            log_samples=False,
            batch_size_override=None,
        )


def test_run_uses_explicit_grader_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from osmosis_ai.rollout.context import GraderContext
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

    explicit_module = types.ModuleType("explicit_grader_mod")

    class ExplicitGrader(Grader):
        async def grade(self, ctx: GraderContext):
            return None

    explicit_grader = ExplicitGrader
    explicit_config = GraderConfig(name="explicit-grader")
    explicit_module.ExplicitGrader = explicit_grader
    explicit_module.grader_config = explicit_config
    monkeypatch.setitem(sys.modules, "explicit_grader_mod", explicit_module)

    config = _make_config(
        grader_module="explicit_grader_mod:ExplicitGrader",
        grader_config="explicit_grader_mod:grader_config",
    )
    config._expected_grader = explicit_grader
    config._expected_grader_config = explicit_config

    _run_command(monkeypatch, config)


def test_run_auto_discovers_grader_from_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osmosis_ai.rollout.context import GraderContext
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

    entrypoint_module = types.ModuleType("fake_entrypoint")

    class DiscoveredGrader(Grader):
        async def grade(self, ctx: GraderContext):
            return None

    discovered_grader = DiscoveredGrader
    discovered_config = GraderConfig(name="fake-grader")
    entrypoint_module.DiscoveredGrader = discovered_grader
    entrypoint_module.grader_config = discovered_config
    monkeypatch.setitem(sys.modules, "fake_entrypoint", entrypoint_module)

    config = _make_config(eval_entrypoint="fake_entrypoint.py")
    config._expected_grader = discovered_grader
    config._expected_grader_config = discovered_config

    _run_command(monkeypatch, config)

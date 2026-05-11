from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat, override_output_context
from osmosis_ai.eval.evaluation.cli import EvalCommand


def _make_project(root: Path) -> Path:
    for rel_path in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
        "rollouts/demo",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test\n", encoding="utf-8"
    )
    (root / "rollouts" / "demo" / "pyproject.toml").write_text(
        "[project]\nname='demo'\n",
        encoding="utf-8",
    )
    (root / "rollouts" / "demo" / "workflow.py").write_text(
        "# demo workflow\n",
        encoding="utf-8",
    )
    (root / "data" / "dataset.jsonl").write_text(
        '{"input": "x"}\n',
        encoding="utf-8",
    )
    return root


class _FakeEvalConfig(SimpleNamespace):
    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def _make_config(**overrides):
    values = {
        "eval_dataset": "dataset.jsonl",
        "eval_rollout": "demo",
        "eval_entrypoint": "workflow.py",
        "eval_limit": None,
        "eval_offset": 0,
        "eval_fresh": False,
        "eval_retry_failed": False,
        "llm_model": "openai/gpt-5-mini",
        "llm_base_url": None,
        "llm_api_key_env": None,
        "runs_n": 1,
        "runs_batch_size": 1,
        "runs_pass_threshold": 1.0,
        "output_log_samples": False,
        "output_path": None,
        "output_quiet": True,
        "output_debug": False,
        "timeout_agent_sec": 450.0,
        "timeout_grader_sec": 150.0,
    }
    values.update(overrides)
    return _FakeEvalConfig(**values)


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


def _patch_eval_run_common(
    monkeypatch: pytest.MonkeyPatch,
    project: Path,
    config: SimpleNamespace,
) -> None:
    def _fail_if_old_eval_flow_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("old in-process eval flow should not be called")

    monkeypatch.setattr(
        "osmosis_ai.eval.config.load_eval_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.config.resolve_eval_context_paths",
        lambda cfg, project_root: cfg,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_dataset_rows",
        lambda **kwargs: ([{"id": 1}], None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        _fail_if_old_eval_flow_called,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli._resolve_grader",
        _fail_if_old_eval_flow_called,
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
        "osmosis_ai.rollout.backend.local.backend.LocalBackend",
        _fail_if_old_eval_flow_called,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.resolve_project_root_from_cwd",
        lambda: project,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.validate_project_contract",
        lambda project_root: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_contract.ensure_project_config_path",
        lambda *args, **kwargs: None,
    )


class _FakePlan:
    def __init__(self, *, has_pending_work: bool, already_completed: bool) -> None:
        self._has_pending_work = has_pending_work
        self.already_completed = already_completed
        self.cache_path = Path("/tmp/eval-cache.json")
        self.samples_path = None
        self.completed_runs = {(0, 0, None)} if not has_pending_work else set()
        self.total_expected = 1
        self.cache_data = {
            "runs": [
                {
                    "row_index": 0,
                    "run_index": 0,
                    "model_tag": None,
                    "success": True,
                    "reward": 1.0,
                }
            ],
            "summary": {"total_runs": 1, "passed": 1, "failed": 0},
        }
        self.dataset_fingerprint_warning = None
        self.released = False

    @property
    def has_pending_work(self) -> bool:
        return self._has_pending_work

    def release(self) -> None:
        self.released = True


def _eval_run_kwargs() -> dict[str, Any]:
    return {
        "config_path": "configs/eval/eval.toml",
        "fresh": False,
        "retry_failed": False,
        "limit": None,
        "offset": None,
        "quiet": True,
        "debug": False,
        "output_path": None,
        "log_samples": False,
        "batch_size_override": None,
    }


def test_cached_only_eval_skips_api_key_port_and_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    config = _make_config()
    events: list[str] = []

    _patch_eval_run_common(monkeypatch, project, config)

    def _fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("startup dependency should not be touched")

    class _CachedOnlyOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def plan(self) -> _FakePlan:
            events.append("plan")
            return _FakePlan(has_pending_work=False, already_completed=True)

        async def run_prepared(self, plan: _FakePlan) -> Any:
            raise AssertionError("cached-only eval should not execute")

    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _CachedOnlyOrchestrator,
    )
    monkeypatch.setattr(EvalCommand, "_resolve_api_key", _fail_if_called)
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.FixedPortLock",
        lambda *args, **kwargs: _fail_if_called(),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.assert_user_server_port_free",
        _fail_if_called,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.controller.EvalController.start",
        _fail_if_called,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.start_user_server_process",
        _fail_if_called,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.wait_for_user_server_health",
        _fail_if_called,
    )

    result = EvalCommand().run(**_eval_run_kwargs())

    if isinstance(result, int):
        assert result == 0
    else:
        assert result.data["status"] == "already_completed"
    assert events == ["plan"]


def test_eval_run_hash_payload_preserves_legacy_none_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    config = _make_config()
    captured_config: dict[str, Any] = {}

    _patch_eval_run_common(monkeypatch, project, config)

    def _compute_task_id(
        *,
        config: dict[str, Any],
        rollout_fingerprint: str,
        dataset_fingerprint: str,
        entrypoint: str,
    ) -> tuple[str, str]:
        captured_config.update(config)
        assert rollout_fingerprint == "rollout-fingerprint"
        assert dataset_fingerprint == "dataset-fingerprint"
        assert entrypoint == "workflow.py"
        return "task-id", "config-hash"

    class _CachedOnlyOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def plan(self) -> _FakePlan:
            return _FakePlan(has_pending_work=False, already_completed=True)

        async def run_prepared(self, plan: _FakePlan) -> Any:
            raise AssertionError("cached-only eval should not execute")

    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.cache.compute_task_id",
        _compute_task_id,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _CachedOnlyOrchestrator,
    )

    result = EvalCommand().run(**_eval_run_kwargs())

    if isinstance(result, int):
        assert result == 0
    else:
        assert result.data["status"] == "already_completed"

    for legacy_key in (
        "grader_module",
        "grader_config",
        "baseline_model",
        "baseline_base_url",
        "baseline_api_key_env",
    ):
        assert legacy_key in captured_config
        assert captured_config[legacy_key] is None

    assert captured_config.keys().isdisjoint(
        {
            "eval_fresh",
            "eval_retry_failed",
            "llm_api_key_env",
            "runs_batch_size",
            "output_log_samples",
            "output_path",
            "output_quiet",
            "output_debug",
        }
    )


def test_pending_eval_allows_missing_api_key_for_no_auth_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    config = _make_config()
    events: list[str] = []

    _patch_eval_run_common(monkeypatch, project, config)

    class _PendingOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def plan(self) -> _FakePlan:
            events.append("plan")
            return _FakePlan(has_pending_work=True, already_completed=False)

        async def run_prepared(self, plan: _FakePlan) -> Any:
            events.append("run_prepared")
            return SimpleNamespace(
                status="completed",
                cache_path=plan.cache_path,
                samples_path=plan.samples_path,
                summary=plan.cache_data["summary"],
                total_completed=1,
                total_expected=1,
                cache_data=plan.cache_data,
                dataset_fingerprint_warning=None,
            )

    class _FakeFixedPortLock:
        def acquire(self) -> None:
            events.append("fixed_lock")

        def release(self) -> None:
            events.append("release_fixed_lock")

    def _assert_port_free() -> None:
        events.append("port")

    def _resolve_api_key(self: EvalCommand, cfg: Any) -> None:
        events.append("api_key")
        return None

    async def _preflight(self: Any) -> None:
        events.append(f"preflight:{self.api_key!r}")

    async def _start_controller(self: Any) -> None:
        events.append("controller_start")

    async def _stop_controller(self: Any) -> None:
        events.append("controller_stop")

    class _FakeUserServerProcess:
        async def terminate(self) -> None:
            events.append("terminate_user_server")

    async def _start_user_server_process(*args: Any, **kwargs: Any) -> Any:
        events.append("start_user_server")
        return _FakeUserServerProcess()

    async def _wait_for_user_server_health(
        *, process: Any, timeout_sec: float = 30.0
    ) -> None:
        events.append(f"health:{process.__class__.__name__}")

    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _PendingOrchestrator,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.FixedPortLock",
        lambda *args, **kwargs: _FakeFixedPortLock(),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.assert_user_server_port_free",
        _assert_port_free,
    )
    monkeypatch.setattr(EvalCommand, "_resolve_api_key", _resolve_api_key)
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.litellm_bridge.LiteLLMBridge.preflight_check",
        _preflight,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.controller.EvalController.start",
        _start_controller,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.controller.EvalController.stop",
        _stop_controller,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.start_user_server_process",
        _start_user_server_process,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.wait_for_user_server_health",
        _wait_for_user_server_health,
    )

    result = EvalCommand().run(**_eval_run_kwargs())

    assert result == 0
    assert events == [
        "plan",
        "fixed_lock",
        "port",
        "api_key",
        "preflight:None",
        "controller_start",
        "start_user_server",
        "health:_FakeUserServerProcess",
        "run_prepared",
        "terminate_user_server",
        "controller_stop",
        "release_fixed_lock",
    ]


def test_pending_eval_configured_empty_api_key_env_json_is_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    config = _make_config(llm_api_key_env="MISSING_KEY")
    events: list[str] = []

    _patch_eval_run_common(monkeypatch, project, config)

    class _PendingOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def plan(self) -> _FakePlan:
            events.append("plan")
            return _FakePlan(has_pending_work=True, already_completed=False)

        async def run_prepared(self, plan: _FakePlan) -> Any:
            raise AssertionError(
                "empty configured API key should stop before execution"
            )

    class _FakeFixedPortLock:
        def acquire(self) -> None:
            events.append("fixed_lock")

        def release(self) -> None:
            events.append("release_fixed_lock")

    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _PendingOrchestrator,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.FixedPortLock",
        lambda *args, **kwargs: _FakeFixedPortLock(),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.assert_user_server_port_free",
        lambda: events.append("port"),
    )
    with override_output_context(format=OutputFormat.json):
        with pytest.raises(CLIError) as exc_info:
            EvalCommand().run(**_eval_run_kwargs())

    assert exc_info.value.code == "VALIDATION"
    assert "Environment variable 'MISSING_KEY'" in exc_info.value.message
    assert events == [
        "plan",
        "fixed_lock",
        "port",
        "release_fixed_lock",
    ]


def test_pending_eval_progress_prints_reward_in_rich_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    config = _make_config(output_quiet=False)

    _patch_eval_run_common(monkeypatch, project, config)

    class _ProgressOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            self.on_progress = kwargs["on_progress"]

        def plan(self) -> _FakePlan:
            return _FakePlan(has_pending_work=True, already_completed=False)

        async def run_prepared(self, plan: _FakePlan) -> Any:
            self.on_progress(
                1,
                1,
                {
                    "success": True,
                    "duration_ms": 3100,
                    "tokens": 610,
                    "reward": 1.0,
                    "model_tag": None,
                    "error": None,
                },
            )
            return SimpleNamespace(
                status="completed",
                cache_path=plan.cache_path,
                samples_path=plan.samples_path,
                summary=plan.cache_data["summary"],
                total_completed=1,
                total_expected=1,
                cache_data=plan.cache_data,
                dataset_fingerprint_warning=None,
            )

    class _FakeFixedPortLock:
        def acquire(self) -> None:
            return None

        def release(self) -> None:
            return None

    class _FakeUserServerProcess:
        async def terminate(self) -> None:
            return None

    async def _noop_async(*args: Any, **kwargs: Any) -> None:
        return None

    async def _start_user_server_process(*args: Any, **kwargs: Any) -> Any:
        return _FakeUserServerProcess()

    monkeypatch.setattr(
        "osmosis_ai.eval.evaluation.orchestrator.EvalOrchestrator",
        _ProgressOrchestrator,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.FixedPortLock",
        lambda *args, **kwargs: _FakeFixedPortLock(),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.locks.assert_user_server_port_free",
        lambda: None,
    )
    monkeypatch.setattr(EvalCommand, "_resolve_api_key", lambda self, cfg: None)
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.litellm_bridge.LiteLLMBridge.preflight_check",
        _noop_async,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.controller.EvalController.start",
        _noop_async,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.controller.EvalController.stop",
        _noop_async,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.start_user_server_process",
        _start_user_server_process,
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.process.wait_for_user_server_health",
        _noop_async,
    )

    with override_output_context(format=OutputFormat.rich):
        result = EvalCommand().run(**{**_eval_run_kwargs(), "quiet": False})

    assert result == 0
    assert "[reward=1.000]" in capsys.readouterr().out


def test_eval_run_config_path_must_be_under_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from osmosis_ai.cli.main import main

    project = _make_project(tmp_path / "project")
    outside = tmp_path / "other.toml"
    outside.write_text(
        "[eval]\n"
        "rollout='r'\n"
        "entrypoint='main.py'\n"
        "dataset='data/x.jsonl'\n"
        "[llm]\n"
        "model='openai/test'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    rc = main(["--json", "eval", "run", str(outside)])

    assert rc == 1
    assert "configs/eval" in capsys.readouterr().err

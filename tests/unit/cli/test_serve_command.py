"""Unit tests for `osmosis rollout serve` (v2 config-based CLI)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from osmosis_ai.cli.commands.rollout import (
    _format_http_origin,
    _trace_log_basename,
    _tracing_backend_proxy_cls,
)
from osmosis_ai.cli.main import main
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import (
    ExecutionRequest,
    ExecutionResult,
    GraderConfig,
    RolloutStatus,
)
from osmosis_ai.rollout_v2.validator import ValidationResult


class DummyGrader(Grader):
    async def grade(self, ctx):
        pass


DUMMY_GRADER_CONFIG = GraderConfig(name="dummy_grader")


def _minimal_serve_toml(tmp_path: Path) -> Path:
    p = tmp_path / "serve.toml"
    p.write_text(
        '[serve]\nrollout = "dummy"\nentrypoint = "dummy_entry.py"\n',
        encoding="utf-8",
    )
    return p


def test_validate_only_and_no_validate_mutually_exclusive(tmp_path, capsys):
    """--validate-only and --no-validate cannot be used together."""
    cfg = _minimal_serve_toml(tmp_path)
    rc = main(
        [
            "rollout",
            "serve",
            str(cfg),
            "--validate-only",
            "--no-validate",
        ]
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc != 0
    assert "mutually exclusive" in combined.lower()


def test_missing_config_file_exits_nonzero(tmp_path, capsys):
    missing = tmp_path / "nope.toml"
    rc = main(["rollout", "serve", str(missing)])
    assert rc != 0


def test_validate_subcommand_removed(capsys):
    """Standalone `rollout validate` must not exist."""
    rc = main(["rollout", "validate", "-m", "foo:bar"])
    assert rc != 0


def test_serve_help_shows_config_positional_and_no_module_flag(capsys):
    """Serve help documents v2 interface (config path, no -m)."""
    rc = main(["rollout", "serve", "--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CONFIG_PATH" in out
    assert "Path to serve TOML config file." in out
    assert "--module" not in out
    assert "--local" not in out
    assert re.search(r"(?<!\w)-m(?!\w)", out) is None


@pytest.fixture
def patch_serve_pipeline(monkeypatch, tmp_path):
    """Avoid binding a real server; exercise wiring after config load."""
    cfg = _minimal_serve_toml(tmp_path)

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    class _WF(AgentWorkflow):
        async def run(self, ctx):
            pass

    wf_cfg = AgentWorkflowConfig(name="wf_test")

    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **_: (_WF, wf_cfg, None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.auto_discover_grader",
        lambda _ep: (DummyGrader, DUMMY_GRADER_CONFIG),
    )
    monkeypatch.setattr(
        "osmosis_ai.rollout_v2.validator.validate_backend",
        lambda *_a, **_kw: ValidationResult(valid=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr("uvicorn.run", MagicMock())
    return cfg


def test_serve_fails_without_grader_even_with_no_validate(
    monkeypatch, tmp_path, capsys
):
    """Missing grader cannot be bypassed with --no-validate."""
    cfg = _minimal_serve_toml(tmp_path)

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    class _WF(AgentWorkflow):
        async def run(self, ctx):
            pass

    wf_cfg = AgentWorkflowConfig(name="wf_test")

    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **_: (_WF, wf_cfg, None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.auto_discover_grader",
        lambda _ep: (None, None),
    )
    monkeypatch.setattr("uvicorn.run", MagicMock())

    rc = main(
        ["rollout", "serve", str(cfg), "--no-validate"],
    )
    assert rc != 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "grader" in combined.lower()


def test_serve_fails_without_grader_with_debug_no_validate_in_config(
    monkeypatch, tmp_path, capsys
):
    """[debug].no_validate must not skip the mandatory grader check."""
    cfg = tmp_path / "serve.toml"
    cfg.write_text(
        '[serve]\nrollout = "dummy"\nentrypoint = "dummy_entry.py"\n\n'
        "[debug]\n"
        "no_validate = true\n",
        encoding="utf-8",
    )

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    class _WF(AgentWorkflow):
        async def run(self, ctx):
            pass

    wf_cfg = AgentWorkflowConfig(name="wf_test")

    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **_: (_WF, wf_cfg, None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.auto_discover_grader",
        lambda _ep: (None, None),
    )
    monkeypatch.setattr("uvicorn.run", MagicMock())

    rc = main(["rollout", "serve", str(cfg)])
    assert rc != 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "grader" in combined.lower()


def test_validate_only_success_prints_and_exits(patch_serve_pipeline, capsys):
    cfg_path = patch_serve_pipeline
    rc = main(["rollout", "serve", str(cfg_path), "--validate-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Validation passed" in out


@pytest.mark.parametrize(
    "rollout_id",
    ["../escape", "foo/bar", "..\\win", "a/../b", "nul:foo"],
)
def test_trace_log_basename_has_no_path_components(rollout_id: str):
    name = _trace_log_basename(rollout_id)
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name.endswith(".jsonl")
    assert Path(name).name == name


def test_trace_log_basename_distinct_for_similar_paths():
    """Different rollout IDs must not collide (stem can be similar; hash differs)."""
    a = _trace_log_basename("foo/bar")
    b = _trace_log_basename("foo_bar")
    assert a != b


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("127.0.0.1", 9000, "http://127.0.0.1:9000"),
        ("example.com", 8080, "http://example.com:8080"),
        ("::1", 9000, "http://[::1]:9000"),
        ("2001:db8::1", 443, "http://[2001:db8::1]:443"),
    ],
)
def test_format_http_origin_brackets_ipv6(host: str, port: int, expected: str):
    assert _format_http_origin(host, port) == expected


@pytest.fixture
def serve_app_capture(monkeypatch, tmp_path):
    """Build full serve path; capture FastAPI app passed to uvicorn.run."""
    cfg = tmp_path / "serve.toml"
    cfg.write_text(
        '[serve]\nrollout = "dummy"\nentrypoint = "dummy_entry.py"\n',
        encoding="utf-8",
    )

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    class _WF(AgentWorkflow):
        async def run(self, ctx):
            pass

    wf_cfg = AgentWorkflowConfig(name="wf_test")

    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **_: (_WF, wf_cfg, None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.auto_discover_grader",
        lambda _ep: (DummyGrader, DUMMY_GRADER_CONFIG),
    )
    monkeypatch.setattr(
        "osmosis_ai.rollout_v2.validator.validate_backend",
        lambda *_a, **_kw: ValidationResult(valid=True, errors=[], warnings=[]),
    )
    captured: dict = {}

    def _capture_uvicorn(app, **_kwargs):
        captured["app"] = app

    monkeypatch.setattr("uvicorn.run", _capture_uvicorn)
    return cfg, captured


def test_non_local_health_is_public(serve_app_capture):
    from fastapi.testclient import TestClient

    cfg, captured = serve_app_capture
    rc = main(["rollout", "serve", str(cfg)])
    assert rc == 0
    app = captured["app"]
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_platform_health_not_exposed(serve_app_capture):
    from fastapi.testclient import TestClient

    cfg, captured = serve_app_capture
    rc = main(["rollout", "serve", str(cfg)])
    assert rc == 0
    client = TestClient(captured["app"])
    r = client.get("/platform/health")
    assert r.status_code == 404


def test_post_rollout_missing_label_returns_400_json(serve_app_capture):
    from fastapi.testclient import TestClient

    cfg, captured = serve_app_capture
    rc = main(["rollout", "serve", str(cfg)])
    assert rc == 0
    client = TestClient(captured["app"])
    payload = {
        "rollout_id": "rid-1",
        "initial_messages": [],
        "chat_completions_url": "http://localhost:9/v1/chat/completions",
        "completion_callback_url": "http://localhost:9/cb",
        "label": None,
    }
    r = client.post(
        "/rollout",
        json=payload,
    )
    assert r.status_code == 400
    body = r.json()
    assert "detail" in body


async def test_tracing_backend_proxy_writes_jsonl_under_flat_name(tmp_path):
    inner = MagicMock()
    inner.max_concurrency = 2
    inner.health = MagicMock(return_value={"status": "ok"})

    async def _run_execute(request, on_wf, on_grader=None):
        await on_wf(ExecutionResult(status=RolloutStatus.SUCCESS))

    inner.execute = AsyncMock(side_effect=_run_execute)

    session = tmp_path / "sess1"
    proxy = _tracing_backend_proxy_cls()(inner, session)
    assert session.is_dir()

    req = ExecutionRequest(
        id="r1",
        prompt=[{"role": "user", "content": "hi"}],
        label="L",
    )

    async def on_wf(res: ExecutionResult):
        pass

    await proxy.execute(req, on_wf, None)
    inner.execute.assert_awaited_once()
    trace_files = list(session.glob("*.jsonl"))
    assert len(trace_files) == 1
    assert trace_files[0].name == _trace_log_basename("r1")
    lines = trace_files[0].read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(e.get("event") == "rollout_start" for e in events)
    assert any(e.get("event") == "workflow_complete" for e in events)
    assert any(e.get("rollout_id") == "r1" for e in events)


async def test_tracing_backend_proxy_malicious_rollout_ids_stay_in_session_dir(
    tmp_path,
):
    """Dangerous rollout_id strings must not create nested paths under the session."""
    inner = MagicMock()
    inner.max_concurrency = 2
    inner.health = MagicMock(return_value={"status": "ok"})

    async def _run_execute(request, on_wf, on_grader=None):
        await on_wf(ExecutionResult(status=RolloutStatus.SUCCESS))

    inner.execute = AsyncMock(side_effect=_run_execute)

    session = tmp_path / "sess1"
    proxy = _tracing_backend_proxy_cls()(inner, session)

    dangerous = ["../escape", "foo/bar", "x/../../../etc/passwd"]
    for rid in dangerous:
        req = ExecutionRequest(
            id=rid,
            prompt=[{"role": "user", "content": "hi"}],
            label="L",
        )

        async def on_wf(_res: ExecutionResult):
            pass

        await proxy.execute(req, on_wf, None)

    for p in session.rglob("*.jsonl"):
        assert p.parent.resolve() == session.resolve()
        assert ".." not in p.name
        assert "/" not in p.name
    assert len(list(session.glob("*.jsonl"))) == len(dangerous)


def test_serve_with_trace_dir_creates_session_subdir(monkeypatch, tmp_path, capsys):
    """debug.trace_dir in TOML creates a session directory under the trace root."""
    trace_root = tmp_path / "traces"
    cfg = tmp_path / "serve.toml"
    cfg.write_text(
        '[serve]\nrollout = "dummy"\nentrypoint = "dummy_entry.py"\n\n'
        "[debug]\n"
        f'trace_dir = "{trace_root.as_posix()}"\n',
        encoding="utf-8",
    )

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

    class _WF(AgentWorkflow):
        async def run(self, ctx):
            pass

    wf_cfg = AgentWorkflowConfig(name="wf_test")

    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.load_workflow",
        lambda **_: (_WF, wf_cfg, None),
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.common.cli.auto_discover_grader",
        lambda _ep: (DummyGrader, DUMMY_GRADER_CONFIG),
    )
    monkeypatch.setattr(
        "osmosis_ai.rollout_v2.validator.validate_backend",
        lambda *_a, **_kw: ValidationResult(valid=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr("uvicorn.run", MagicMock())

    rc = main(["rollout", "serve", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Trace" in out
    subdirs = [p for p in trace_root.iterdir() if p.is_dir()]
    assert len(subdirs) == 1

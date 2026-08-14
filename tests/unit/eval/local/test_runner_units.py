"""Supervisor unit tests that need no subprocess: fingerprint, env, orphans."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from osmosis_ai.eval.local.dataset import (
    ResolvedDataset,
    resolve_explicit_dataset_file,
    select_rows,
)
from osmosis_ai.eval.local.runner import (
    RESERVED_ENV_NAMES,
    EvalRunSpec,
    LocalEvalError,
    build_run_inputs,
    build_subprocess_env,
    compute_source_digest,
    format_input_diff,
    generated_run_name,
    reap_verified_orphan,
    reserve_free_port,
)
from osmosis_ai.eval.local.state import (
    ChildProcessRecord,
    OrphanChildError,
    diff_inputs,
    digest_of,
)


def _spec(**overrides: Any) -> EvalRunSpec:
    payload: dict[str, Any] = {
        "rollout_name": "echo-rollout",
        "entrypoint": "main.py",
        "model_path": "openai/gpt-5-mini",
        "dataset_name": "echo",
        "n": 1,
        "pass_threshold": 1.0,
    }
    payload.update(overrides)
    return EvalRunSpec(**payload)


def _dataset(sha: str = "a" * 64) -> ResolvedDataset:
    return ResolvedDataset(
        path=Path("/tmp/echo.jsonl"),
        sha256=sha,
        extension="jsonl",
        source="explicit",
        dataset_name="echo.jsonl",
    )


# --------------------------------------------------------------------------- #
# Source digest
# --------------------------------------------------------------------------- #


def test_source_digest_changes_with_code(tmp_path: Path) -> None:
    project = tmp_path / "rollout"
    project.mkdir()
    (project / "main.py").write_text("print(1)")
    before = compute_source_digest(project)
    (project / "main.py").write_text("print(2)")
    assert compute_source_digest(project) != before


def test_source_digest_is_a_full_sha256(tmp_path: Path) -> None:
    project = tmp_path / "rollout"
    project.mkdir()
    (project / "main.py").write_text("print(1)")
    digest = compute_source_digest(project)
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_source_digest_excludes_a_nested_output_dir(tmp_path: Path) -> None:
    # Otherwise the run would change its own digest just by writing results.
    project = tmp_path / "rollout"
    output = project / ".osmosis" / "evals"
    output.mkdir(parents=True)
    (project / "main.py").write_text("print(1)")
    before = compute_source_digest(project, exclude=project / ".osmosis")
    (output / "index.jsonl").write_text('{"row_index": 0}\n')
    assert compute_source_digest(project, exclude=project / ".osmosis") == before


def test_source_digest_ignores_caches(tmp_path: Path) -> None:
    project = tmp_path / "rollout"
    (project / "__pycache__").mkdir(parents=True)
    (project / "main.py").write_text("print(1)")
    before = compute_source_digest(project)
    (project / "__pycache__" / "main.pyc").write_bytes(b"\x00")
    assert compute_source_digest(project) == before


def test_source_digest_refuses_a_directory_symlink(tmp_path: Path) -> None:
    project = tmp_path / "rollout"
    project.mkdir()
    (project / "main.py").write_text("print(1)")
    (tmp_path / "elsewhere").mkdir()
    (project / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="rollout source contains a directory"):
        compute_source_digest(project)


# --------------------------------------------------------------------------- #
# Fingerprint (§9.5)
# --------------------------------------------------------------------------- #


def _inputs(
    spec: EvalRunSpec, dataset: ResolvedDataset, tmp_path: Path, **kw: Any
) -> dict[str, Any]:
    path = tmp_path / "d.jsonl"
    if not path.exists():
        path.write_text(
            '{"user_prompt": "a", "ground_truth": "1"}\n'
            '{"user_prompt": "b", "ground_truth": "2"}\n'
        )
    selection = select_rows(path, **kw)
    return build_run_inputs(
        spec, dataset=dataset, selection=selection, rollout_source_digest="b" * 64
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_path", "openai/gpt-4o"),
        ("n", 3),
        ("entrypoint", "server.py"),
        ("pass_threshold", 0.5),
        ("agent_timeout_sec", 999.0),
    ],
)
def test_semantic_changes_change_the_fingerprint(
    tmp_path: Path, field_name: str, value: Any
) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    changed = _inputs(_spec(**{field_name: value}), _dataset(), tmp_path)
    assert digest_of(changed) != digest_of(baseline)
    assert diff_inputs(baseline, changed) != []


def test_dataset_bytes_change_the_fingerprint(tmp_path: Path) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    changed = _inputs(_spec(), _dataset(sha="c" * 64), tmp_path)
    assert digest_of(changed) != digest_of(baseline)


def test_source_digest_changes_the_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"user_prompt": "a", "ground_truth": "1"}\n')
    selection = select_rows(path)
    first = build_run_inputs(
        _spec(), dataset=_dataset(), selection=selection, rollout_source_digest="1" * 64
    )
    second = build_run_inputs(
        _spec(), dataset=_dataset(), selection=selection, rollout_source_digest="2" * 64
    )
    assert digest_of(first) != digest_of(second)


def test_row_selection_changes_the_fingerprint(tmp_path: Path) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    narrowed = _inputs(_spec(), _dataset(), tmp_path, row_selector=(0,))
    assert digest_of(narrowed) != digest_of(baseline)
    assert baseline["dataset"]["selected_source_rows"] == "0-1"
    assert narrowed["dataset"]["selected_source_rows"] == "0"


@pytest.mark.parametrize("field_name", ["batch_size", "branch", "commit_sha"])
def test_throughput_and_display_fields_are_excluded(
    tmp_path: Path, field_name: str
) -> None:
    values: dict[str, Any] = {
        "batch_size": 32,
        "branch": "other",
        "commit_sha": "abcdef1",
    }
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    changed = _inputs(_spec(**{field_name: values[field_name]}), _dataset(), tmp_path)
    assert digest_of(changed) == digest_of(baseline)


def test_secret_names_are_included_and_values_are_never_present(tmp_path: Path) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    with_secret = _inputs(_spec(secret_names=("OPENAI_API_KEY",)), _dataset(), tmp_path)
    assert digest_of(with_secret) != digest_of(baseline)
    assert with_secret["secret_names"] == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" not in str(with_secret).replace("'OPENAI_API_KEY'", "")


def test_env_ordering_does_not_change_the_fingerprint(tmp_path: Path) -> None:
    first = _inputs(_spec(env={"A": "1", "B": "2"}), _dataset(), tmp_path)
    second = _inputs(_spec(env={"B": "2", "A": "1"}), _dataset(), tmp_path)
    assert digest_of(first) == digest_of(second)


def test_input_diff_renders_field_level_lines() -> None:
    diffs = diff_inputs({"n": 1, "model_path": "a"}, {"n": 5, "model_path": "a"})
    rendered = format_input_diff(diffs)
    assert rendered == "  - n: 1 -> 5"


# --------------------------------------------------------------------------- #
# Generated run names (§4.4)
# --------------------------------------------------------------------------- #


def test_generated_name_carries_stem_stamp_and_fingerprint() -> None:
    name = generated_run_name("my-eval", "abc123def456", now="20260814T010203Z")
    assert name == "my-eval-20260814T010203Z-abc123de"


def test_generated_name_sanitizes_an_unsafe_stem() -> None:
    name = generated_run_name("my eval/v2", "abc123def456", now="20260814T010203Z")
    assert "/" not in name
    assert name.startswith("my-eval-v2-")


def test_generated_name_survives_an_empty_stem() -> None:
    assert generated_run_name("", "abc123def456", now="20260814T010203Z").startswith(
        "eval-"
    )


# --------------------------------------------------------------------------- #
# Subprocess environment (§8)
# --------------------------------------------------------------------------- #


def _env(**kw: Any) -> dict[str, str]:
    payload: dict[str, Any] = {
        "base": {"PATH": "/usr/bin"},
        "config_env": {},
        "secrets": {},
        "port": 1234,
        "artifact_root": Path("/runs/rollout_trials"),
        "instance_id": "iid",
    }
    payload.update(kw)
    return build_subprocess_env(**payload)


def test_internal_variables_are_applied_last() -> None:
    env = _env(config_env={"LOG_LEVEL": "DEBUG"}, secrets={"OPENAI_API_KEY": "sk-x"})
    assert env["_OSMOSIS_ROLLOUT_PORT"] == "1234"
    assert env["_OSMOSIS_ROLLOUT_ARTIFACT_ROOT"] == "/runs/rollout_trials"
    assert env["_OSMOSIS_ROLLOUT_INSTANCE_ID"] == "iid"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["LOG_LEVEL"] == "DEBUG"
    assert env["PATH"] == "/usr/bin"


@pytest.mark.parametrize("reserved", sorted(RESERVED_ENV_NAMES))
def test_config_cannot_hijack_supervisor_variables(reserved: str) -> None:
    with pytest.raises(LocalEvalError, match="supervisor-owned"):
        _env(config_env={reserved: "hijacked"})


@pytest.mark.parametrize("reserved", sorted(RESERVED_ENV_NAMES))
def test_secrets_cannot_hijack_supervisor_variables(reserved: str) -> None:
    with pytest.raises(LocalEvalError, match="supervisor-owned"):
        _env(secrets={reserved: "hijacked"})


def test_secrets_override_config_env_for_the_same_name() -> None:
    env = _env(config_env={"TOKEN": "from-config"}, secrets={"TOKEN": "from-secret"})
    assert env["TOKEN"] == "from-secret"


# --------------------------------------------------------------------------- #
# Port reservation
# --------------------------------------------------------------------------- #


def test_reserved_port_is_usable() -> None:
    import socket

    port = reserve_free_port()
    assert 1024 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


# --------------------------------------------------------------------------- #
# Verified orphan cleanup (§8)
# --------------------------------------------------------------------------- #


def _child(pid: int, *, port: int = 9, instance_id: str = "iid") -> ChildProcessRecord:
    return ChildProcessRecord(
        supervisor_pid=os.getpid(),
        child_pid=pid,
        child_pgid=pid,
        port=port,
        instance_id=instance_id,
    )


def _logs() -> tuple[list[tuple[str, str, str]], Any]:
    entries: list[tuple[str, str, str]] = []

    def log(level: str, step: str, message: str, **_details: Any) -> None:
        entries.append((level, step, message))

    return entries, log


async def test_a_dead_recorded_child_is_just_cleared() -> None:
    entries, log = _logs()
    async with httpx.AsyncClient() as client:
        # PID 2**31-1 is above every real pid on the platforms we support.
        await reap_verified_orphan(_child(2**31 - 1), client=client, log=log)
    assert entries == [("info", "orphan", "clearing a stale rollout-server record")]


async def test_a_live_but_unverifiable_child_refuses_with_instructions() -> None:
    _, log = _logs()
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OrphanChildError, match="could not be verified"):
            await reap_verified_orphan(_child(os.getpid()), client=client, log=log)


async def test_a_child_whose_instance_id_mismatches_is_never_killed() -> None:
    # A reused pid must not be killed just because something answers /health.
    _, log = _logs()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"instance_id": "someone-else"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OrphanChildError):
            await reap_verified_orphan(_child(os.getpid()), client=client, log=log)


async def test_a_verified_orphan_is_terminated(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    def fake_terminate(pgid: int, *, grace_sec: float, poll_sec: float = 0.05) -> None:
        killed.append(pgid)

    monkeypatch.setattr(
        "osmosis_ai.eval.local.runner.terminate_process_group", fake_terminate
    )
    entries, log = _logs()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"instance_id": "iid"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await reap_verified_orphan(_child(os.getpid()), client=client, log=log)
    assert killed == [os.getpid()]
    assert entries[0][0] == "warning"


# --------------------------------------------------------------------------- #
# Dataset plumbing sanity
# --------------------------------------------------------------------------- #


def test_explicit_dataset_flows_into_the_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"user_prompt": "a", "ground_truth": "1"}\n')
    resolved = resolve_explicit_dataset_file(path)
    inputs = build_run_inputs(
        _spec(),
        dataset=resolved,
        selection=select_rows(path),
        rollout_source_digest="b" * 64,
    )
    assert inputs["dataset"]["sha256"] == resolved.sha256

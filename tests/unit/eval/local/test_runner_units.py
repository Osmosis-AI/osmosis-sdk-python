"""Supervisor unit tests that need no rollout server: fingerprint, env, startup."""

from __future__ import annotations

import asyncio
import contextlib
import json
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
    LocalEvalOptions,
    _classify_terminal,
    build_run_inputs,
    build_subprocess_env,
    changed_input_keys,
    compute_source_digest,
    generated_run_name,
    reserve_free_port,
)
from osmosis_ai.eval.local.state import digest_of
from osmosis_ai.rollout.controller import TerminalCallbackResult
from osmosis_ai.rollout.types import GraderCompleteRequest, GraderStatus, RolloutSample


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
        ("agent_timeout_sec", 999.0),
    ],
)
def test_semantic_changes_change_the_fingerprint(
    tmp_path: Path, field_name: str, value: Any
) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    changed = _inputs(_spec(**{field_name: value}), _dataset(), tmp_path)
    assert digest_of(changed) != digest_of(baseline)
    # The refusal message has to be able to name what moved.
    assert changed_input_keys(baseline, changed) != []


def test_pass_threshold_does_not_change_the_fingerprint(tmp_path: Path) -> None:
    baseline = _inputs(_spec(), _dataset(), tmp_path)
    changed = _inputs(_spec(pass_threshold=0.5), _dataset(), tmp_path)
    assert changed == baseline


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


def test_changed_input_keys_names_the_changed_top_level_keys() -> None:
    previous = {"n": 1, "model_path": "a", "rollout": {"source_digest": "b" * 64}}
    current = {"n": 5, "model_path": "a", "rollout": {"source_digest": "e" * 64}}
    # Naming the top-level key is what makes the refusal actionable; a nested
    # change still surfaces as its owning key.
    assert changed_input_keys(previous, current) == ["n", "rollout"]


def test_changed_input_keys_reports_added_and_removed_keys() -> None:
    assert changed_input_keys({"a": 1}, {"b": 2}) == ["a", "b"]


def test_identical_inputs_have_no_changed_keys(tmp_path: Path) -> None:
    # A plain resume of an unchanged run must never be refused.
    assert (
        changed_input_keys(
            _inputs(_spec(), _dataset(), tmp_path),
            _inputs(_spec(), _dataset(), tmp_path),
        )
        == []
    )


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
# Rollout-server startup on a pinned port (§8)
# --------------------------------------------------------------------------- #


async def _serve_health(payload: dict[str, Any]) -> asyncio.Server:
    """Answer one ``GET /health`` per connection with *payload*, on a free port."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(Exception):
            await reader.readuntil(b"\r\n\r\n")
            body = json.dumps(payload).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            await writer.drain()
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


@pytest.mark.parametrize(
    "health",
    [
        {"status": "ok", "instance_id": "someone-elses-instance"},
        # A server too old to report an instance id is just as foreign.
        {"status": "ok"},
    ],
    ids=["other-instance-id", "no-instance-id"],
)
async def test_a_foreign_rollout_server_on_a_pinned_port_is_refused(
    tmp_path: Path, health: dict[str, Any]
) -> None:
    # Something unrelated already owns the pinned --rollout-port. Driving it
    # would dispatch this run's work to a process the supervisor cannot manage,
    # so startup must refuse instead of silently succeeding.
    server = await _serve_health(health)
    port = int(server.sockets[0].getsockname()[1])
    # The entrypoint never binds anything: the port is already taken, and the
    # point of the test is what the supervisor does with the answer it gets.
    (tmp_path / "main.py").write_text("import time\n\ntime.sleep(60)\n")
    runner = _runner(
        _spec(),
        tmp_path,
        options=LocalEvalOptions(name="run-1", rollout_port=port),
    )
    runner._run_dir = tmp_path / "evals" / "run-1"
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(LocalEvalError, match="already listening"):
                await runner._start_rollout_server(secrets={}, client=client)
    finally:
        server.close()
        await server.wait_closed()
    # The refusal is not allowed to leak the child it spawned.
    assert runner._child is None


# --------------------------------------------------------------------------- #
# Concurrency resolution (§10)
# --------------------------------------------------------------------------- #


async def _resolved_concurrency(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    health: dict[str, Any] | None,
) -> int:
    from osmosis_ai.eval.local import runner as runner_module

    async def fake_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return health

    monkeypatch.setattr(runner_module, "probe_health", fake_probe)
    async with httpx.AsyncClient() as client:
        return await runner._resolve_concurrency(client, "http://127.0.0.1:1")


@pytest.mark.parametrize(
    ("health", "expected"),
    [
        # Harbor advertises its queue depth at top level...
        ({"max_queue_depth": 3}, 3),
        # ...LocalBackend reports the limiter snapshot it actually enforces.
        ({"concurrency": {"max_concurrent": 4}}, 4),
    ],
    ids=["harbor-queue-depth", "local-backend-limiter"],
)
async def test_health_capacity_caps_an_unset_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    health: dict[str, Any],
    expected: int,
) -> None:
    runner = _runner(_spec(), tmp_path)
    assert await _resolved_concurrency(runner, monkeypatch, health) == expected


@pytest.mark.parametrize(
    "health",
    [
        None,
        {},
        {"max_queue_depth": 0},
        {"max_queue_depth": "many"},
        {"max_queue_depth": True},
        {"concurrency": {"max_concurrent": None}},
        {"concurrency": {"max_concurrent": 0}},
        {"concurrency": "unbounded"},
    ],
    ids=[
        "unreachable",
        "empty",
        "zero-depth",
        "garbage-depth",
        "bool-depth",
        "unbounded-limiter",
        "zero-limiter",
        "garbage-limiter",
    ],
)
async def test_absent_or_garbage_capacity_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, health: dict[str, Any] | None
) -> None:
    # No advertised cap means one worker, not an unbounded fan-out.
    runner = _runner(_spec(), tmp_path)
    assert await _resolved_concurrency(runner, monkeypatch, health) == 1


async def test_health_capacity_bounds_a_configured_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Over-queueing a backend that reported its own limit creates work it cannot
    # service, so the advertised capacity is a hard cap on the request.
    runner = _runner(
        _spec(batch_size=16),
        tmp_path,
        options=LocalEvalOptions(name="run-1", max_in_flight=8),
    )
    health = {"concurrency": {"max_concurrent": 2}}
    assert await _resolved_concurrency(runner, monkeypatch, health) == 2


async def test_max_in_flight_takes_precedence_over_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(
        _spec(batch_size=1),
        tmp_path,
        options=LocalEvalOptions(name="run-1", max_in_flight=3),
    )
    assert await _resolved_concurrency(runner, monkeypatch, {}) == 3


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


# --------------------------------------------------------------------------- #
# Supervisor deadline and secret redaction
# --------------------------------------------------------------------------- #


def _runner(
    spec: EvalRunSpec,
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    options: LocalEvalOptions | None = None,
) -> Any:
    from osmosis_ai.eval.local.dataset import select_rows
    from osmosis_ai.eval.local.runner import LocalEvalRunner

    data = tmp_path / "d.jsonl"
    data.write_text('{"user_prompt": "a", "ground_truth": "1"}\n')

    class _Hooks:
        def note(self, message: str) -> None: ...
        def confirm_dispatch(self, *, pending: int, model_path: str) -> None: ...
        def resolve_secrets(self, names: Any) -> dict[str, str]:
            return {}

        def progress(self, snapshot: Any) -> None: ...

    return LocalEvalRunner(
        spec=spec,
        options=options or LocalEvalOptions(name="run-1"),
        dataset=_dataset(),
        selection=select_rows(data),
        rollout_dir=tmp_path,
        output_root=output_root or tmp_path / "evals",
        hooks=_Hooks(),
    )


def test_configured_timeouts_set_the_supervisor_deadline(tmp_path: Path) -> None:
    runner = _runner(_spec(agent_timeout_sec=450.0, grader_timeout_sec=150.0), tmp_path)
    # Server budget plus the callback/network grace (§10).
    assert runner._callback_deadline() == 450.0 + 150.0 + 60.0


def test_an_unconfigured_timeout_still_yields_a_finite_deadline(tmp_path: Path) -> None:
    # An unbounded wait would hang every worker forever on a lost callback.
    runner = _runner(_spec(), tmp_path)
    deadline = runner._callback_deadline()
    assert deadline is not None
    assert deadline > 0


@pytest.mark.parametrize(
    ("agent", "grader"),
    [(None, 30.0), (30.0, None)],
)
def test_an_unbounded_phase_keeps_the_default_deadline(
    tmp_path: Path, agent: float | None, grader: float | None
) -> None:
    # ``None`` means "run unbounded", so bounding the item to the phase that is
    # configured would stamp still-running work `callback_timeout` -- a durable
    # failed record that a plain resume then skips.
    from osmosis_ai.eval.local.runner import _DEFAULT_ITEM_DEADLINE_SEC

    runner = _runner(
        _spec(agent_timeout_sec=agent, grader_timeout_sec=grader), tmp_path
    )
    assert runner._callback_deadline() == _DEFAULT_ITEM_DEADLINE_SEC


def test_an_output_root_equal_to_the_rollout_dir_is_refused(tmp_path: Path) -> None:
    # Excluding the output tree would exclude the whole project, freezing the
    # digest that resume compares against.
    runner = _runner(_spec(), tmp_path, output_root=tmp_path)
    with pytest.raises(LocalEvalError, match="rollout source directory"):
        runner._source_digest()


def test_the_redactor_replaces_secrets_and_keeps_context() -> None:
    from osmosis_ai.eval.local.runner import SecretRedactor

    redactor = SecretRedactor(["sk-super-secret-value"])
    scrubbed = redactor.scrub("Traceback: auth failed with sk-super-secret-value here")
    assert "sk-super-secret-value" not in scrubbed
    assert "Traceback: auth failed with" in scrubbed
    assert "[REDACTED]" in scrubbed


def test_the_redactor_ignores_short_placeholder_values() -> None:
    from osmosis_ai.eval.local.runner import SecretRedactor

    # "dummy" and friends are placeholders; redacting them would blank
    # unrelated text.
    redactor = SecretRedactor(["dummy", ""])
    assert redactor.scrub("using dummy credentials") == "using dummy credentials"


def test_the_redactor_prefers_the_longest_overlapping_value() -> None:
    from osmosis_ai.eval.local.runner import SecretRedactor

    redactor = SecretRedactor(["token-abcdef", "token-abcdef-longer"])
    scrubbed = redactor.scrub("value=token-abcdef-longer")
    assert scrubbed == "value=[REDACTED]"


# --------------------------------------------------------------------------- #
# Terminal classification and per-item journalling
# --------------------------------------------------------------------------- #


def _terminal(
    *,
    status: GraderStatus = GraderStatus.SUCCESS,
    reward: float | None = 1.0,
    remove_sample: bool = False,
) -> TerminalCallbackResult:
    return TerminalCallbackResult(
        rollout_id="f" * 32,
        source="grader",
        grader=GraderCompleteRequest(
            status=status,
            rollout_id="f" * 32,
            sample=RolloutSample(
                messages=[{"role": "assistant", "content": "ok"}],
                reward=reward,
                remove_sample=remove_sample,
            ),
        ),
    )


def test_a_crashed_grader_that_removed_the_sample_is_a_failure() -> None:
    # "skipped" would drop the row from the scored denominator, so a run whose
    # graders all crashed would report a clean pass rate and exit 0.
    assert _classify_terminal(
        _terminal(status=GraderStatus.FAILURE, reward=None, remove_sample=True)
    ) == ("failed", None, "grader_failed")


def test_remove_sample_from_a_successful_grader_is_still_skipped() -> None:
    assert _classify_terminal(_terminal(remove_sample=True)) == ("skipped", None, None)


async def test_a_retried_failure_writes_its_own_terminal_record(tmp_path: Path) -> None:
    from osmosis_ai.eval.local.runner import WorkItem
    from osmosis_ai.eval.local.state import TerminalJournal, TerminalRecord

    runner = _runner(_spec(), tmp_path)
    journal = TerminalJournal(tmp_path / "events.jsonl")
    journal.open_for_append(journal.replay())
    runner._journal = journal
    item = WorkItem(row=runner._selection.rows[0], run_index=0)
    runner._latest[item.key] = TerminalRecord(
        row_index=item.row.row_index,
        run_index=0,
        rollout_id="a" * 32,
        status="failed",
        error_type="stale",
    )
    runner._dispatch_context["b" * 32] = item
    try:
        await runner._journal_supervisor_failure(item, RuntimeError("boom"))
    finally:
        journal.close()

    # Keeping the earlier attempt's record would leave --retry-failed reporting
    # a stale error with nothing in the journal about the attempt that just ran.
    record = runner._latest[item.key]
    assert record.rollout_id == "b" * 32
    assert (record.status, record.error_type) == ("failed", "supervisor_error")


def test_the_run_log_redacts_and_is_owner_only(tmp_path: Path) -> None:
    from osmosis_ai.eval.local.runner import RunLog, SecretRedactor

    redactor = SecretRedactor(["sk-live-abcdef123456"])
    log = RunLog(tmp_path / "logs.txt", redact=redactor.scrub)
    try:
        log.write("info", "rollout-server", "boom sk-live-abcdef123456")
        log.write("info", "dispatch", "detail", token="sk-live-abcdef123456")
    finally:
        log.close()
    text = (tmp_path / "logs.txt").read_text()
    assert "sk-live-abcdef123456" not in text
    assert text.count("[REDACTED]") == 2
    assert (tmp_path / "logs.txt").stat().st_mode & 0o077 == 0

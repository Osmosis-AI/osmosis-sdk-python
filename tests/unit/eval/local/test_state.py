"""Durable-state tests: journal replay, manifest lock, and the run flock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from osmosis_ai.eval.local.state import (
    LOCAL_STATE_SCHEMA_VERSION,
    JournalCorruptionError,
    LocalEvalStateError,
    RunLock,
    RunLockedError,
    RunManifest,
    ServerProcessState,
    TerminalJournal,
    TerminalRecord,
    archive_run_directory,
    atomic_write_json,
    digest_of,
    process_start_token,
    reap_orphan_server,
    validate_run_name,
)


def _record(
    row: int, run: int, *, status: str = "success", **overrides: object
) -> TerminalRecord:
    payload: dict[str, object] = {
        "row_index": row,
        "run_index": run,
        "rollout_id": f"{row:016x}{run:016x}",
        "status": status,
        "reward": 1.0,
        "tokens": 7,
        "duration_ms": 12.5,
    }
    payload.update(overrides)
    return TerminalRecord(**payload)  # type: ignore[arg-type]


async def _open_journal(path: Path) -> TerminalJournal:
    journal = TerminalJournal(path)
    journal.open_for_append(journal.replay())
    return journal


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #


async def test_append_then_replay_round_trips(tmp_path: Path) -> None:
    journal = await _open_journal(tmp_path / "events.jsonl")
    try:
        await journal.append(_record(0, 0))
        await journal.append(
            _record(1, 0, status="failed", reward=None, error_type="boom")
        )
    finally:
        journal.close()

    replay = TerminalJournal(tmp_path / "events.jsonl").replay()
    assert [r.key for r in replay.records] == [(0, 0), (1, 0)]
    assert replay.truncated_bytes == 0
    failed = replay.latest[(1, 0)]
    assert failed.status == "failed"
    assert failed.reward is None
    assert failed.error_type == "boom"


async def test_none_values_are_omitted_not_null(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    journal = await _open_journal(path)
    try:
        await journal.append(_record(0, 0, reward=None, tokens=None))
    finally:
        journal.close()
    payload = json.loads(path.read_text().splitlines()[0])
    assert "reward" not in payload
    assert "tokens" not in payload


async def test_every_record_is_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    journal = await _open_journal(path)
    try:
        await journal.append(_record(0, 0))
        await journal.append(_record(0, 1))
    finally:
        journal.close()
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2


def test_latest_terminal_record_wins_in_append_order(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    # A retry appends a second record for the same key. Append order decides,
    # and a record carries no timestamp at all, so no clock can outvote it.
    first = _record(0, 0, status="failed", reward=0.0)
    second = _record(0, 0, status="success", reward=1.0)
    path.write_bytes(first.to_journal_line() + second.to_journal_line())

    latest = TerminalJournal(path).replay().latest
    assert latest[(0, 0)].status == "success"


def test_replay_discards_a_partial_trailing_record(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    good = _record(0, 0)
    # A crash mid-append can leave valid JSON with no terminating newline.
    fragment = _record(0, 1).to_journal_line().rstrip(b"\n")
    path.write_bytes(good.to_journal_line() + fragment)

    replay = TerminalJournal(path).replay()
    assert [r.key for r in replay.records] == [(0, 0)]
    assert replay.truncated_bytes == len(fragment)
    assert replay.committed_size == len(good.to_journal_line())


async def test_open_for_append_truncates_the_fragment(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    good = _record(0, 0)
    path.write_bytes(good.to_journal_line() + b'{"row_index": 0, "run_i')

    journal = TerminalJournal(path)
    journal.open_for_append(journal.replay())
    try:
        await journal.append(_record(0, 1))
    finally:
        journal.close()

    replay = TerminalJournal(path).replay()
    assert [r.key for r in replay.records] == [(0, 0), (0, 1)]
    assert replay.truncated_bytes == 0


def test_replay_refuses_a_malformed_committed_record(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        _record(0, 0).to_journal_line()
        + b"not json\n"
        + _record(0, 1).to_journal_line()
    )
    with pytest.raises(JournalCorruptionError, match="invalid JSON"):
        TerminalJournal(path).replay()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"row_index": "0"}, "row_index must be an integer"),
        ({"status": "cancelled"}, "status must be one of"),
        ({"rollout_id": ""}, "rollout_id must be a single path segment"),
        # The id names a directory under rollout_trials/, so a traversal in a
        # replayed record would read and project files outside the run.
        ({"rollout_id": "../../etc"}, "rollout_id must be a single path segment"),
        ({"rollout_id": "nested/id"}, "rollout_id must be a single path segment"),
        ({"rollout_id": 7}, "rollout_id must be a string"),
        ({"tokens": "many"}, "tokens must be an integer"),
        # A required field written as JSON null is as unusable as an absent one.
        ({"run_index": None}, "run_index is missing"),
    ],
)
def test_replay_refuses_records_with_bad_fields(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    payload = _record(0, 0).to_payload()
    payload.update(mutation)
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(JournalCorruptionError, match=match):
        TerminalJournal(path).replay()


def test_replay_of_a_missing_journal_is_empty(tmp_path: Path) -> None:
    replay = TerminalJournal(tmp_path / "absent.jsonl").replay()
    assert replay.records == ()
    assert replay.committed_size == 0
    assert replay.latest == {}


async def test_append_before_open_is_refused(tmp_path: Path) -> None:
    journal = TerminalJournal(tmp_path / "events.jsonl")
    with pytest.raises(LocalEvalStateError, match="not open for appending"):
        await journal.append(_record(0, 0))


# --------------------------------------------------------------------------- #
# Manifest and resolved-input lock
# --------------------------------------------------------------------------- #


def _inputs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_path": "openai/gpt-5-mini",
        "dataset_sha256": "a" * 64,
        "n": 1,
        "rollout": {"entrypoint": "main.py", "source_digest": "b" * 64},
    }
    payload.update(overrides)
    return payload


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = RunManifest.create(
        local_run_id="c" * 32,
        run_name="my-run",
        inputs=_inputs(),
        provenance={"sdk_version": "0.3.0", "git_head": "d" * 40},
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = RunManifest.read(path)
    assert loaded.inputs == manifest.inputs
    assert loaded.provenance["sdk_version"] == "0.3.0"
    assert "inputs_digest" not in json.loads(path.read_text())
    assert path.read_text().endswith("}\n")


def test_manifest_digest_ignores_key_order() -> None:
    reordered = dict(reversed(list(_inputs().items())))
    assert digest_of(reordered) == digest_of(_inputs())


def test_manifest_read_refuses_a_foreign_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(
        path,
        {
            "schema_version": LOCAL_STATE_SCHEMA_VERSION + 1,
            "inputs": {},
        },
    )
    with pytest.raises(LocalEvalStateError, match="state schema version"):
        RunManifest.read(path)


# --------------------------------------------------------------------------- #
# Run naming and archive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["my-run", "run_1", "a", "v0.3.0-smoke"])
def test_valid_run_names(name: str) -> None:
    assert validate_run_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", ".locks", "a/b", "a\\b", "-lead", ".lead", "has space", "a" * 200],
)
def test_invalid_run_names(name: str) -> None:
    with pytest.raises(LocalEvalStateError):
        validate_run_name(name)


def test_archive_run_directory_moves_results_aside(tmp_path: Path) -> None:
    run_dir = tmp_path / "my-run"
    (run_dir / "rollout_trials").mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("{}\n")

    archived = archive_run_directory(run_dir, now="20260814T000000Z")
    assert archived.name == "my-run.archive-20260814T000000Z"
    assert (archived / "events.jsonl").read_text() == "{}\n"
    assert not run_dir.exists()


def test_archive_run_directory_never_overwrites_an_archive(tmp_path: Path) -> None:
    (tmp_path / "my-run.archive-20260814T000000Z").mkdir()
    run_dir = tmp_path / "my-run"
    run_dir.mkdir()
    archived = archive_run_directory(run_dir, now="20260814T000000Z")
    assert archived.name == "my-run.archive-20260814T000000Z-1"


# --------------------------------------------------------------------------- #
# Run lock
# --------------------------------------------------------------------------- #


def test_lock_refuses_a_second_supervisor(tmp_path: Path) -> None:
    path = tmp_path / ".locks" / "my-run.lock"
    with RunLock(path):
        with pytest.raises(RunLockedError, match="already holds"):
            RunLock(path).acquire()


def test_lock_is_reacquirable_after_release(tmp_path: Path) -> None:
    path = tmp_path / ".locks" / "my-run.lock"
    with RunLock(path):
        pass
    with RunLock(path):
        pass


def test_the_lock_still_holds_after_the_run_directory_is_archived(
    tmp_path: Path,
) -> None:
    # The lock lives outside the run dir precisely so --fresh cannot unlock it:
    # an inode inside the archived directory would leave the new path open to a
    # second supervisor.
    evals = tmp_path / "evals"
    run_dir = evals / "my-run"
    run_dir.mkdir(parents=True)
    lock_path = evals / ".locks" / "my-run.lock"
    with RunLock(lock_path):
        inode = lock_path.stat().st_ino
        archive_run_directory(run_dir, now="20260814T000000Z")
        with pytest.raises(RunLockedError):
            RunLock(lock_path).acquire()
        assert lock_path.stat().st_ino == inode


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #


def test_atomic_write_json_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    atomic_write_json(path, {"summary": {"pass_rate": 0.5}})
    atomic_write_json(path, {"summary": {"pass_rate": 1.0}})
    assert json.loads(path.read_text())["summary"]["pass_rate"] == 1.0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json"]


def test_atomic_write_json_refuses_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "metrics.json", {"reward": float("nan")})


# --------------------------------------------------------------------------- #
# Rollout-server ownership record
# --------------------------------------------------------------------------- #


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A live process in its own group, like a spawned rollout server."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _state_for(
    process: subprocess.Popen[bytes], **overrides: object
) -> ServerProcessState:
    token = process_start_token(process.pid)
    assert token is not None
    payload: dict[str, object] = {
        "pid": process.pid,
        "pgid": process.pid,
        "start_token": token,
        "instance_id": "iid",
        "port": 4242,
        "created_at": "2026-08-23T00:00:00Z",
    }
    payload.update(overrides)
    return ServerProcessState(**payload)  # type: ignore[arg-type]


def test_start_token_is_stable_for_a_live_process() -> None:
    token = process_start_token(os.getpid())
    assert token
    assert process_start_token(os.getpid()) == token


def test_start_token_is_none_for_dead_or_invalid_pids() -> None:
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    assert process_start_token(process.pid) is None
    assert process_start_token(0) is None
    assert process_start_token(-1) is None


def test_server_state_round_trips(
    tmp_path: Path, sleeper: subprocess.Popen[bytes]
) -> None:
    path = tmp_path / "server.json"
    state = _state_for(sleeper)
    state.write(path)
    assert ServerProcessState.read(path) == state
    assert state.is_owner_alive()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"pid": "12", "pgid": 12, "start_token": "t"}',
        '{"pid": 12, "pgid": 0, "start_token": "t"}',
        '{"pid": 12, "pgid": 12, "start_token": ""}',
        '{"pid": true, "pgid": 12, "start_token": "t"}',
    ],
)
def test_server_state_read_tolerates_garbage(tmp_path: Path, content: str) -> None:
    # The record only ever enables extra cleanup, so nothing about it may
    # raise into a run.
    path = tmp_path / "server.json"
    path.write_text(content, encoding="utf-8")
    assert ServerProcessState.read(path) is None
    assert ServerProcessState.read(tmp_path / "missing.json") is None


def test_reap_of_a_missing_record_is_a_no_op(tmp_path: Path) -> None:
    assert reap_orphan_server(tmp_path / "server.json", grace_sec=0.5) is None


def test_reap_terminates_a_verified_orphan_group(
    tmp_path: Path, sleeper: subprocess.Popen[bytes]
) -> None:
    path = tmp_path / "server.json"
    state = _state_for(sleeper)
    state.write(path)

    assert reap_orphan_server(path, grace_sec=5.0) == state
    assert not path.exists()
    assert sleeper.wait(timeout=10) != 0


def test_reap_never_kills_a_recycled_pid(
    tmp_path: Path, sleeper: subprocess.Popen[bytes]
) -> None:
    # The sleeper stands in for an unrelated process that happens to hold the
    # recorded pid/pgid today: its start token cannot match the recorded one.
    path = tmp_path / "server.json"
    _state_for(sleeper, start_token="a-start-time-from-another-life").write(path)

    assert reap_orphan_server(path, grace_sec=0.5) is None
    assert sleeper.poll() is None
    # A record that failed verification can never verify again; it is dropped.
    assert not path.exists()


def test_reap_of_a_dead_pid_drops_the_record(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", ""], start_new_session=True)
    token = process_start_token(process.pid)
    process.wait()
    path = tmp_path / "server.json"
    ServerProcessState(
        pid=process.pid,
        pgid=process.pid,
        start_token=token or "gone-before-capture",
        instance_id="iid",
        port=4242,
        created_at="2026-08-23T00:00:00Z",
    ).write(path)

    assert reap_orphan_server(path, grace_sec=0.5) is None
    assert not path.exists()

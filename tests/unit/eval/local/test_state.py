"""Durable-state tests: journal replay, manifest lock, and the run flock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.eval.local.state import (
    LOCAL_STATE_SCHEMA_VERSION,
    JournalCorruptionError,
    LocalEvalStateError,
    RunLock,
    RunLockedError,
    RunManifest,
    TerminalJournal,
    TerminalRecord,
    archive_run_directory,
    atomic_write_json,
    digest_of,
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
    assert loaded.inputs_digest == digest_of(_inputs())
    assert loaded.provenance["sdk_version"] == "0.3.0"
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
            "inputs_digest": "x",
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

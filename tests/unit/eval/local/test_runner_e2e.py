"""LocalBackend end-to-end: a real rollout-server subprocess and a real stub proxy.

These tests spawn the rollout server the way ``osmosis eval run`` does, so they
cover the parts no in-process fake can: artifact-root override, HTTP dispatch,
callback delivery, journal-before-ack ordering, and resume across processes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.dataset import select_rows
from osmosis_ai.eval.local.runner import LocalEvalOptions, ResumeRefusedError
from osmosis_ai.eval.local.state import TerminalJournal, TerminalRecord
from osmosis_ai.rollout.http_driver import (
    HttpRolloutDriver,
    RolloutAdmissionTimeoutError,
    RolloutProtocolError,
)

from .conftest import RecordingHooks, RunnerHarness

pytestmark = pytest.mark.slow


async def test_a_full_run_produces_the_download_layout(harness: RunnerHarness) -> None:
    summary = await harness.runner().run()

    assert summary.total_work_items == 4
    assert summary.dispatched == 4
    assert summary.succeeded == 4

    run_dir = harness.run_dir()
    for name in (
        "manifest.json",
        "events.jsonl",
        "index.jsonl",
        "progress.json",
        "summary.jsonl",
        "metrics.json",
        "logs.txt",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "rollout_trials").is_dir()

    rows = harness.index_rows()
    assert [(row["row_index"], row["run_index"]) for row in rows] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    ]
    for row in rows:
        assert len(row["rollout_id"]) == 32
        assert row["trajectory_filename"] == "trajectory.json"
        assert isinstance(row["duration_ms"], float)
    assert (run_dir / "summary.jsonl").read_text() == (
        run_dir / "index.jsonl"
    ).read_text()


async def test_the_stub_completion_is_graded_and_rewarded(
    harness: RunnerHarness,
) -> None:
    # The contract stub replies "ok" and every row's label is "ok", so a correct
    # end-to-end path yields reward 1.0 -- this is the reward-plumbing assertion.
    summary = await harness.runner().run()
    assert summary.succeeded == 4
    assert summary.metrics["pass_rate"] == 1
    assert summary.metrics["graded"] == 4
    rewards = {row["reward"] for row in harness.index_rows()}
    assert rewards == {1.0}


async def test_trajectories_and_projections_are_written(harness: RunnerHarness) -> None:
    await harness.runner().run()
    run_dir = harness.run_dir()
    for row_index in range(4):
        projection = run_dir / "trajectories" / f"row_{row_index}_run_0.json"
        assert projection.is_file()
        document = json.loads(projection.read_text())
        rollout_id = document["extra"]["osmosis"]["rollout_id"]
        canonical = run_dir / "rollout_trials" / rollout_id / "trajectory.json"
        assert canonical.is_file()
        # Independent copies: editing the projection must not touch the source.
        assert projection.resolve() != canonical.resolve()
        assert document["extra"]["osmosis"]["request_extra_fields"] == {
            "row_index": row_index,
            "run_index": 0,
        }


async def test_artifacts_are_projected_when_the_workflow_writes_them(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_WRITE_ARTIFACT", "1")
    await harness.runner().run()
    projected = sorted((harness.run_dir() / "artifacts").glob("row_*_run_0/note.txt"))
    assert len(projected) == 4
    assert projected[0].read_text() == "ok"


async def test_progress_json_reports_the_dataset_size(harness: RunnerHarness) -> None:
    selection = select_rows(harness.dataset.path, row_selector=(1, 2))
    await harness.runner(selection=selection).run()
    assert json.loads((harness.run_dir() / "progress.json").read_text()) == {
        "total_runs": 2,
        "sampled_rows": 2,
        "total_dataset_rows": 4,
    }


async def test_row_index_is_the_position_in_the_selected_set(
    harness: RunnerHarness,
) -> None:
    selection = select_rows(harness.dataset.path, row_selector=(2, 3))
    await harness.runner(selection=selection).run()
    rows = harness.index_rows()
    assert [row["row_index"] for row in rows] == [0, 1]
    # The dataset offset survives in the journal, for local UX and provenance.
    # The journal is appended in completion order, which concurrent dispatch
    # makes nondeterministic; only the set of offsets is the contract.
    assert sorted(line["source_row_index"] for line in harness.journal_lines()) == [
        2,
        3,
    ]


async def test_multiple_attempts_per_row(harness: RunnerHarness) -> None:
    selection = select_rows(harness.dataset.path, row_selector=(0,))
    summary = await harness.runner(spec=harness.spec(n=3), selection=selection).run()
    assert summary.total_work_items == 3
    rows = harness.index_rows()
    assert [row["run_index"] for row in rows] == [0, 1, 2]
    assert summary.metrics["n_runs"] == 3
    assert len(summary.metrics["pass_at_k"]) >= 2


async def test_logs_txt_uses_the_download_line_format(harness: RunnerHarness) -> None:
    await harness.runner().run()
    lines = (harness.run_dir() / "logs.txt").read_text().splitlines()
    assert lines, "no log lines written"
    for line in lines:
        stamp, level, rest = line.split(" ", 2)
        assert stamp.endswith("Z")
        assert level in ("INFO", "WARNING", "ERROR")
        assert rest.startswith("[")
    assert any("[rollout-server]" in line for line in lines)


# --------------------------------------------------------------------------- #
# Resume, fresh, retry
# --------------------------------------------------------------------------- #


async def test_a_second_run_of_the_same_name_dispatches_nothing(
    harness: RunnerHarness,
) -> None:
    await harness.runner().run()
    hooks = RecordingHooks()
    summary = await harness.runner(hooks=hooks).run()
    assert summary.dispatched == 0
    assert summary.resumed == 4
    assert hooks.confirmations == []
    # No pending work means credentials and the subprocess are never needed.
    assert hooks.secret_requests == []
    assert all(row["resumed"] is True for row in harness.index_rows())
    # The complete run offered a new run; declining kept the no-op.
    assert hooks.new_run_prompts == [("run-1", 4)]


async def test_accepting_the_new_run_prompt_starts_a_generated_name(
    harness: RunnerHarness,
) -> None:
    await harness.runner().run()
    journal_before = (harness.run_dir() / "events.jsonl").read_bytes()

    hooks = RecordingHooks(accept_new_run=True)
    summary = await harness.runner(hooks=hooks).run()

    assert hooks.new_run_prompts == [("run-1", 4)]
    assert summary.run_name != "run-1"
    assert summary.dispatched == 4
    assert summary.resumed == 0
    # The complete run is left in place -- nothing archived, nothing rewritten.
    assert (harness.run_dir() / "events.jsonl").read_bytes() == journal_before
    assert not list(harness.output_root.glob("run-1.archive-*"))
    assert len(harness.index_rows(summary.run_name)) == 4


async def test_a_partial_journal_reruns_only_the_missing_items(
    harness: RunnerHarness,
) -> None:
    # Simulate a crash after two work items were durably journaled.
    await harness.runner(
        selection=select_rows(harness.dataset.path, row_selector=(0, 1))
    ).run()
    first_pass = {(r["row_index"], r["run_index"]) for r in harness.index_rows()}
    assert first_pass == {(0, 0), (1, 0)}

    hooks = RecordingHooks()
    summary = await harness.runner(
        hooks=hooks,
        options=LocalEvalOptions(name="run-1"),
        selection=select_rows(harness.dataset.path, row_selector=(0, 1)),
    ).run()
    assert summary.dispatched == 0
    assert summary.resumed == 2


async def test_a_changed_fingerprint_refuses_and_names_what_changed(
    harness: RunnerHarness,
) -> None:
    await harness.runner().run()
    with pytest.raises(ResumeRefusedError) as excinfo:
        await harness.runner(spec=harness.spec(model_path="openai/gpt-4o")).run()
    message = str(excinfo.value)
    # The message is the whole refusal surface, so it has to name the input that
    # moved and the flag that gets past it.
    assert "Changed: model_path" in message
    assert "--fresh" in message


async def test_changed_rollout_code_refuses_resume(harness: RunnerHarness) -> None:
    await harness.runner().run()
    (harness.rollout_dir / "extra.py").write_text("# a code change\n")
    # Editing the rollout source moves rollout.source_digest, reported by its
    # top-level key -- enough to point the reader at the code they just changed.
    with pytest.raises(ResumeRefusedError, match=r"Changed: rollout\b"):
        await harness.runner().run()


async def test_fresh_archives_the_previous_results_under_the_same_name(
    harness: RunnerHarness,
) -> None:
    await harness.runner().run()
    original_manifest = json.loads((harness.run_dir() / "manifest.json").read_text())

    hooks = RecordingHooks()
    summary = await harness.runner(
        hooks=hooks,
        spec=harness.spec(model_path="openai/gpt-4o"),
        options=LocalEvalOptions(name="run-1", fresh=True),
    ).run()

    assert summary.dispatched == 4
    assert summary.resumed == 0
    archives = sorted(harness.output_root.glob("run-1.archive-*"))
    assert len(archives) == 1
    # Archive, never silent-delete: the old journal is still readable.
    assert (archives[0] / "events.jsonl").is_file()
    assert json.loads((archives[0] / "manifest.json").read_text()) == original_manifest
    assert any("archived previous results" in note for note in hooks.notes)


async def test_retry_failed_reruns_failures_with_a_fresh_rollout_id(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_GRADER_CRASH", "1")
    selection = select_rows(harness.dataset.path, row_selector=(0,))
    first = await harness.runner(selection=selection).run()
    assert first.failed == 1
    failed_rollout_id = harness.index_rows()[0]["rollout_id"]

    monkeypatch.delenv("OSMOSIS_TEST_GRADER_CRASH")
    second = await harness.runner(
        selection=selection,
        options=LocalEvalOptions(name="run-1", retry_failed=True),
    ).run()
    assert second.succeeded == 1
    row = harness.index_rows()[0]
    assert row["status"] == "success"
    assert row["rollout_id"] != failed_rollout_id
    # Both attempts stay in the journal; the later one wins.
    assert len(harness.journal_lines()) == 2
    # The superseded attempt's artifacts remain for diagnosis.
    assert (harness.run_dir() / "rollout_trials" / failed_rollout_id).is_dir()


async def test_failures_are_skipped_by_default_on_resume(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_GRADER_CRASH", "1")
    selection = select_rows(harness.dataset.path, row_selector=(0,))
    await harness.runner(selection=selection).run()

    monkeypatch.delenv("OSMOSIS_TEST_GRADER_CRASH")
    summary = await harness.runner(selection=selection).run()
    assert summary.dispatched == 0
    assert summary.failed == 1


async def test_a_durably_journaled_item_never_reruns(harness: RunnerHarness) -> None:
    """The kill -9 gate, exercised by pre-seeding the journal directly."""
    run_dir = harness.run_dir()
    selection = select_rows(harness.dataset.path, row_selector=(0, 1))
    # Create the run so its manifest matches, then journal one result by hand.
    await harness.runner(selection=selection).run()
    baseline = harness.journal_lines()
    assert len(baseline) == 2

    # A journal that already covers every work item leaves nothing to dispatch,
    # even though no rollout directory for the hand-written id exists.
    journal = TerminalJournal(run_dir / "events.jsonl")
    replay = journal.replay()
    journal.open_for_append(replay)
    try:
        await journal.append(
            TerminalRecord(
                row_index=0,
                run_index=0,
                rollout_id="f" * 32,
                status="failed",
                source_row_index=0,
                duration_ms=1.0,
                error_type="hand_written",
            )
        )
    finally:
        journal.close()

    summary = await harness.runner(selection=selection).run()
    assert summary.dispatched == 0
    row = next(r for r in harness.index_rows() if r["row_index"] == 0)
    assert row["rollout_id"] == "f" * 32
    assert row["error_type"] == "hand_written"
    # No trajectory exists for the hand-written attempt, so the key is omitted.
    assert "trajectory_filename" not in row


# --------------------------------------------------------------------------- #
# Failure and skip handling
# --------------------------------------------------------------------------- #


async def test_a_grader_crash_is_a_terminal_failure(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_GRADER_CRASH", "1")
    summary = await harness.runner().run()
    assert summary.failed == 4
    assert summary.metrics["graded"] == 0
    assert summary.metrics["pass_rate"] == 0
    assert len(summary.failures) == 4
    assert summary.failures[0].rollout_dir.is_dir()


async def test_remove_sample_becomes_a_skipped_row(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_REMOVE_SAMPLE", "1")
    summary = await harness.runner().run()
    assert summary.skipped == 4
    assert summary.metrics["skipped"] == 4
    assert summary.metrics["completed_samples"] == 0
    assert all(row["status"] == "skipped" for row in harness.index_rows())


async def test_a_missing_entrypoint_fails_before_dispatch(
    harness: RunnerHarness,
) -> None:
    from osmosis_ai.eval.local.runner import LocalEvalError

    with pytest.raises(LocalEvalError, match="entrypoint"):
        await harness.runner(spec=harness.spec(entrypoint="absent.py")).run()


# --------------------------------------------------------------------------- #
# Concurrency and confirmation
# --------------------------------------------------------------------------- #


# Smoke coverage only: these prove the knobs are accepted end to end and every
# work item still lands. The bound itself is unit-tested against
# ``_resolve_concurrency`` in test_runner_units.py.


async def test_batch_size_is_accepted_and_the_run_completes(
    harness: RunnerHarness,
) -> None:
    summary = await harness.runner(spec=harness.spec(batch_size=2)).run()
    assert summary.dispatched == 4


async def test_max_in_flight_and_batch_size_together_complete_the_run(
    harness: RunnerHarness,
) -> None:
    summary = await harness.runner(
        spec=harness.spec(batch_size=1),
        options=LocalEvalOptions(name="run-1", max_in_flight=4),
    ).run()
    assert summary.dispatched == 4


async def test_confirmation_receives_the_pending_count(harness: RunnerHarness) -> None:
    hooks = RecordingHooks()
    await harness.runner(hooks=hooks).run()
    assert hooks.confirmations == [(4, "openai/gpt-5-mini")]
    assert hooks.progress_snapshots[-1].completed == 4


async def test_the_run_narrates_its_stages(harness: RunnerHarness) -> None:
    """Without --verbose these lines are the whole run report, so every wait a
    user sits through -- preflight, server startup, scheduling -- names itself."""
    hooks = RecordingHooks()
    await harness.runner(hooks=hooks).run()
    narration = " | ".join(hooks.stages)
    assert "4 of 4 work items pending" in narration
    assert "checking model openai/gpt-5-mini" in narration
    assert "starting rollout server" in narration
    assert "rollout server healthy on port" in narration
    assert "running 4 work items" in narration


async def test_progress_opens_before_the_first_result(harness: RunnerHarness) -> None:
    """A rollout can take minutes; the display must exist during that wait."""
    hooks = RecordingHooks()
    await harness.runner(hooks=hooks).run()
    assert hooks.progress_snapshots[0].completed == 0
    assert hooks.progress_snapshots[0].total == 4


async def test_declining_confirmation_dispatches_nothing(
    harness: RunnerHarness,
) -> None:
    hooks = RecordingHooks(refuse_confirmation=True)
    with pytest.raises(RuntimeError, match="declined"):
        await harness.runner(hooks=hooks).run()
    assert harness.index_rows() == []
    # The run directory and manifest exist, so the next attempt resumes cleanly.
    assert (harness.run_dir() / "manifest.json").is_file()


async def test_secrets_are_requested_only_when_work_is_pending(
    harness: RunnerHarness,
) -> None:
    hooks = RecordingHooks()
    await harness.runner(
        hooks=hooks, spec=harness.spec(secret_names=("MY_TOKEN",))
    ).run()
    assert hooks.secret_requests == [["MY_TOKEN"]]


async def test_a_second_supervisor_is_refused(harness: RunnerHarness) -> None:
    from osmosis_ai.eval.local.state import RunLock, RunLockedError

    lock_path = harness.output_root / ".locks" / "run-1.lock"
    with RunLock(lock_path):
        with pytest.raises(RunLockedError, match="already holds"):
            await harness.runner().run()


async def test_the_rollout_server_child_does_not_outlive_the_run(
    harness: RunnerHarness,
) -> None:
    await harness.runner().run()
    logs = (harness.run_dir() / "logs.txt").read_text()
    port_line = next(
        line for line in logs.splitlines() if "rollout server healthy" in line
    )
    port = json.loads(port_line[port_line.index("{") :])["port"]
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        assert sock.connect_ex(("127.0.0.1", port)) != 0


async def test_an_unnamed_run_gets_a_generated_directory(
    harness: RunnerHarness,
) -> None:
    summary = await harness.runner(options=LocalEvalOptions()).run()
    assert re.fullmatch(r"[a-z]+-[a-z]+-\d{1,2}", summary.run_name)
    assert summary.run_dir.name == summary.run_name
    assert (summary.run_dir / "index.jsonl").is_file()


async def test_the_manifest_records_provenance_but_no_secret_values(
    harness: RunnerHarness,
) -> None:
    hooks = RecordingHooks(secrets={"MY_TOKEN": "super-secret-value"})
    await harness.runner(
        hooks=hooks, spec=harness.spec(secret_names=("MY_TOKEN",))
    ).run()
    run_dir = harness.run_dir()
    manifest_text = (run_dir / "manifest.json").read_text()
    assert "super-secret-value" not in manifest_text
    assert "MY_TOKEN" in manifest_text
    assert "super-secret-value" not in (run_dir / "logs.txt").read_text()
    assert "super-secret-value" not in (run_dir / "events.jsonl").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["provenance"]["sdk_version"]
    assert manifest["schema_version"] == 1


def test_the_e2e_rollout_project_is_isolated_from_the_repo_venv(
    rollout_project: Path,
) -> None:
    # The supervisor launches the entrypoint with the current interpreter, so a
    # rollout project needs no virtualenv of its own.
    assert not (rollout_project / ".venv").exists()
    assert os.access(rollout_project / "main.py", os.R_OK)


# --------------------------------------------------------------------------- #
# Process-wide failures must not stamp the queue failed (§9.3)
# --------------------------------------------------------------------------- #


async def test_a_rejected_model_key_halts_dispatch_and_leaves_work_pending(
    harness: RunnerHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The queue says nothing about a bad provider key. Stamping the remaining
    # items `failed` would make a plain resume skip work that never executed.
    from tests.unit.rollout.openai_stub import FORCE_AUTH_ERROR_KEY

    monkeypatch.setenv("OPENAI_API_KEY", FORCE_AUTH_ERROR_KEY)
    runner = harness.runner()
    hooks = RecordingHooks()
    runner._hooks = hooks

    from osmosis_ai.eval.local.runner import LocalEvalError

    with pytest.raises(LocalEvalError, match="preflight failed"):
        await runner.run()
    # Preflight fails before any dispatch, so nothing is journaled at all.
    assert harness.journal_lines() == []

    # With a working key again, every item is still pending.
    monkeypatch.setenv("OPENAI_API_KEY", "stub-llm-key")
    summary = await harness.runner().run()
    assert summary.dispatched == 4
    assert summary.succeeded == 4


async def test_a_dead_rollout_server_halts_instead_of_waiting_out_deadlines(
    harness: RunnerHarness,
) -> None:
    runner = harness.runner(options=LocalEvalOptions(name="run-1", max_in_flight=1))
    original_run_item = runner._run_work_item
    killed = {"done": False}

    async def kill_after_first(item: Any, driver: Any) -> None:
        if killed["done"]:
            return await original_run_item(item, driver)
        result = await original_run_item(item, driver)
        killed["done"] = True
        child = runner._child
        assert child is not None
        child.kill()
        return result

    runner._run_work_item = kill_after_first  # type: ignore[method-assign]
    summary = await runner.run()

    # One item completed; the rest are pending, not failed, and the run did not
    # sit on a callback that can no longer arrive.
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(harness.journal_lines()) == 1
    logs = (harness.run_dir() / "logs.txt").read_text()
    assert "halting dispatch" in logs


def _refuse_admission(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    """Refuse every admission the way a POST /rollout *status_code* would."""

    async def refused(self: Any, init: Any) -> None:
        raise RolloutProtocolError(
            f"POST /rollout returned {status_code}; only 202 and 429 are accepted",
            status_code=status_code,
        )

    monkeypatch.setattr(HttpRolloutDriver, "_admit", refused)


@pytest.mark.parametrize(
    "admission_fault",
    [
        pytest.param(
            RolloutProtocolError(
                "POST /rollout returned 503; only 202 and 429 are accepted",
                status_code=503,
            ),
            id="5xx",
        ),
        pytest.param(
            RolloutAdmissionTimeoutError("rollout was not admitted within 1.0 seconds"),
            id="admission-timeout",
        ),
    ],
)
async def test_a_process_wide_admission_fault_halts_and_the_rows_resume(
    harness: RunnerHarness,
    monkeypatch: pytest.MonkeyPatch,
    admission_fault: Exception,
) -> None:
    # A restarting server or backpressure that outlives the admission budget
    # says nothing about the rows themselves: they must stay pending rather
    # than earn durable failed records, and a later invocation picks them up.
    async def failing(self: Any, init: Any) -> None:
        raise admission_fault

    with monkeypatch.context() as admission_patch:
        admission_patch.setattr(HttpRolloutDriver, "_admit", failing)
        hooks = RecordingHooks()
        summary = await harness.runner(hooks=hooks).run()

        assert summary.failed == 0
        assert harness.journal_lines() == []
        assert summary.cancelled is True
        assert any("stopping dispatch" in note for note in hooks.notes)

    resumed = await harness.runner().run()
    assert resumed.succeeded == 4


async def test_a_4xx_admission_is_a_terminal_row_failure(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server refused *these* requests, so each one is attributable.
    _refuse_admission(monkeypatch, 422)
    summary = await harness.runner().run()

    assert summary.failed == 4
    assert summary.cancelled is False
    assert {row["error_type"] for row in harness.index_rows()} == {
        "rollout_protocol_error"
    }


async def test_a_crashed_worker_is_surfaced_instead_of_a_silent_short_run(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.eval.local.runner import LocalEvalRunner

    async def exploding_journal(self: Any, item: Any, exc: BaseException) -> None:
        raise OSError("journal write failed")

    _refuse_admission(monkeypatch, 422)
    monkeypatch.setattr(
        LocalEvalRunner, "_journal_supervisor_failure", exploding_journal
    )
    summary = await harness.runner().run()

    assert harness.journal_lines() == []
    assert summary.cancelled is True
    assert "a worker failed: OSError" in (harness.run_dir() / "logs.txt").read_text()


async def test_a_retried_item_is_not_marked_resumed(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSMOSIS_TEST_GRADER_CRASH", "1")
    selection = select_rows(harness.dataset.path, row_selector=(0,))
    await harness.runner(selection=selection).run()
    assert harness.index_rows()[0].get("resumed") is None

    monkeypatch.delenv("OSMOSIS_TEST_GRADER_CRASH")
    await harness.runner(
        selection=selection,
        options=LocalEvalOptions(name="run-1", retry_failed=True),
    ).run()
    row = harness.index_rows()[0]
    assert row["status"] == "success"
    # `resumed` is the platform's carry-forward flag; this result was produced now.
    assert "resumed" not in row


async def test_secret_values_never_reach_logs_txt(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-live-supersecretvalue123"
    monkeypatch.setenv("OSMOSIS_TEST_ECHO_SECRET", "1")
    hooks = RecordingHooks(secrets={"MY_TOKEN": secret})
    await harness.runner(
        hooks=hooks, spec=harness.spec(secret_names=("MY_TOKEN",))
    ).run()
    run_dir = harness.run_dir()
    logs = (run_dir / "logs.txt").read_text()
    assert secret not in logs
    assert "[REDACTED]" in logs
    assert secret not in (run_dir / "manifest.json").read_text()
    assert secret not in (run_dir / "events.jsonl").read_text()


async def test_a_server_that_fails_startup_is_reported_immediately(
    harness: RunnerHarness,
) -> None:
    # A rollout server that dies during startup must not cost the full health
    # timeout on every port attempt.
    (harness.rollout_dir / "main.py").write_text(
        'raise SystemExit("startup exploded")\n', encoding="utf-8"
    )
    import time as _time

    from osmosis_ai.eval.local.runner import LocalEvalError

    started = _time.monotonic()
    with pytest.raises(LocalEvalError, match="before becoming healthy"):
        await harness.runner().run()
    assert _time.monotonic() - started < 30
    assert harness.journal_lines() == []


class _FakeTunnel:
    """Stands in for CloudflaredTunnel: 'publishes' the listener's own URL.

    The advertised URL equals the loopback URL, so the whole tunnel-mode
    plumbing (bridge keepalive on, advertised chat base, watchdog, teardown)
    runs against a reachable endpoint without any real tunnel.
    """

    instances: list[_FakeTunnel] = []

    def __init__(self, *, local_url: str, **_hooks: Any) -> None:
        import asyncio

        self._local_url = local_url
        self.on_spawn = _hooks.get("on_spawn")
        self._exited = asyncio.Event()
        self._returncode: int | None = None
        self.stopped = False
        self.public_url: str | None = None
        self.verified = True
        _FakeTunnel.instances.append(self)

    async def start(self) -> str:
        self.public_url = self._local_url
        return self._local_url

    async def wait(self) -> int | None:
        await self._exited.wait()
        return self._returncode

    def die(self, returncode: int) -> None:
        self._returncode = returncode
        self._exited.set()

    async def stop(self) -> bool:
        self.stopped = True
        self._exited.set()
        return True


async def test_tunnel_mode_runs_the_full_path_and_stops_the_tunnel(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.eval.local import runner as runner_module

    _FakeTunnel.instances.clear()
    monkeypatch.setattr(runner_module, "CloudflaredTunnel", _FakeTunnel)
    summary = await harness.runner(
        options=LocalEvalOptions(name="run-1", tunnel="cloudflared")
    ).run()
    assert summary.succeeded == 4
    (tunnel,) = _FakeTunnel.instances
    assert tunnel.stopped
    assert callable(tunnel.on_spawn)
    assert harness.hooks.statuses == [
        "waiting for cloudflared URL and public readiness (up to 30s)"
    ]
    healthy_index = next(
        index
        for index, stage in enumerate(harness.hooks.stages)
        if "rollout server healthy" in stage
    )
    tunnel_index = next(
        index
        for index, stage in enumerate(harness.hooks.stages)
        if "starting cloudflared" in stage
    )
    assert healthy_index < tunnel_index
    assert any("tunnel ready" in stage for stage in harness.hooks.stages)


async def test_rollout_server_failure_does_not_start_a_tunnel(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.eval.local import runner as runner_module
    from osmosis_ai.eval.local.runner import LocalEvalError

    _FakeTunnel.instances.clear()
    monkeypatch.setattr(runner_module, "CloudflaredTunnel", _FakeTunnel)
    runner = harness.runner(
        options=LocalEvalOptions(name="run-1", tunnel="cloudflared")
    )

    async def fail_server(**kwargs: object) -> str:
        raise LocalEvalError("server failed")

    monkeypatch.setattr(runner, "_start_rollout_server", fail_server)
    with pytest.raises(LocalEvalError, match="server failed"):
        await runner.run()

    assert _FakeTunnel.instances == []


async def test_tunnel_death_halts_dispatch_and_leaves_work_pending(
    harness: RunnerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.eval.local import runner as runner_module

    _FakeTunnel.instances.clear()

    class _DyingTunnel(_FakeTunnel):
        async def start(self) -> str:
            url = await super().start()
            self.die(1)
            return url

    monkeypatch.setattr(runner_module, "CloudflaredTunnel", _DyingTunnel)
    # Slow rollouts guarantee the watchdog fires while work is in flight.
    spec = harness.spec(env={"OSMOSIS_TEST_WORKFLOW_SLEEP": "30"})
    summary = await harness.runner(
        spec=spec,
        options=LocalEvalOptions(name="run-1", tunnel="cloudflared"),
    ).run()
    assert summary.cancelled
    assert summary.succeeded == 0
    assert any("cloudflared tunnel exited" in note for note in harness.hooks.notes)
    # Nothing was stamped failed: the items stay pending for a resume.
    assert harness.journal_lines() == []

"""The crash-recovery acceptance gates (§17).

A durably acknowledged terminal result must never run again across ``kill -9``
or Ctrl-C; unacknowledged work must. Both are exercised against a real
supervisor in its own process, because that is the only way to remove the
supervisor without letting it clean up after itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osmosis_ai.eval.local.dataset import select_rows
from osmosis_ai.eval.local.state import RunLock

from .conftest import RecordingHooks, RunnerHarness

pytestmark = pytest.mark.slow

_SUPERVISOR_SCRIPT = '''\
"""Run one local eval supervisor, driven entirely by argv/env. Test-only."""

import asyncio
import sys
from pathlib import Path

from osmosis_ai.eval.local.dataset import resolve_explicit_dataset_file, select_rows
from osmosis_ai.eval.local.runner import (
    EvalRunSpec,
    LocalEvalOptions,
    LocalEvalRunner,
)


class Hooks:
    def note(self, message):
        print(f"NOTE {message}", flush=True)

    def confirm_dispatch(self, *, pending, model_path):
        print(f"PENDING {pending}", flush=True)

    def resolve_secrets(self, names):
        return {}

    def progress(self, snapshot):
        print(f"PROGRESS {snapshot.completed}/{snapshot.total}", flush=True)


async def main() -> int:
    rollout_dir, output_root, dataset_path, proxy_url, proxy_token = sys.argv[1:6]
    dataset_file = Path(dataset_path)
    runner = LocalEvalRunner(
        spec=EvalRunSpec(
            rollout_name="echo-rollout",
            entrypoint="main.py",
            model_path="openai/gpt-5-mini",
            dataset_name="echo",
            n=1,
            pass_threshold=1.0,
            agent_timeout_sec=60.0,
            grader_timeout_sec=30.0,
            batch_size=1,
        ),
        options=LocalEvalOptions(name="run-1", yes=True, max_in_flight=1),
        dataset=resolve_explicit_dataset_file(dataset_file),
        selection=select_rows(dataset_file),
        rollout_dir=Path(rollout_dir),
        output_root=Path(output_root),
        proxy_base_url=proxy_url,
        proxy_auth_token=proxy_token,
        hooks=Hooks(),
        config_stem="echo-eval",
    )
    try:
        summary = await runner.run()
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
        return 130
    print(f"DONE succeeded={summary.succeeded} cancelled={summary.cancelled}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
'''


def _write_supervisor_script(tmp_path: Path) -> Path:
    path = tmp_path / "run_supervisor.py"
    path.write_text(_SUPERVISOR_SCRIPT, encoding="utf-8")
    return path


def _spawn_supervisor(
    script: Path, harness: RunnerHarness, *, workflow_sleep: str
) -> subprocess.Popen[str]:
    env = {
        **os.environ,
        "OSMOSIS_TEST_WORKFLOW_SLEEP": workflow_sleep,
        "PYTHONUNBUFFERED": "1",
    }
    return subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(harness.rollout_dir),
            str(harness.output_root),
            str(harness.dataset.path),
            harness.proxy_base_url,
            harness.proxy_token,
        ],
        cwd=str(Path(__file__).resolve().parents[4]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _journal_records(harness: RunnerHarness) -> list[dict[str, object]]:
    path = harness.run_dir() / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and line.endswith("}")
    ]


async def _wait_for_journal(
    harness: RunnerHarness,
    *,
    count: int,
    process: subprocess.Popen[str] | None = None,
    timeout: float = 60.0,
) -> list[dict[str, object]]:
    """Poll the journal without blocking: the stub proxy shares this event loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = _journal_records(harness)
        if len(records) >= count:
            return records
        if process is not None and process.poll() is not None:
            break
        await asyncio.sleep(0.05)
    output = ""
    if process is not None and process.stdout is not None:
        with contextlib.suppress(Exception):
            output = process.stdout.read() or ""
    raise AssertionError(
        f"journal never reached {count} records "
        f"(got {len(_journal_records(harness))}); supervisor output:\n{output}"
    )


async def _await_exit(process: subprocess.Popen[str], *, timeout: float) -> int | None:
    """Reap the supervisor off-loop so the stub proxy keeps serving."""
    try:
        return await asyncio.to_thread(process.wait, timeout)
    except subprocess.TimeoutExpired:
        return None


def _kill_group(process: subprocess.Popen[str], sig: int) -> None:
    os.killpg(os.getpgid(process.pid), sig)


async def test_kill_9_never_reruns_a_durably_acknowledged_item(
    tmp_path: Path, harness: RunnerHarness
) -> None:
    script = _write_supervisor_script(tmp_path)
    # One worker and a slow workflow, so exactly the first item lands durably.
    process = _spawn_supervisor(script, harness, workflow_sleep="1.5")
    try:
        durable = await _wait_for_journal(harness, count=1, process=process)
        _kill_group(process, signal.SIGKILL)
    finally:
        await _await_exit(process, timeout=30)

    survivors = _journal_records(harness)
    assert len(survivors) >= 1
    committed = {(r["row_index"], r["run_index"]) for r in survivors}
    committed_ids = {r["rollout_id"] for r in survivors}
    assert (durable[0]["row_index"], durable[0]["run_index"]) in committed

    # Resume in-process. The killed supervisor left the rollout server alive, so
    # this also exercises verified orphan reaping through the lock metadata.
    hooks = RecordingHooks()
    summary = await harness.runner(hooks=hooks).run()

    assert summary.dispatched == 4 - len(survivors)
    assert summary.resumed == len(survivors)
    assert summary.succeeded + summary.failed + summary.skipped == 4
    # Every previously committed attempt keeps its original rollout id: it was
    # never re-executed.
    final_ids = {row["rollout_id"] for row in harness.index_rows()}
    assert committed_ids <= final_ids


async def test_sigint_leaves_cancelled_work_pending(
    tmp_path: Path, harness: RunnerHarness
) -> None:
    script = _write_supervisor_script(tmp_path)
    process = _spawn_supervisor(script, harness, workflow_sleep="2.0")
    try:
        await _wait_for_journal(harness, count=1, process=process)
        _kill_group(process, signal.SIGINT)
        exit_code = await _await_exit(process, timeout=90)
    finally:
        if process.poll() is None:
            _kill_group(process, signal.SIGKILL)
            await _await_exit(process, timeout=30)

    assert exit_code in (0, 130)
    committed = _journal_records(harness)
    # A cancelled attempt writes no terminal record, unlike Harbor's
    # CancelledError result which is then skipped as complete.
    assert len(committed) < 4

    hooks = RecordingHooks()
    summary = await harness.runner(hooks=hooks).run()
    assert summary.dispatched == 4 - len(committed)
    assert summary.succeeded + summary.failed + summary.skipped == 4
    assert len(harness.index_rows()) == 4


async def test_a_partial_journal_record_is_truncated_on_resume(
    harness: RunnerHarness,
) -> None:
    selection = select_rows(harness.dataset.path, row_selector=(0, 1))
    await harness.runner(selection=selection).run()
    journal = harness.run_dir() / "events.jsonl"
    committed = journal.read_bytes()

    # Simulate a crash mid-append: a complete-looking record with no newline.
    journal.write_bytes(committed + b'{"row_index": 1, "run_index": 0, "roll')

    hooks = RecordingHooks()
    summary = await harness.runner(hooks=hooks, selection=selection).run()
    assert summary.dispatched == 0
    assert journal.read_bytes() == committed
    logs = (harness.run_dir() / "logs.txt").read_text()
    assert "discarded a partial trailing journal record" in logs


async def test_the_lock_records_the_child_while_the_run_is_live(
    harness: RunnerHarness,
) -> None:
    """The orphan-cleanup metadata must be written and fsynced right after spawn."""
    observed: list[object] = []
    runner = harness.runner()
    original = runner._start_rollout_server

    async def spying_start(*, secrets, lock, client):
        base_url = await original(secrets=secrets, lock=lock, client=client)
        observed.append(lock.read_child())
        return base_url

    runner._start_rollout_server = spying_start  # type: ignore[method-assign]
    await runner.run()

    assert len(observed) == 1
    record = observed[0]
    assert record is not None
    assert record.child_pid > 0  # type: ignore[union-attr]
    assert record.instance_id  # type: ignore[union-attr]
    # Cleared again after a confirmed clean exit.
    with RunLock(harness.output_root / ".locks" / "run-1.lock") as lock:
        assert lock.read_child() is None

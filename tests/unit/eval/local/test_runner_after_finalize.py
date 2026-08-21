"""The eval-run upload callback executes before the supervisor lock is released."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osmosis_ai.eval.local.runner import LocalEvalRunner, RunSummary
from osmosis_ai.eval.local.state import LOCKS_DIRNAME, RunLock, RunLockedError


@pytest.mark.asyncio
async def test_after_finalize_runs_under_the_same_run_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = LocalEvalRunner.__new__(LocalEvalRunner)
    runner._options = SimpleNamespace(name="run-1")  # type: ignore[attr-defined]
    runner._output_root = tmp_path  # type: ignore[attr-defined]
    summary = RunSummary(
        run_dir=tmp_path / "run-1",
        local_run_id="a" * 32,
        run_name="run-1",
        total_work_items=1,
        dispatched=1,
        succeeded=1,
        failed=0,
        skipped=0,
        resumed=0,
        cancelled=False,
    )

    async def run_locked(*, run_name: str) -> RunSummary:
        assert run_name == "run-1"
        return summary

    monkeypatch.setattr(runner, "_run_locked", run_locked)
    lock_path = tmp_path / LOCKS_DIRNAME / "run-1.lock"
    callback_observed_lock: list[bool] = []

    def after_finalize(_summary: RunSummary) -> None:
        try:
            with RunLock(lock_path):
                pass
        except RunLockedError:
            callback_observed_lock.append(True)

    result = await runner.run(after_finalize=after_finalize)

    assert result is summary
    assert callback_observed_lock == [True]
    with RunLock(lock_path):
        pass

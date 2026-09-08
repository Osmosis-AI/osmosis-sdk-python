"""Retention of Harbor's native per-trial logs (§4.3, §2.7)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor.artifacts import (
    retain_trial_logs,
)
from osmosis_ai.rollout.backend.harbor.backend import HarborBackend
from osmosis_ai.rollout.backend.harbor.trial import TRIAL_NAME_PREFIX, PendingTrial

ROLLOUT_ID = "a" * 32
ARTIFACTS_LOGGER = "osmosis_ai.rollout.backend.harbor.artifacts"


def _trial_dir(trials_dir: Path) -> Path:
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{ROLLOUT_ID}"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "trial.log").write_text("trial started\n")
    (trial_dir / "agent" / "agent.log").write_text("agent said hi\n")
    (trial_dir / "verifier" / "test-stdout.txt").write_text("1 passed\n")
    return trial_dir


def _backend(tmp_path: Path, trials_dir: Path, artifact_root: Path) -> HarborBackend:
    """A backend wired to this test's directories, cleanup left at its default."""
    task = tmp_path / "template-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "task.toml").write_text('[task]\nname = "template-task"\n')
    backend = HarborBackend(
        orchestrator=TrialQueue(n_concurrent=1),
        tasks_dir=task,
        agent="terminus-2",
        trials_dir=trials_dir,
    )
    backend.rollouts_dir = tmp_path / "rollouts"
    backend.artifact_root = artifact_root
    return backend


def _retention_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == ARTIFACTS_LOGGER
    ]


def test_trial_logs_land_beside_the_canonical_artifacts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trials_dir = tmp_path / "trials"
    artifact_root = tmp_path / "rollout_trials"
    _trial_dir(trials_dir)

    with caplog.at_level(logging.INFO, logger=ARTIFACTS_LOGGER):
        assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True

    logs = artifact_root / ROLLOUT_ID / "logs"
    assert (logs / "trial.log").read_text() == "trial started\n"
    assert (logs / "agent" / "agent.log").read_text() == "agent said hi\n"
    assert (logs / "verifier" / "test-stdout.txt").read_text() == "1 passed\n"
    # Never under artifacts/: that tree is enumerated wholesale by the platform.
    assert not (artifact_root / ROLLOUT_ID / "artifacts").exists()
    # The reported count is the files actually written, not "something ran".
    assert _retention_logs(caplog) == [
        f"Retained 3 Harbor trial log file(s) for rollout {ROLLOUT_ID}"
    ]


def test_an_exception_file_is_retained_when_present(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    (trial_dir / "exception.txt").write_text("boom\n")
    artifact_root = tmp_path / "rollout_trials"

    retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID)
    assert (
        artifact_root / ROLLOUT_ID / "logs" / "exception.txt"
    ).read_text() == "boom\n"


def test_retention_survives_the_trial_directory_being_removed(tmp_path: Path) -> None:
    import shutil

    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    artifact_root = tmp_path / "rollout_trials"

    retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID)
    # cleanup_successful_trials removes the trial dir; the logs must outlive it.
    shutil.rmtree(trial_dir)
    assert (artifact_root / ROLLOUT_ID / "logs" / "trial.log").is_file()


def test_a_missing_trial_directory_is_not_an_error(tmp_path: Path) -> None:
    assert retain_trial_logs(tmp_path / "trials", tmp_path / "out", ROLLOUT_ID) is True
    assert not (tmp_path / "out").exists()


def test_only_the_known_diagnostic_entries_are_copied(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    # Harbor state files are not logs and stay out of the retained set.
    (trial_dir / "config.json").write_text("{}")
    (trial_dir / "result.json").write_text("{}")
    artifact_root = tmp_path / "rollout_trials"

    retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID)
    logs = artifact_root / ROLLOUT_ID / "logs"
    assert sorted(p.name for p in logs.iterdir()) == ["agent", "trial.log", "verifier"]


def test_retention_is_idempotent(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    _trial_dir(trials_dir)
    artifact_root = tmp_path / "rollout_trials"
    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True
    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True
    assert (
        artifact_root / ROLLOUT_ID / "logs" / "trial.log"
    ).read_text() == "trial started\n"


def test_a_symlinked_trial_log_is_skipped(tmp_path: Path) -> None:
    # These files cross the sandbox trust boundary; following a link would copy
    # host content into a durable, user-visible location.
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    (trial_dir / "trial.log").unlink()
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE")
    (trial_dir / "trial.log").symlink_to(secret)
    artifact_root = tmp_path / "rollout_trials"

    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True
    logs = artifact_root / ROLLOUT_ID / "logs"
    assert not (logs / "trial.log").exists()
    assert (logs / "agent" / "agent.log").is_file()


def test_a_symlinked_destination_file_is_replaced_not_followed(
    tmp_path: Path,
) -> None:
    # copyfile opens the destination for writing, so a link planted at
    # logs/<name> would otherwise truncate whatever it points at.
    trials_dir = tmp_path / "trials"
    _trial_dir(trials_dir)
    artifact_root = tmp_path / "rollout_trials"
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    logs = artifact_root / ROLLOUT_ID / "logs"
    logs.mkdir(parents=True)
    (logs / "trial.log").symlink_to(victim)

    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True
    assert victim.read_text() == "precious"
    assert not (logs / "trial.log").is_symlink()
    assert (logs / "trial.log").read_text() == "trial started\n"


def test_a_symlinked_destination_component_is_refused(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    _trial_dir(trials_dir)
    artifact_root = tmp_path / "rollout_trials"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (artifact_root / ROLLOUT_ID).mkdir(parents=True)
    (artifact_root / ROLLOUT_ID / "logs").symlink_to(elsewhere)

    # Best-effort: retention reports failure rather than writing through the link.
    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is False
    assert not (elsewhere / "trial.log").exists()


async def test_a_failed_retention_keeps_the_trial_directory(tmp_path: Path) -> None:
    """Cleanup must not delete the only surviving copy of these logs.

    ``artifacts/`` is relocated separately, so the trial directory is all that
    is left of Harbor's own agent/verifier output once retention fails.
    """
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    (trial_dir / "artifacts").mkdir()
    (trial_dir / "artifacts" / "out.txt").write_text("scored\n")
    artifact_root = tmp_path / "rollout_trials"
    # The same planted link as above: retention cannot write anything durable.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (artifact_root / ROLLOUT_ID).mkdir(parents=True)
    (artifact_root / ROLLOUT_ID / "logs").symlink_to(elsewhere)
    backend = _backend(tmp_path, trials_dir, artifact_root)
    assert backend.cleanup_successful_trials is True

    backend.archive_trial(
        ROLLOUT_ID,
        SimpleNamespace(exception_info=None),
        PendingTrial(),
    )

    assert (trial_dir / "trial.log").read_text() == "trial started\n"
    assert (trial_dir / "agent" / "agent.log").is_file()
    assert not (elsewhere / "trial.log").exists()
    # The artifacts still reach durable storage; only the trial dir is kept.
    assert (
        artifact_root / ROLLOUT_ID / "artifacts" / "out.txt"
    ).read_text() == "scored\n"


def test_a_symlinked_log_directory_is_not_followed(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    import shutil as _shutil

    _shutil.rmtree(trial_dir / "agent")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "leaked.txt").write_text("PRIVATE")
    (trial_dir / "agent").symlink_to(tmp_path / "outside")
    artifact_root = tmp_path / "rollout_trials"

    retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID)
    assert not (artifact_root / ROLLOUT_ID / "logs" / "agent" / "leaked.txt").exists()


def test_multi_step_trials_retain_their_per_step_logs(tmp_path: Path) -> None:
    """Harbor relocates agent/verifier into steps/<name>/ for a multi-step trial.

    Retaining only the root entries would leave such a run with nothing but
    trial.log once cleanup_successful_trials removes the trial directory.
    """
    trials_dir = tmp_path / "trials"
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{ROLLOUT_ID}"
    (trial_dir / "steps" / "solve" / "agent").mkdir(parents=True)
    (trial_dir / "steps" / "solve" / "verifier").mkdir(parents=True)
    (trial_dir / "steps" / "check" / "agent").mkdir(parents=True)
    (trial_dir / "trial.log").write_text("multi-step\n")
    (trial_dir / "steps" / "solve" / "agent" / "agent.log").write_text("step 1\n")
    (trial_dir / "steps" / "solve" / "verifier" / "test-stdout.txt").write_text("ok\n")
    (trial_dir / "steps" / "check" / "agent" / "agent.log").write_text("step 2\n")
    artifact_root = tmp_path / "rollout_trials"

    assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True

    logs = artifact_root / ROLLOUT_ID / "logs"
    assert (logs / "trial.log").read_text() == "multi-step\n"
    assert (logs / "steps" / "solve" / "agent" / "agent.log").read_text() == "step 1\n"
    assert (
        logs / "steps" / "solve" / "verifier" / "test-stdout.txt"
    ).read_text() == "ok\n"
    assert (logs / "steps" / "check" / "agent" / "agent.log").read_text() == "step 2\n"


def test_a_symlinked_steps_directory_is_not_walked(tmp_path: Path) -> None:
    trials_dir = tmp_path / "trials"
    trial_dir = _trial_dir(trials_dir)
    (tmp_path / "outside" / "agent").mkdir(parents=True)
    (tmp_path / "outside" / "agent" / "leaked.txt").write_text("PRIVATE")
    (trial_dir / "steps").symlink_to(tmp_path / "outside")
    artifact_root = tmp_path / "rollout_trials"

    retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID)
    assert not (artifact_root / ROLLOUT_ID / "logs" / "steps").exists()


def test_a_trial_with_nothing_to_retain_reports_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Harbor leaves the mount targets behind even when they stay empty; an INFO
    # here would send a reader looking for files that were never written.
    trials_dir = tmp_path / "trials"
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{ROLLOUT_ID}"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    artifact_root = tmp_path / "rollout_trials"

    with caplog.at_level(logging.INFO, logger=ARTIFACTS_LOGGER):
        # An empty trial is not a failure: it must not block cleanup.
        assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True

    assert _retention_logs(caplog) == []


def test_a_skipped_entry_alone_reports_nothing_retained(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trials_dir = tmp_path / "trials"
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{ROLLOUT_ID}"
    trial_dir.mkdir(parents=True)
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE")
    (trial_dir / "trial.log").symlink_to(secret)
    artifact_root = tmp_path / "rollout_trials"

    with caplog.at_level(logging.INFO, logger=ARTIFACTS_LOGGER):
        assert retain_trial_logs(trials_dir, artifact_root, ROLLOUT_ID) is True

    assert not (artifact_root / ROLLOUT_ID / "logs" / "trial.log").exists()
    assert [m for m in _retention_logs(caplog) if m.startswith("Retained")] == []

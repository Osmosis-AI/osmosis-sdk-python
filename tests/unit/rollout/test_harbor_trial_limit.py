"""Harbor trial limit: the platform's number, the rollout's own queue, startup log."""

import asyncio
import logging

import pytest
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor.backend import HarborBackend
from osmosis_ai.rollout.backend.harbor.trial_limit import resolve_trial_limit
from osmosis_ai.rollout.server import create_rollout_server

PLATFORM_ENV = "_OSMOSIS_ROLLOUT_CONCURRENCY"


@pytest.fixture
def template_task(tmp_path):
    task = tmp_path / "template-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "task.toml").write_text('[task]\nname = "template-task"\n')
    return task


@pytest.mark.parametrize(
    ("platform", "own", "value", "level", "message"),
    [
        (
            "32",
            None,
            32,
            logging.INFO,
            "Trial limit: 32 at once (set by the platform for this run).",
        ),
        (
            "32",
            8,
            8,
            logging.WARNING,
            "Trial limit: 8 at once, set by this rollout's code "
            "(TrialQueue(n_concurrent=8)). The platform allows 32 for this run, so "
            "it will run 8 at a time instead of 32. To use the platform's number, "
            "remove the orchestrator= argument from HarborBackend.",
        ),
        (
            "32",
            128,
            32,
            logging.INFO,
            "Trial limit: 32 at once. This rollout's code asks for 128; the "
            "platform allows 32 for this run.",
        ),
        (
            None,
            None,
            8,
            logging.INFO,
            "Trial limit: 8 at once (SDK default for local runs; eval batch_size "
            "is capped to match).",
        ),
        (
            None,
            16,
            16,
            logging.INFO,
            "Trial limit: 16 at once, set by this rollout's code "
            "(TrialQueue(n_concurrent=16)).",
        ),
    ],
    ids=["platform", "own-lower", "own-higher", "local-default", "local-own"],
)
def test_resolve_trial_limit(monkeypatch, platform, own, value, level, message):
    if platform is None:
        monkeypatch.delenv(PLATFORM_ENV, raising=False)
    else:
        monkeypatch.setenv(PLATFORM_ENV, platform)
    queue = None if own is None else TrialQueue(n_concurrent=own)

    limit = resolve_trial_limit(queue)

    assert (limit.value, limit.log_level, limit.message) == (value, level, message)


def test_zero_platform_limit_refuses_to_start(monkeypatch):
    monkeypatch.setenv(PLATFORM_ENV, "0")
    with pytest.raises(ValueError, match="must be at least 1, got 0"):
        resolve_trial_limit(None)


def test_zero_trial_limit_refuses_to_start(monkeypatch, template_task):
    monkeypatch.setenv(PLATFORM_ENV, "32")
    with pytest.raises(ValueError) as error:
        HarborBackend(
            orchestrator=TrialQueue(n_concurrent=0),
            tasks_dir=template_task,
            agent="terminus-2",
        )
    assert str(error.value) == (
        "Trial limit must be at least 1, got 0 (TrialQueue(n_concurrent=0))."
    )


def test_backend_without_queue_runs_the_platform_number(monkeypatch, template_task):
    monkeypatch.setenv(PLATFORM_ENV, "32")
    backend = HarborBackend(tasks_dir=template_task, agent="terminus-2")

    assert backend.orchestrator._n_concurrent == 32
    assert backend.health()["concurrency"] == {"max_concurrent": 32}


async def test_platform_number_lowers_a_larger_queue(monkeypatch, template_task):
    monkeypatch.setenv(PLATFORM_ENV, "2")
    queue = TrialQueue(n_concurrent=128)
    in_flight = 0
    peak = 0

    async def tracked_submit(config):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    monkeypatch.setattr(queue, "submit", tracked_submit)
    backend = HarborBackend(
        orchestrator=queue, tasks_dir=template_task, agent="terminus-2"
    )

    await asyncio.gather(*(backend.run_trial(None) for _ in range(6)))

    assert peak == 2


def test_server_logs_the_trial_limit_at_startup(monkeypatch, template_task, caplog):
    monkeypatch.setenv(PLATFORM_ENV, "32")
    backend = HarborBackend(
        orchestrator=TrialQueue(n_concurrent=8),
        tasks_dir=template_task,
        agent="terminus-2",
    )

    with caplog.at_level(logging.INFO):
        create_rollout_server(backend=backend, configure_logging=False)

    [record] = [r for r in caplog.records if r.getMessage().startswith("Trial limit")]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == backend.trial_limit.message

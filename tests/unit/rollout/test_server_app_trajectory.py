from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server.app import _handle_rollout
from osmosis_ai.rollout.types import (
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    RolloutInitRequest,
    RolloutSample,
    RolloutStatus,
)


def make_request() -> RolloutInitRequest:
    return RolloutInitRequest(
        rollout_id="r1",
        initial_messages=[{"role": "user", "content": "hi"}],
        chat_completions_url="http://controller/chat/completions",
        extra_fields={"eval_run_id": "er-1", "row_index": 0},
    )


def sample(reward: float | None = None) -> RolloutSample:
    return RolloutSample(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        reward=reward,
    )


class StubBackend(ExecutionBackend):
    def __init__(self, outcome: ExecutionOutcome) -> None:
        self.outcome = outcome

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        return self.outcome


def patch_artifact_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: root
    )


async def test_records_graded_result(tmp_path: Path, monkeypatch) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        ExecutionOutcome(
            workflow=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample()),
            grader=ExecutionResult(
                status=RolloutStatus.SUCCESS, sample=sample(reward=0.7)
            ),
        )
    )

    response = await _handle_rollout(backend, make_request())

    assert response.status is RolloutStatus.SUCCESS
    assert response.sample is not None and response.sample.reward == 0.7
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["extra"]["osmosis"]["reward"] == 0.7
    assert doc["extra"]["osmosis"]["request_extra_fields"]["eval_run_id"] == "er-1"


async def test_archive_keeps_workflow_sample_when_grader_has_none(
    tmp_path: Path, monkeypatch
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        ExecutionOutcome(
            workflow=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample()),
            grader=ExecutionResult(
                status=RolloutStatus.FAILURE, err_message="grading failed"
            ),
        )
    )

    response = await _handle_rollout(backend, make_request())

    assert response.status is RolloutStatus.FAILURE
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert len(doc["steps"]) == 2


async def test_failed_result_does_not_expose_a_reward(
    tmp_path: Path, monkeypatch
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        ExecutionOutcome(
            workflow=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample()),
            grader=ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=sample(reward=0.7),
                err_message="grading failed",
            ),
        )
    )

    response = await _handle_rollout(backend, make_request())

    assert response.sample is not None
    assert response.sample.reward is None

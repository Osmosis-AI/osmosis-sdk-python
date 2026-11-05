from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server.app import _handle_rollout, create_rollout_server
from osmosis_ai.rollout.types import (
    POLLING_LEASE_HEADER,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
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


@pytest.mark.parametrize("workflow_has_sample", [False, True])
async def test_workflow_error_preserves_available_sample(
    tmp_path: Path, monkeypatch, workflow_has_sample: bool
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    workflow_sample = sample() if workflow_has_sample else None
    grader_sample = sample(reward=0.7)
    workflow = ExecutionResult(
        status=RolloutStatus.FAILURE,
        sample=workflow_sample,
        err_message="agent failed",
        err_category=RolloutErrorCategory.AGENT_ERROR,
    )
    outcome = ExecutionOutcome(
        workflow=workflow,
        grader=ExecutionResult(
            status=RolloutStatus.FAILURE,
            sample=grader_sample,
            err_message="grading failed",
            err_category=RolloutErrorCategory.VALIDATION_ERROR,
        ),
    )

    response = await _handle_rollout(StubBackend(outcome), make_request())

    assert response.status is RolloutStatus.FAILURE
    assert response.err_message == "agent failed"
    assert response.err_category is RolloutErrorCategory.AGENT_ERROR
    assert response.sample is not None
    assert response.sample.messages == sample().messages
    assert response.sample.reward is None
    assert outcome.result.sample is (workflow_sample or grader_sample)
    assert workflow.sample is workflow_sample
    assert grader_sample.reward == 0.7


@pytest.mark.parametrize("bad_field", ["trajectory_messages", "messages"])
@pytest.mark.parametrize("bad_value", [object(), float("nan"), "\ud800"])
async def test_unserializable_diagnostics_do_not_lose_the_reward(
    tmp_path: Path, monkeypatch, bad_field: str, bad_value: object
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    original = sample(reward=0.7)
    setattr(original, bad_field, [{"role": "assistant", "content": bad_value}])
    result = ExecutionResult(status=RolloutStatus.SUCCESS, sample=original)
    app = create_rollout_server(backend=StubBackend(ExecutionOutcome(workflow=result)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            admission = await client.post(
                "/rollout", json=make_request().model_dump(mode="json")
            )
            response = await client.get(
                "/rollout/r1/result",
                headers={POLLING_LEASE_HEADER: admission.json()["polling_lease_token"]},
            )
            while response.json()["status"] in {"queued", "running", "grading"}:
                response = await client.get(
                    "/rollout/r1/result",
                    headers={
                        POLLING_LEASE_HEADER: admission.json()["polling_lease_token"]
                    },
                )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["sample"]["reward"] == 0.7
    assert "trajectory_messages" not in payload["sample"]
    if bad_field == "trajectory_messages":
        assert payload["sample"]["messages"] == list(sample().messages)
    elif isinstance(bad_value, float):
        assert payload["sample"]["messages"] == [{"role": "assistant", "content": None}]
    else:
        assert payload["sample"]["messages"] == []
    assert getattr(original, bad_field)[0]["content"] is bad_value


async def test_invalid_unicode_in_labels_and_errors_is_still_pollable(
    tmp_path: Path, monkeypatch
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    original = RolloutSample(
        messages=sample().messages,
        reward=0.7,
        label="label-\ud800",
        metrics={"value": "\ud800"},
        extra_fields={"value": "\ud800"},
    )
    result = ExecutionResult(
        status=RolloutStatus.FAILURE, sample=original, err_message="error-\ud800"
    )
    app = create_rollout_server(backend=StubBackend(ExecutionOutcome(workflow=result)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            admission = await client.post(
                "/rollout", json=make_request().model_dump(mode="json")
            )
            response = await client.get(
                "/rollout/r1/result",
                headers={POLLING_LEASE_HEADER: admission.json()["polling_lease_token"]},
            )
            while response.json()["status"] in {"queued", "running", "grading"}:
                response = await client.get(
                    "/rollout/r1/result",
                    headers={
                        POLLING_LEASE_HEADER: admission.json()["polling_lease_token"]
                    },
                )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failure"
    assert payload["err_message"] == "error-?"
    assert payload["sample"]["label"] == "label-?"
    assert payload["sample"]["reward"] is None
    assert original.label == "label-\ud800"


class BlockingBackend(ExecutionBackend):
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        await self.release.wait()
        return ExecutionOutcome(
            workflow=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample(0.7))
        )


async def finish_rollout(client: httpx.AsyncClient) -> None:
    admission = await client.post(
        "/rollout", json=make_request().model_dump(mode="json")
    )
    headers = {POLLING_LEASE_HEADER: admission.json()["polling_lease_token"]}
    response = await client.get("/rollout/r1/result", headers=headers)
    while response.json()["status"] in {"queued", "running", "grading"}:
        response = await client.get("/rollout/r1/result", headers=headers)


@pytest.mark.parametrize(
    ("result_sample", "status_code"),
    [(sample(reward=0.7), 200), (None, 204)],
    ids=["saved", "none-saved"],
)
async def test_trajectory_route_serves_the_saved_document(
    tmp_path: Path, monkeypatch, result_sample, status_code
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    result = ExecutionResult(status=RolloutStatus.SUCCESS, sample=result_sample)
    app = create_rollout_server(backend=StubBackend(ExecutionOutcome(workflow=result)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            await finish_rollout(client)
            response = await client.get("/rollout/r1/trajectory")

    assert response.status_code == status_code
    if status_code == 200:
        assert response.content == (tmp_path / "r1" / "trajectory.json").read_bytes()


async def test_trajectory_route_rejects_unknown_and_unfinished_rollouts(
    tmp_path: Path, monkeypatch
) -> None:
    patch_artifact_root(monkeypatch, tmp_path)
    backend = BlockingBackend()
    app = create_rollout_server(backend=backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://rollout"
        ) as client:
            unknown = await client.get("/rollout/r1/trajectory")
            await client.post("/rollout", json=make_request().model_dump(mode="json"))
            running = await client.get("/rollout/r1/trajectory")
            backend.release.set()

    assert unknown.status_code == 404
    assert running.status_code == 409

"""Tests for the trajectory saving wiring in the rollout server app."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server.app import _handle_rollout
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutInitRequest,
    RolloutSample,
    RolloutStatus,
)


def make_request(**overrides: Any) -> RolloutInitRequest:
    payload: dict[str, Any] = {
        "rollout_id": "r1",
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": "http://controller/chat/completions",
        "completion_callback_url": "http://controller/v1/rollout/completed",
        "grader_callback_url": "http://controller/v1/grader/completed",
        "extra_fields": {"eval_run_id": "er-1", "row_index": 0},
    }
    payload.update(overrides)
    return RolloutInitRequest(**payload)


def make_sample(reward: float | None = None) -> RolloutSample:
    return RolloutSample(
        id="s1",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        reward=reward,
    )


class StubBackend(ExecutionBackend):
    """Invokes callbacks like a real backend, then optionally raises."""

    def __init__(
        self,
        workflow_result: ExecutionResult,
        grader_result: ExecutionResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.workflow_result = workflow_result
        self.grader_result = grader_result
        self.raises = raises
        self.received_grader_callback: bool | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        self.received_grader_callback = on_grader_complete is not None
        await on_workflow_complete(self.workflow_result)
        if on_grader_complete and self.grader_result is not None:
            await on_grader_complete(self.grader_result)
        if self.raises is not None:
            raise self.raises


def patch_callbacks(
    monkeypatch, ack_bodies: dict[str, dict[str, Any]] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Stub callback posts; ``ack_bodies`` maps URL to the controller's ack body."""
    posted: list[tuple[str, dict[str, Any]]] = []

    async def fake_post(
        *, url: str, payload: dict[str, Any], headers: Any = None
    ) -> Any:
        posted.append((url, payload))
        body = (ack_bodies or {}).get(url, {"ok": True})
        return SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr("osmosis_ai.rollout.server.app.post_json_with_retry", fake_post)
    return posted


def patch_artifact_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: root
    )


async def test_records_graded_result(tmp_path: Path, monkeypatch) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample()}
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample(reward=0.7)}
        ),
    )

    await _handle_rollout(backend, make_request())

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["extra"]["osmosis"]["reward"] == 0.7
    assert doc["extra"]["osmosis"]["request_extra_fields"]["eval_run_id"] == "er-1"
    assert [url for url, _ in posted] == [
        "http://controller/v1/rollout/completed",
        "http://controller/v1/grader/completed",
    ]


async def test_records_workflow_result_without_grader_callback(
    tmp_path: Path, monkeypatch
) -> None:
    patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample()}
        ),
    )

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    assert backend.received_grader_callback is False
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert "reward" not in doc["extra"]["osmosis"]


async def test_grader_failure_without_samples_keeps_workflow_samples(
    tmp_path: Path, monkeypatch
) -> None:
    patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample()}
        ),
        grader_result=ExecutionResult(status=RolloutStatus.FAILURE),
    )

    await _handle_rollout(backend, make_request())

    assert (tmp_path / "r1" / "trajectory.json").exists()


async def test_records_even_when_backend_raises(tmp_path: Path, monkeypatch) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample()}
        ),
        raises=RuntimeError("boom"),
    )

    await _handle_rollout(backend, make_request())

    assert (tmp_path / "r1" / "trajectory.json").exists()
    failure_payloads = [p for _, p in posted if p.get("status") == "failure"]
    assert failure_payloads


async def test_backend_failure_without_samples_records_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(status=RolloutStatus.FAILURE),
        raises=RuntimeError("boom"),
    )

    await _handle_rollout(backend, make_request())

    assert not (tmp_path / "r1").exists()
    assert posted


async def test_callback_ack_report_lands_in_document(
    tmp_path: Path, monkeypatch
) -> None:
    # The grader ack carries per-call metrics; the completion ack does not.
    patch_callbacks(
        monkeypatch,
        ack_bodies={
            "http://controller/v1/grader/completed": {
                "ok": True,
                "trajectory": {
                    "model_name": "openai/gpt-5-mini",
                    "samples": {
                        "s1": {
                            "llm_call_metrics": [
                                {"prompt_tokens": 12, "completion_tokens": 4}
                            ]
                        }
                    },
                },
            }
        },
    )
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample()}
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, samples={"s1": make_sample(reward=1.0)}
        ),
    )

    await _handle_rollout(backend, make_request())

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["agent"]["model_name"] == "openai/gpt-5-mini"
    agent_steps = [s for s in doc["steps"] if s["source"] == "agent"]
    assert agent_steps[0]["metrics"]["prompt_tokens"] == 12
    assert doc["final_metrics"]["total_completion_tokens"] == 4

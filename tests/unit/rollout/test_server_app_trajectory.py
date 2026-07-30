"""Tests for the trajectory saving wiring in the rollout server app."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.backend.native_harbor import backend as native_backend_module
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
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample(reward=0.7)
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
    grader_payload = posted[1][1]
    assert grader_payload["sample"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert "trajectory_messages" not in grader_payload["sample"]


async def test_result_extra_fields_are_posted_and_archived(
    tmp_path: Path, monkeypatch
) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    diagnostics = {
        "backend": "native_harbor",
        "phase": "agent",
        "harbor_exception_type": "AgentTimeoutError",
        "category": "timeout",
        "timings_sec": {"agent": 42.0},
    }
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.FAILURE,
            sample=make_sample(),
            extra_fields=diagnostics,
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.FAILURE,
            sample=make_sample(),
            extra_fields=diagnostics,
        ),
    )

    await _handle_rollout(backend, make_request())

    completion_payload = posted[0][1]
    assert completion_payload["extra_fields"] == diagnostics
    grader_payload = posted[1][1]
    assert grader_payload["extra_fields"] == diagnostics
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["extra"]["osmosis"]["result_extra_fields"] == diagnostics
    assert json.loads((tmp_path / "r1" / "diagnostics.json").read_text()) == (
        diagnostics
    )


async def test_result_extra_fields_are_archived_without_a_sample(
    tmp_path: Path, monkeypatch
) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    diagnostics = {
        "backend": "native_harbor",
        "phase": "setup",
        "harbor_exception_type": "RuntimeError",
        "category": "agent_error",
    }
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.FAILURE,
            extra_fields=diagnostics,
        )
    )

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    assert posted[0][1]["extra_fields"] == diagnostics
    assert json.loads((tmp_path / "r1" / "diagnostics.json").read_text()) == (
        diagnostics
    )
    assert not (tmp_path / "r1" / "trajectory.json").exists()


async def test_native_late_failure_without_grader_url_keeps_final_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = NativeHarborBackend(trials_dir=tmp_path / "trials")
    backend.artifact_root = tmp_path
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 0.7}),
        step_results=None,
        exception_info=None,
    )

    async def fake_submit(_queue: Any, trial_config: Any) -> Any:
        hook_event = SimpleNamespace(
            trial_name=trial_config.trial_name,
            result=result,
        )
        await backend._on_verification_started(hook_event)
        result.exception_info = SimpleNamespace(
            exception_message="verifier timed out",
            exception_type="VerifierTimeoutError",
        )
        await backend._on_trial_ended(hook_event)
        return result

    monkeypatch.setattr(native_backend_module.TrialQueue, "submit", fake_submit)

    await _handle_rollout(
        backend,
        make_request(
            grader_callback_url=None,
            metadata={"harbor_task": "/tmp/task"},
        ),
    )

    assert len(posted) == 1
    assert posted[0][1]["status"] == "success"
    diagnostics = json.loads((tmp_path / "r1" / "diagnostics.json").read_text())
    assert diagnostics["phase"] == "verification"
    assert diagnostics["harbor_exception_type"] == "VerifierTimeoutError"
    assert diagnostics["category"] == "agent_error"


async def test_records_workflow_result_without_grader_callback(
    tmp_path: Path, monkeypatch
) -> None:
    patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
    )

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    assert backend.received_grader_callback is False
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert "reward" not in doc["extra"]["osmosis"]


async def test_grader_failure_without_sample_keeps_workflow_sample(
    tmp_path: Path, monkeypatch
) -> None:
    patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
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
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        raises=RuntimeError("boom"),
    )

    await _handle_rollout(backend, make_request())

    assert (tmp_path / "r1" / "trajectory.json").exists()
    failure_payloads = [p for _, p in posted if p.get("status") == "failure"]
    assert failure_payloads


async def test_failure_completion_ack_report_lands_in_document(
    tmp_path: Path, monkeypatch
) -> None:
    # The backend dies after the workflow completed; only the ack of the
    # resulting failure completion callback carries the controller's metrics.
    report_body = {
        "trajectory": {
            "model_name": "openai/gpt-5-mini",
            "samples": {"s1": {"llm_call_metrics": [{"prompt_tokens": 12}]}},
        }
    }

    async def fake_post(
        *, url: str, payload: dict[str, Any], headers: Any = None
    ) -> Any:
        body = report_body if payload.get("status") == "failure" else {"ok": True}
        return SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr("osmosis_ai.rollout.server.app.post_json_with_retry", fake_post)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = StubBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        raises=RuntimeError("boom"),
    )

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["agent"]["model_name"] == "openai/gpt-5-mini"
    agent_steps = [s for s in doc["steps"] if s["source"] == "agent"]
    assert agent_steps[0]["metrics"]["prompt_tokens"] == 12


async def test_backend_failure_without_sample_records_nothing(
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
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample(reward=1.0)
        ),
    )

    await _handle_rollout(backend, make_request())

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["agent"]["model_name"] == "openai/gpt-5-mini"
    agent_steps = [s for s in doc["steps"] if s["source"] == "agent"]
    assert agent_steps[0]["metrics"]["prompt_tokens"] == 12
    assert doc["final_metrics"]["total_completion_tokens"] == 4

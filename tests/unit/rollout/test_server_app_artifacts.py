"""Tests that _handle_rollout sanitizes and threads artifacts onto the grader
callback body, and that absent artifacts keep the callback byte-identical."""

from __future__ import annotations

from typing import Any

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server import app as app_module
from osmosis_ai.rollout.server.app import _handle_rollout
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutInitRequest,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.artifacts import MAX_ARTIFACTS_BYTES

GRADER_URL = "http://controller/grader"
COMPLETE_URL = "http://controller/complete"


class GraderCallbackBackend(ExecutionBackend):
    """Fires the workflow + grader callbacks with a caller-supplied result."""

    def __init__(self, grader_result: ExecutionResult) -> None:
        self.grader_result = grader_result

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        await on_workflow_complete(ExecutionResult(status=RolloutStatus.SUCCESS))
        if on_grader_complete:
            await on_grader_complete(self.grader_result)


def _init_request() -> RolloutInitRequest:
    return RolloutInitRequest(
        rollout_id="r1",
        initial_messages=[{"role": "user", "content": "hi"}],
        label="test-label",
        chat_completions_url="http://llm",
        completion_callback_url=COMPLETE_URL,
        grader_callback_url=GRADER_URL,
    )


async def _run(monkeypatch, grader_result: ExecutionResult) -> dict[str, Any]:
    posted: list[tuple[str, dict[str, Any]]] = []

    async def _record(*, url, payload, headers):
        posted.append((url, payload))

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr(app_module, "post_json_with_retry", _record)
    await _handle_rollout(GraderCallbackBackend(grader_result), _init_request())

    grader_payloads = [p for (url, p) in posted if url == GRADER_URL]
    assert len(grader_payloads) == 1
    return grader_payloads[0]


class TestGraderCallbackArtifacts:
    async def test_valid_artifacts_threaded(self, monkeypatch):
        artifacts = {"judge": {"explanation": "ok"}}
        result = ExecutionResult(status=RolloutStatus.SUCCESS, artifacts=artifacts)
        payload = await _run(monkeypatch, result)
        assert payload["artifacts"] == artifacts

    async def test_absent_artifacts_drops_key_byte_identical(self, monkeypatch):
        result = ExecutionResult(status=RolloutStatus.SUCCESS)
        payload = await _run(monkeypatch, result)

        assert "artifacts" not in payload
        # Other fields keep emitting their current nulls (bare model_dump).
        assert payload["err_message"] is None
        assert payload["err_category"] is None
        assert payload == {
            "rollout_id": "r1",
            "status": "success",
            "samples": {},
            "err_message": None,
            "err_category": None,
        }

    async def test_oversized_artifacts_sanitized_to_error(self, monkeypatch):
        artifacts = {"blob": "x" * (MAX_ARTIFACTS_BYTES + 1)}
        result = ExecutionResult(status=RolloutStatus.SUCCESS, artifacts=artifacts)
        payload = await _run(monkeypatch, result)
        assert payload["artifacts"]["_error"]["code"] == "artifacts_too_large"

    async def test_artifacts_on_grader_failure_callback(self, monkeypatch):
        artifacts = {"judge": {"explanation": "failed because ..."}}
        result = ExecutionResult(status=RolloutStatus.FAILURE, artifacts=artifacts)
        payload = await _run(monkeypatch, result)
        assert payload["status"] == "failure"
        assert payload["artifacts"] == artifacts

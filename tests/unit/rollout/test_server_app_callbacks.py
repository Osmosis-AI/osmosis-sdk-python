"""The server's terminal-callback contract: exactly one per channel.

A completion callback closes out ``rollout_id`` on the controller. The server's
error path exists so a backend that dies without reporting cannot leave the
controller waiting — not to append a second, contradicting verdict to a rollout
that already reported one.
"""

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

COMPLETION_URL = "http://controller/v1/rollout/completed"
GRADER_URL = "http://controller/v1/grader/completed"


def make_request(**overrides: Any) -> RolloutInitRequest:
    payload: dict[str, Any] = {
        "rollout_id": "r1",
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": "http://controller/chat/completions",
        "completion_callback_url": COMPLETION_URL,
        "grader_callback_url": GRADER_URL,
    }
    payload.update(overrides)
    return RolloutInitRequest(**payload)


def make_sample(reward: float | None = None) -> RolloutSample:
    return RolloutSample(
        messages=[{"role": "assistant", "content": "hello"}], reward=reward
    )


class ScriptedBackend(ExecutionBackend):
    """Delivers the given results, then optionally dies."""

    def __init__(
        self,
        *,
        workflow_result: ExecutionResult | None = None,
        grader_result: ExecutionResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.workflow_result = workflow_result
        self.grader_result = grader_result
        self.raises = raises

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        if self.workflow_result is not None:
            await on_workflow_complete(self.workflow_result)
        if on_grader_complete and self.grader_result is not None:
            await on_grader_complete(self.grader_result)
        if self.raises is not None:
            raise self.raises


def patch_callbacks(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    posted: list[tuple[str, dict[str, Any]]] = []

    async def fake_post(
        *, url: str, payload: dict[str, Any], headers: Any = None
    ) -> Any:
        # The real transport encodes JSON; a payload that cannot round-trip
        # here would raise on the wire instead.
        json.dumps(payload, allow_nan=False)
        posted.append((url, payload))
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True})

    monkeypatch.setattr("osmosis_ai.rollout.server.app.post_json_with_retry", fake_post)
    return posted


def patch_artifact_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: root
    )


def completions(posted: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [payload for url, payload in posted if url == COMPLETION_URL]


def graders(posted: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [payload for url, payload in posted if url == GRADER_URL]


async def test_success_completion_is_not_followed_by_a_failure(
    tmp_path: Path, monkeypatch
) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = ScriptedBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        raises=RuntimeError("grading blew up"),
    )

    await _handle_rollout(backend, make_request())

    assert [p["status"] for p in completions(posted)] == ["success"]


async def test_grader_failure_is_still_reported_after_a_successful_completion(
    tmp_path: Path, monkeypatch
) -> None:
    # The truthful report for "workflow finished, grading died" is one success
    # completion plus one grader failure — not a retracted completion.
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = ScriptedBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        raises=RuntimeError("grading blew up"),
    )

    await _handle_rollout(backend, make_request())

    assert [p["status"] for p in completions(posted)] == ["success"]
    assert [p["status"] for p in graders(posted)] == ["failure"]


async def test_failure_completion_is_not_duplicated(
    tmp_path: Path, monkeypatch
) -> None:
    # A second "Internal server error" would also overwrite the real diagnosis.
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = ScriptedBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.FAILURE, err_message="workflow exploded"
        ),
        raises=RuntimeError("and then the backend died"),
    )

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    assert [p["err_message"] for p in completions(posted)] == ["workflow exploded"]


async def test_completion_is_fabricated_when_nothing_was_reported(
    tmp_path: Path, monkeypatch
) -> None:
    # The safety net the guard must not remove.
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = ScriptedBackend(raises=RuntimeError("died on entry"))

    await _handle_rollout(backend, make_request(grader_callback_url=None))

    assert [p["status"] for p in completions(posted)] == ["failure"]
    assert completions(posted)[0]["err_message"] == "Internal server error"


async def test_grader_callback_is_not_duplicated(tmp_path: Path, monkeypatch) -> None:
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    backend = ScriptedBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample()
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS, sample=make_sample(reward=0.7)
        ),
        raises=RuntimeError("died after both callbacks"),
    )

    await _handle_rollout(backend, make_request())

    assert [p["status"] for p in graders(posted)] == ["success"]
    assert [p["status"] for p in completions(posted)] == ["success"]


async def test_non_finite_metrics_do_not_break_the_grader_payload(
    tmp_path: Path, monkeypatch
) -> None:
    # Graders write metrics by mutating the dict, past every validator; the
    # payload still has to be encodable or the whole callback is lost.
    posted = patch_callbacks(monkeypatch)
    patch_artifact_root(monkeypatch, tmp_path)
    sample = make_sample(reward=0.7)
    sample.metrics["ok"] = 1.5
    sample.metrics["ratio"] = float("nan")
    backend = ScriptedBackend(
        workflow_result=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample),
        grader_result=ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample),
    )

    await _handle_rollout(backend, make_request())

    payload = graders(posted)[0]
    assert payload["sample"]["metrics"] == {"ok": 1.5}
    assert payload["sample"]["reward"] == 0.7

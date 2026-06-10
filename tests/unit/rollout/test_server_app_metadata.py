"""Tests that _handle_rollout threads metadata into ExecutionRequest."""

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


class CapturingBackend(ExecutionBackend):
    """Records the ExecutionRequest it receives without running anything."""

    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        self.request = request


def _make_init_request(metadata: dict[str, Any] | None) -> RolloutInitRequest:
    return RolloutInitRequest(
        rollout_id="r1",
        initial_messages=[{"role": "user", "content": "hi"}],
        label="test-label",
        metadata=metadata,
        chat_completions_url="http://llm",
        completion_callback_url="http://controller/complete",
    )


class TestHandleRolloutMetadata:
    async def test_metadata_dict_threaded(self, monkeypatch):
        # No callbacks fire from CapturingBackend, but stub HTTP to be safe.
        async def _fail(*_args, **_kwargs):  # pragma: no cover - defensive
            raise AssertionError("post_json_with_retry should not be called")

        monkeypatch.setattr(app_module, "post_json_with_retry", _fail)

        backend = CapturingBackend()
        metadata = {"tools": ["search"], "difficulty": 3}
        await _handle_rollout(backend, _make_init_request(metadata))

        assert backend.request is not None
        assert backend.request.metadata == metadata

    async def test_metadata_none_threaded(self, monkeypatch):
        async def _fail(*_args, **_kwargs):  # pragma: no cover - defensive
            raise AssertionError("post_json_with_retry should not be called")

        monkeypatch.setattr(app_module, "post_json_with_retry", _fail)

        backend = CapturingBackend()
        await _handle_rollout(backend, _make_init_request(None))

        assert backend.request is not None
        assert backend.request.metadata is None

    async def test_failure_path_posts_error_callback(self, monkeypatch):
        """If the backend raises, metadata still does not break the error path."""
        posted: list[str] = []

        async def _record(*, url, payload, headers):
            posted.append(url)

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(app_module, "post_json_with_retry", _record)

        class FailingBackend(ExecutionBackend):
            async def execute(
                self, request, on_workflow_complete, on_grader_complete=None
            ):
                raise RuntimeError("boom")

        await _handle_rollout(FailingBackend(), _make_init_request({"k": "v"}))
        assert "http://controller/complete" in posted


def test_capturing_backend_smoke():
    """ExecutionResult import is exercised to keep the contract obvious."""
    result = ExecutionResult(status=RolloutStatus.SUCCESS)
    assert result.status == RolloutStatus.SUCCESS

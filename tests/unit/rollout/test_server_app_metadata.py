"""Tests that _handle_rollout threads metadata into ExecutionRequest."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.context import get_rollout_context
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


class ContextCapturingBackend(ExecutionBackend):
    def __init__(self) -> None:
        self.api_key: str | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        ctx = get_rollout_context()
        self.api_key = None if ctx is None else ctx.api_key
        await on_workflow_complete(ExecutionResult(status=RolloutStatus.SUCCESS))


class TestHandleRolloutApiKeys:
    async def test_llm_api_key_goes_to_context_controller_key_to_callbacks(
        self, monkeypatch
    ) -> None:
        posted_headers: list[dict[str, str] | None] = []

        async def _record(*, url, payload, headers):
            posted_headers.append(headers)

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(app_module, "post_json_with_retry", _record)
        backend = ContextCapturingBackend()
        request = RolloutInitRequest(
            rollout_id="r1",
            initial_messages=[{"role": "user", "content": "hi"}],
            chat_completions_url="http://proxy/v1/eval-sessions/r1",
            controller_api_key="controller-key",
            llm_api_key="session-token",
            completion_callback_url="http://controller/complete",
        )
        await _handle_rollout(backend, request)

        assert backend.api_key == "session-token"
        assert posted_headers
        assert posted_headers[0] == {"Authorization": "Bearer controller-key"}

    async def test_llm_api_key_falls_back_to_controller_api_key(
        self, monkeypatch
    ) -> None:
        async def _record(*, url, payload, headers):
            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(app_module, "post_json_with_retry", _record)
        backend = ContextCapturingBackend()
        await _handle_rollout(
            backend,
            RolloutInitRequest(
                rollout_id="r1",
                initial_messages=[{"role": "user", "content": "hi"}],
                chat_completions_url="http://llm",
                controller_api_key="shared-key",
                completion_callback_url="http://controller/complete",
            ),
        )
        assert backend.api_key == "shared-key"

    async def test_explicit_empty_llm_api_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RolloutInitRequest(
                rollout_id="r1",
                initial_messages=[{"role": "user", "content": "hi"}],
                chat_completions_url="http://llm",
                controller_api_key="shared-key",
                llm_api_key="",
                completion_callback_url="http://controller/complete",
            )

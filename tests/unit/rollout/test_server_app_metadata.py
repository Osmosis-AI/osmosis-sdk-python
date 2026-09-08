from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import get_rollout_context
from osmosis_ai.rollout.server import app as app_module
from osmosis_ai.rollout.server.app import _handle_rollout
from osmosis_ai.rollout.types import (
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    RolloutInitRequest,
    RolloutStatus,
)


def test_wire_required_fields() -> None:
    def required(model: type[BaseModel]) -> frozenset[str]:
        return frozenset(
            name for name, field in model.model_fields.items() if field.is_required()
        )

    assert required(RolloutInitRequest) == frozenset(
        {"rollout_id", "initial_messages", "chat_completions_url"}
    )


class CapturingBackend(ExecutionBackend):
    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None
        self.api_key: str | None = None

    async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        self.request = request
        ctx = get_rollout_context()
        self.api_key = None if ctx is None else ctx.api_key
        result = ExecutionResult(status=RolloutStatus.SUCCESS)
        return ExecutionOutcome(workflow=result, grader=result)


def make_request(metadata: dict[str, Any] | None = None) -> RolloutInitRequest:
    return RolloutInitRequest(
        rollout_id="r1",
        initial_messages=[{"role": "user", "content": "hi"}],
        label="test-label",
        metadata=metadata,
        chat_completions_url="http://llm",
        llm_api_key="session-token",
    )


@pytest.fixture(autouse=True)
def no_archive(monkeypatch):
    async def skip(**kwargs):
        return None

    monkeypatch.setattr(app_module, "save_trajectory", skip)


async def test_metadata_and_grade_are_threaded_to_backend() -> None:
    backend = CapturingBackend()
    await _handle_rollout(backend, make_request({"difficulty": 3}))
    assert backend.request is not None
    assert backend.request.metadata == {"difficulty": 3}
    assert backend.request.grade is True


async def test_llm_api_key_is_used_by_the_rollout_context() -> None:
    backend = CapturingBackend()
    await _handle_rollout(backend, make_request())
    assert backend.api_key == "session-token"


async def test_backend_exception_becomes_a_failure_result() -> None:
    class FailingBackend(ExecutionBackend):
        async def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
            raise RuntimeError("boom")

    response = await _handle_rollout(FailingBackend(), make_request())
    assert response.status is RolloutStatus.FAILURE
    assert response.err_message == "boom"


def test_empty_llm_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RolloutInitRequest(
            rollout_id="r1",
            initial_messages=[],
            chat_completions_url="http://llm",
            llm_api_key="",
        )

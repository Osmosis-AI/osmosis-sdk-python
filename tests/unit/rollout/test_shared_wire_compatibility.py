"""Compatibility checks for the shared remote-rollout wire subset.

Fixtures under ``tests/golden/rollout/shared_wire/`` capture the minimum JSON
contract accepted by remote rollout consumers. These tests stay repository-local
and verify that additive SDK changes preserve shared required fields and callback
status values.

The fixture omits ``llm_api_key`` to exercise the backwards-compatible
``controller_api_key`` fallback. ``RolloutStatus`` retains the richer SDK
lifecycle and intentionally excludes ``pending``; ``GraderStatus`` includes
``pending`` as part of its existing contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server import app as app_module
from osmosis_ai.rollout.server.app import _handle_rollout
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutInitRequest,
    RolloutStatus,
)

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "rollout" / "shared_wire"

SHARED_INIT_REQUIRED = frozenset(
    {
        "initial_messages",
        "chat_completions_url",
        "completion_callback_url",
    }
)
SHARED_CALLBACK_STATUSES = frozenset({"success", "failure"})


def _load(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


class _ContextCapturingBackend(ExecutionBackend):
    def __init__(self) -> None:
        self.api_key: str | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        from osmosis_ai.rollout.context import get_rollout_context

        ctx = get_rollout_context()
        self.api_key = None if ctx is None else ctx.api_key


def test_init_request_without_llm_api_key_is_valid() -> None:
    payload = _load("init_request.json")
    assert "llm_api_key" not in payload
    request = RolloutInitRequest.model_validate(payload)
    assert request.llm_api_key is None
    assert request.controller_api_key == "controller-key"
    assert request.initial_messages == [{"role": "user", "content": "hi"}]


async def test_omitted_llm_api_key_preserves_controller_api_key_fallback(
    monkeypatch,
) -> None:
    async def _record(*, url, payload, headers):
        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr(app_module, "post_json_with_retry", _record)
    backend = _ContextCapturingBackend()
    request = RolloutInitRequest.model_validate(_load("init_request.json"))
    await _handle_rollout(backend, request)
    assert backend.api_key == "controller-key"


def test_completion_success_and_failure_payloads_parse() -> None:
    success = RolloutCompleteRequest.model_validate(_load("completion_success.json"))
    failure = RolloutCompleteRequest.model_validate(_load("completion_failure.json"))
    assert success.status is RolloutStatus.SUCCESS
    assert failure.status is RolloutStatus.FAILURE
    assert failure.err_message == "agent exploded"
    assert failure.err_category is not None
    assert failure.err_category.value == "agent_error"


def test_grader_payload_with_sample_reward_is_compatible() -> None:
    request = GraderCompleteRequest.model_validate(_load("grader_with_reward.json"))
    assert request.status is GraderStatus.SUCCESS
    assert request.sample is not None
    assert request.sample.reward == 1.0
    assert request.sample.remove_sample is False


def test_additive_sdk_fields_do_not_change_shared_required_or_callback_statuses() -> (
    None
):
    init_required = {
        name
        for name, field in RolloutInitRequest.model_fields.items()
        if field.is_required()
    }
    assert init_required == SHARED_INIT_REQUIRED
    assert "llm_api_key" in RolloutInitRequest.model_fields
    assert not RolloutInitRequest.model_fields["llm_api_key"].is_required()

    complete_required = {
        name
        for name, field in RolloutCompleteRequest.model_fields.items()
        if field.is_required()
    }
    grader_required = {
        name
        for name, field in GraderCompleteRequest.model_fields.items()
        if field.is_required()
    }
    assert complete_required == {"status"}
    assert grader_required == {"status"}

    complete_statuses = {item.value for item in RolloutStatus}
    grader_statuses = {item.value for item in GraderStatus}
    assert complete_statuses >= SHARED_CALLBACK_STATUSES
    assert grader_statuses >= SHARED_CALLBACK_STATUSES
    # Keep the richer SDK rollout lifecycle distinct from grader pending state.
    assert "pending" not in complete_statuses

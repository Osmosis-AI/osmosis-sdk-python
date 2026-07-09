"""Untrusted rollout ids feed filesystem paths, so reject unsafe ones at the
request boundary."""

from typing import Any

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout.types import ExecutionRequest, RolloutInitRequest

UNSAFE_IDS = [
    "../other",
    "a/b",
    r"a\b",
    "/tmp/other",
    r"\tmp\other",
    "a\x00b",
    "..",
    ".",
    "",
]
SAFE_IDS = ["r1", "rollout-xyz", "abc_123", "550e8400-e29b-41d4-a716-446655440000"]


def _init_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rollout_id": "r1",
        "initial_messages": [{"role": "user", "content": "hi"}],
        "chat_completions_url": "http://controller/chat/completions",
        "completion_callback_url": "http://controller/v1/rollout/completed",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("rollout_id", UNSAFE_IDS)
def test_execution_request_rejects_unsafe_id(rollout_id: str) -> None:
    with pytest.raises(ValidationError):
        ExecutionRequest(id=rollout_id, prompt=[])


@pytest.mark.parametrize("rollout_id", SAFE_IDS)
def test_execution_request_accepts_safe_id(rollout_id: str) -> None:
    assert ExecutionRequest(id=rollout_id, prompt=[]).id == rollout_id


@pytest.mark.parametrize("rollout_id", UNSAFE_IDS)
def test_rollout_init_request_rejects_unsafe_id(rollout_id: str) -> None:
    with pytest.raises(ValidationError):
        RolloutInitRequest(**_init_payload(rollout_id=rollout_id))


@pytest.mark.parametrize("rollout_id", SAFE_IDS)
def test_rollout_init_request_accepts_safe_id(rollout_id: str) -> None:
    assert RolloutInitRequest(**_init_payload(rollout_id=rollout_id)).rollout_id == (
        rollout_id
    )

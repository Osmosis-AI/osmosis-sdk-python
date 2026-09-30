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


@pytest.mark.parametrize("timeout", [-10.0, -1.0, -0.001, 0.0, 0])
def test_rollout_init_request_rejects_non_positive_agent_timeout(
    timeout: float,
) -> None:
    with pytest.raises(ValidationError, match="timeout must be positive"):
        RolloutInitRequest(**_init_payload(agent_timeout_sec=timeout))


@pytest.mark.parametrize("timeout", [-10.0, -1.0, -0.001, 0.0, 0])
def test_rollout_init_request_rejects_non_positive_grader_timeout(
    timeout: float,
) -> None:
    with pytest.raises(ValidationError, match="timeout must be positive"):
        RolloutInitRequest(**_init_payload(grader_timeout_sec=timeout))


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_rollout_init_request_rejects_non_finite_timeouts(timeout: float) -> None:
    with pytest.raises(ValidationError, match="timeout must be finite"):
        RolloutInitRequest(**_init_payload(agent_timeout_sec=timeout))
    with pytest.raises(ValidationError, match="timeout must be finite"):
        RolloutInitRequest(**_init_payload(grader_timeout_sec=timeout))


@pytest.mark.parametrize("timeout", [0.05, 1.0, 30.0, 100, None])
def test_rollout_init_request_accepts_valid_timeouts(timeout: float | None) -> None:
    req = RolloutInitRequest(
        **_init_payload(agent_timeout_sec=timeout, grader_timeout_sec=timeout)
    )
    assert req.agent_timeout_sec == timeout
    assert req.grader_timeout_sec == timeout

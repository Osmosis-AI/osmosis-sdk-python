"""Reusable factory / helper functions for tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from osmosis_ai.rollout import RolloutMetrics


def make_rollout_payload(
    rollout_id: str = "test-123",
    idempotency_key: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal valid RolloutRequest JSON payload."""
    payload: dict[str, Any] = {
        "rollout_id": rollout_id,
        "server_url": "http://localhost:8080",
        "messages": [{"role": "user", "content": "Hello"}],
        "completion_params": {"temperature": 0.7},
    }
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    payload.update(overrides)
    return payload


def mock_llm_client() -> MagicMock:
    """Create a mocked OsmosisLLMClient usable as an async context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.complete_rollout = AsyncMock()
    mock.get_metrics = MagicMock(return_value=RolloutMetrics())
    return mock

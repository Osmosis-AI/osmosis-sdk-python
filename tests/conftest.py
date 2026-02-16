"""Shared pytest fixtures for osmosis_ai tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionToolSchema,
    RolloutRequest,
)
from tests._data import DATA_DIR
from tests._helpers import MockAgentLoop


def _load_json(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text())


@pytest.fixture
def mock_agent_loop() -> MockAgentLoop:
    """Create a mock agent loop for testing."""
    return MockAgentLoop()


@pytest.fixture
def sample_rollout_request() -> RolloutRequest:
    """Create a sample RolloutRequest for testing."""
    return RolloutRequest(**_load_json("sample_rollout_request.json"))


@pytest.fixture
def sample_tool_schema() -> OpenAIFunctionToolSchema:
    """Create a sample OpenAIFunctionToolSchema for testing."""
    return OpenAIFunctionToolSchema(**_load_json("sample_tool_schema.json"))


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    """Create sample messages for testing."""
    return _load_json("sample_messages.json")["system_user_pair"]


@pytest.fixture
def sample_assistant_message() -> dict[str, Any]:
    """Create a sample assistant message for testing."""
    return _load_json("sample_messages.json")["assistant_response"]


@pytest.fixture
def sample_assistant_message_with_tool_calls() -> dict[str, Any]:
    """Create a sample assistant message with tool calls."""
    return _load_json("sample_messages.json")["assistant_with_tool_calls"]


@pytest.fixture
def sample_tool_message() -> dict[str, Any]:
    """Create a sample tool result message."""
    return _load_json("sample_messages.json")["tool_result"]

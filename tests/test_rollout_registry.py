# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.registry."""

from __future__ import annotations

from typing import List

import pytest

from osmosis_ai.rollout import (
    AgentLoopNotFoundError,
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
)
from osmosis_ai.rollout.registry import (
    AgentLoopRegistry,
    _REGISTRY,
    get_agent_loop,
    list_agent_loops,
    register_agent_loop,
    unregister_agent_loop,
)


class TestAgentLoop(RolloutAgentLoop):
    """Test agent loop implementation."""

    name = "test_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


class AnotherAgentLoop(RolloutAgentLoop):
    """Another test agent loop."""

    name = "another_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


# =============================================================================
# AgentLoopRegistry Class Tests
# =============================================================================


def test_registry_register_and_get() -> None:
    """Verify registering and getting an agent loop."""
    registry = AgentLoopRegistry()
    loop = TestAgentLoop()

    registry.register(loop)

    retrieved = registry.get("test_agent")
    assert retrieved is loop


def test_registry_get_returns_none_for_unknown() -> None:
    """Verify get returns None for unknown agent loop."""
    registry = AgentLoopRegistry()

    result = registry.get("nonexistent")
    assert result is None


def test_registry_register_duplicate_raises() -> None:
    """Verify registering duplicate name raises ValueError."""
    registry = AgentLoopRegistry()
    loop1 = TestAgentLoop()
    loop2 = TestAgentLoop()

    registry.register(loop1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(loop2)


def test_registry_unregister_success() -> None:
    """Verify successful unregistration."""
    registry = AgentLoopRegistry()
    loop = TestAgentLoop()

    registry.register(loop)
    result = registry.unregister("test_agent")

    assert result is True
    assert registry.get("test_agent") is None


def test_registry_unregister_not_found() -> None:
    """Verify unregister returns False for not found."""
    registry = AgentLoopRegistry()

    result = registry.unregister("nonexistent")
    assert result is False


def test_registry_list_names() -> None:
    """Verify listing all registered names."""
    registry = AgentLoopRegistry()
    loop1 = TestAgentLoop()
    loop2 = AnotherAgentLoop()

    registry.register(loop1)
    registry.register(loop2)

    names = registry.list_names()
    assert "test_agent" in names
    assert "another_agent" in names
    assert len(names) == 2


def test_registry_list_names_empty() -> None:
    """Verify list_names returns empty list for empty registry."""
    registry = AgentLoopRegistry()

    names = registry.list_names()
    assert names == []


def test_registry_clear() -> None:
    """Verify clearing the registry."""
    registry = AgentLoopRegistry()
    loop = TestAgentLoop()

    registry.register(loop)
    assert len(registry.list_names()) == 1

    registry.clear()
    assert len(registry.list_names()) == 0


# =============================================================================
# Global Function Tests
# =============================================================================


@pytest.fixture(autouse=True)
def clean_global_registry():
    """Clean global registry before and after each test."""
    # Clear before test
    _REGISTRY.clear()
    yield
    # Clear after test
    _REGISTRY.clear()


def test_global_register_agent_loop() -> None:
    """Verify global register_agent_loop function."""
    loop = TestAgentLoop()

    register_agent_loop(loop)

    assert _REGISTRY.get("test_agent") is loop


def test_global_register_duplicate_raises() -> None:
    """Verify global register raises for duplicates."""
    loop1 = TestAgentLoop()
    loop2 = TestAgentLoop()

    register_agent_loop(loop1)

    with pytest.raises(ValueError, match="already registered"):
        register_agent_loop(loop2)


def test_global_unregister_agent_loop_success() -> None:
    """Verify global unregister_agent_loop function."""
    loop = TestAgentLoop()
    register_agent_loop(loop)

    result = unregister_agent_loop("test_agent")

    assert result is True
    assert _REGISTRY.get("test_agent") is None


def test_global_unregister_agent_loop_not_found() -> None:
    """Verify global unregister returns False for not found."""
    result = unregister_agent_loop("nonexistent")
    assert result is False


def test_global_get_agent_loop_success() -> None:
    """Verify global get_agent_loop returns the loop."""
    loop = TestAgentLoop()
    register_agent_loop(loop)

    retrieved = get_agent_loop("test_agent")
    assert retrieved is loop


def test_global_get_agent_loop_not_found_raises() -> None:
    """Verify get_agent_loop raises AgentLoopNotFoundError."""
    with pytest.raises(AgentLoopNotFoundError) as exc_info:
        get_agent_loop("nonexistent")

    assert exc_info.value.name == "nonexistent"
    assert exc_info.value.available == []


def test_global_get_agent_loop_includes_available_in_error() -> None:
    """Verify error includes available agent loops."""
    loop = TestAgentLoop()
    register_agent_loop(loop)

    with pytest.raises(AgentLoopNotFoundError) as exc_info:
        get_agent_loop("wrong_name")

    assert exc_info.value.name == "wrong_name"
    assert "test_agent" in exc_info.value.available


def test_global_list_agent_loops() -> None:
    """Verify global list_agent_loops function."""
    loop1 = TestAgentLoop()
    loop2 = AnotherAgentLoop()

    register_agent_loop(loop1)
    register_agent_loop(loop2)

    names = list_agent_loops()
    assert "test_agent" in names
    assert "another_agent" in names


def test_global_list_agent_loops_empty() -> None:
    """Verify list_agent_loops returns empty list when empty."""
    names = list_agent_loops()
    assert names == []

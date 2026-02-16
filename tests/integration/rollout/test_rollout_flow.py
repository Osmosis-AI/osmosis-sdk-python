"""Integration tests for the complete rollout lifecycle.

Tests the full init -> chat -> completed flow using real FastAPI app
with httpx.AsyncClient, minimal mocking (only the LLM client).
"""

from __future__ import annotations

import pytest


class TestRolloutLifecycle:
    """Test the complete rollout request lifecycle."""

    @pytest.mark.integration
    async def test_init_chat_complete_flow(self):
        """Full rollout: init request -> agent runs -> completion reported."""
        pytest.skip("TODO: implement rollout lifecycle integration test")

    @pytest.mark.integration
    async def test_concurrent_rollouts(self):
        """Multiple rollouts can execute concurrently without interference."""
        pytest.skip("TODO: implement concurrent rollout integration test")

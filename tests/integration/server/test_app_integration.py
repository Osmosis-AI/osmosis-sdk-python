"""Integration tests for the FastAPI rollout server.

Tests HTTP endpoints using httpx.AsyncClient against a real create_app() instance.
Only the external LLM client is mocked.
"""

from __future__ import annotations

import pytest


class TestServerEndpoints:
    """Test server HTTP endpoints with real request/response cycle."""

    @pytest.mark.integration
    async def test_health_endpoint(self):
        """GET /health returns 200 with server status."""
        pytest.skip("TODO: implement health endpoint integration test")

    @pytest.mark.integration
    async def test_rollout_endpoint_validation(self):
        """POST /v1/rollout/init validates request schema."""
        pytest.skip("TODO: implement rollout validation integration test")

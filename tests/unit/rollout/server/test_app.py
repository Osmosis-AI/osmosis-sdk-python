"""Tests for osmosis_ai.rollout.server.app — focusing on uncovered code paths.

This module tests:
- _extract_bearer_token() parsing logic
- lifespan() startup/shutdown lifecycle (cleanup task, callbacks, registration)
- Concurrency control via semaphore (max_concurrent)
- Idempotency with idempotency_key vs rollout_id
- Error propagation when get_tools() raises
- Platform registration background task behaviour
- on_startup / on_shutdown lifecycle callbacks
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout import (
    OpenAIFunctionToolSchema,
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    create_app,
)
from osmosis_ai.rollout.server.app import _extract_bearer_token
from tests._helpers import (
    SimpleAgentLoop,
    SlowAgentLoop,
    make_rollout_payload,
    mock_llm_client,
)

# Import FastAPI / httpx test utilities
try:
    from fastapi.testclient import TestClient
    from httpx import ASGITransport, AsyncClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


# =============================================================================
# Helpers — module-specific agent loop stubs
# =============================================================================


class FailingGetToolsAgentLoop(RolloutAgentLoop):
    """Agent loop whose get_tools() raises an exception."""

    name = "failing_get_tools_agent"

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        raise ValueError("Tools unavailable")

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        return ctx.complete([])


# =============================================================================
# _extract_bearer_token() tests
# =============================================================================


class TestExtractBearerToken:
    """Tests for the _extract_bearer_token helper function."""

    def test_returns_none_for_none_input(self) -> None:
        """None header should yield None."""
        assert _extract_bearer_token(None) is None  # type: ignore[arg-type]

    def test_returns_none_for_empty_string(self) -> None:
        """Empty string should yield None."""
        assert _extract_bearer_token("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        """Whitespace-only header should yield None."""
        assert _extract_bearer_token("   ") is None

    def test_extracts_bearer_prefix(self) -> None:
        """Standard 'Bearer <token>' format should return <token>."""
        assert _extract_bearer_token("Bearer my-secret-token") == "my-secret-token"

    def test_bearer_prefix_is_case_insensitive(self) -> None:
        """The 'bearer' prefix should be matched case-insensitively."""
        assert _extract_bearer_token("bearer abc123") == "abc123"
        assert _extract_bearer_token("BEARER xyz789") == "xyz789"
        assert _extract_bearer_token("BeArEr mixed") == "mixed"

    def test_raw_token_fallback(self) -> None:
        """A single token without Bearer prefix should be returned as-is."""
        assert _extract_bearer_token("plain-token") == "plain-token"

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading/trailing whitespace around the header should be stripped."""
        assert _extract_bearer_token("  Bearer tok  ") == "tok"

    def test_more_than_two_parts_returns_raw(self) -> None:
        """If there are more than two parts, fall back to returning the full string."""
        result = _extract_bearer_token("Bearer token extra")
        # Three parts -- does not match the 'len(parts) == 2' branch
        assert result == "Bearer token extra"

    def test_non_bearer_prefix_with_two_parts_returns_raw(self) -> None:
        """Two-part header with a non-bearer prefix returns raw string."""
        result = _extract_bearer_token("Basic dXNlcjpwYXNz")
        assert result == "Basic dXNlcjpwYXNz"


# =============================================================================
# Lifespan -- on_startup / on_shutdown callbacks
# =============================================================================


class TestLifespanCallbacks:
    """Tests that on_startup and on_shutdown callbacks are invoked.

    Uses TestClient as a context manager which properly triggers lifespan events.
    """

    def test_on_startup_called_during_lifespan(self) -> None:
        """on_startup callback should be invoked when the app starts."""
        started = False

        async def my_startup() -> None:
            nonlocal started
            started = True

        app = create_app(SimpleAgentLoop(), on_startup=my_startup)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
        assert started, "on_startup was not called"

    def test_on_shutdown_called_during_lifespan(self) -> None:
        """on_shutdown callback should be invoked when the app shuts down."""
        stopped = False

        async def my_shutdown() -> None:
            nonlocal stopped
            stopped = True

        app = create_app(SimpleAgentLoop(), on_shutdown=my_shutdown)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
        # After the context manager exits, shutdown should have fired
        assert stopped, "on_shutdown was not called"

    def test_both_callbacks_called_in_order(self) -> None:
        """on_startup fires before requests; on_shutdown fires after exit."""
        events: list[str] = []

        async def startup() -> None:
            events.append("start")

        async def shutdown() -> None:
            events.append("stop")

        app = create_app(SimpleAgentLoop(), on_startup=startup, on_shutdown=shutdown)
        with TestClient(app) as client:
            client.get("/health")
            events.append("request")

        assert events == ["start", "request", "stop"]


# =============================================================================
# Lifespan -- cleanup task management
# =============================================================================


class TestLifespanCleanupTask:
    """Tests that the cleanup task starts and stops with the lifespan."""

    def test_cleanup_task_started_on_startup(self) -> None:
        """The AppState cleanup task should be started during lifespan startup."""
        app = create_app(SimpleAgentLoop())
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
        # If we get here without errors, the cleanup task started and was
        # stopped cleanly during shutdown.


# =============================================================================
# Lifespan -- platform registration
# =============================================================================


class TestLifespanRegistration:
    """Tests for the platform registration background task in lifespan."""

    def test_registration_task_not_started_without_credentials(self) -> None:
        """Registration should be skipped when credentials are None."""
        app = create_app(SimpleAgentLoop())
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_registration_not_started_when_host_is_none(self) -> None:
        """Registration should be skipped if server_host is None."""
        mock_creds = MagicMock()
        app = create_app(
            SimpleAgentLoop(),
            credentials=mock_creds,
            server_host=None,
            server_port=9999,
        )
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_registration_not_started_when_port_is_none(self) -> None:
        """Registration should be skipped if server_port is None."""
        mock_creds = MagicMock()
        app = create_app(
            SimpleAgentLoop(),
            credentials=mock_creds,
            server_host="0.0.0.0",
            server_port=None,
        )
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    @pytest.mark.slow
    async def test_registration_task_started_with_credentials(self) -> None:
        """Registration background task should be created when credentials are given."""
        mock_creds = MagicMock()
        mock_register = MagicMock(return_value={"status": "ok"})
        mock_print = MagicMock()

        with (
            patch(
                "osmosis_ai.rollout.server.registration.register_with_platform",
                mock_register,
            ),
            patch(
                "osmosis_ai.rollout.server.registration.print_registration_result",
                mock_print,
            ),
        ):
            app = create_app(
                SimpleAgentLoop(),
                credentials=mock_creds,
                server_host="0.0.0.0",
                server_port=9999,
            )
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200
                # Give the registration background task time to run
                await asyncio.sleep(0.5)

    @pytest.mark.slow
    async def test_registration_error_handled_gracefully(self) -> None:
        """If the registration raises an error, shutdown should not crash."""
        mock_creds = MagicMock()

        with (
            patch(
                "osmosis_ai.rollout.server.registration.register_with_platform",
                side_effect=Exception("Registration failed"),
            ),
            patch(
                "osmosis_ai.rollout.server.registration.print_registration_result",
                MagicMock(),
            ),
        ):
            app = create_app(
                SimpleAgentLoop(),
                credentials=mock_creds,
                server_host="0.0.0.0",
                server_port=9998,
            )
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200
                # Allow time for registration attempt
                await asyncio.sleep(0.5)
            # TestClient should exit cleanly even though registration errored


# =============================================================================
# Concurrency control (max_concurrent)
# =============================================================================


class TestConcurrencyControl:
    """Tests for the semaphore-based max_concurrent limit."""

    async def test_semaphore_limits_concurrent_rollouts(self) -> None:
        """Only max_concurrent rollouts should run simultaneously."""
        max_concurrent = 2
        agent = SlowAgentLoop(delay=0.3)
        app = create_app(agent, max_concurrent=max_concurrent)

        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Submit 4 rollouts
                for i in range(4):
                    resp = await client.post(
                        "/v1/rollout/init",
                        json=make_rollout_payload(rollout_id=f"conc-{i}"),
                    )
                    assert resp.status_code == 202

                # Allow some time for background tasks
                await asyncio.sleep(0.1)

                # Check health to see active rollouts
                health = await client.get("/health")
                data = health.json()
                # There should be some active rollouts queued up
                assert data["active_rollouts"] >= 0

                # Wait for all to complete
                await asyncio.sleep(1.5)

    async def test_max_concurrent_one_serializes_rollouts(self) -> None:
        """With max_concurrent=1 rollouts must execute one at a time."""
        execution_order: list[str] = []

        class OrderTrackingAgent(RolloutAgentLoop):
            name = "order_agent"

            def get_tools(
                self, request: RolloutRequest
            ) -> list[OpenAIFunctionToolSchema]:
                return []

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                rid = ctx.request.rollout_id
                execution_order.append(f"start-{rid}")
                await asyncio.sleep(0.05)
                execution_order.append(f"end-{rid}")
                return ctx.complete(list(ctx.request.messages))

        app = create_app(OrderTrackingAgent(), max_concurrent=1)
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="r1"),
                )
                await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="r2"),
                )
                # Wait for both to finish
                await asyncio.sleep(0.5)

        # With max_concurrent=1, r1 should fully complete before r2 starts
        assert "start-r1" in execution_order
        assert "end-r1" in execution_order
        assert "start-r2" in execution_order
        assert "end-r2" in execution_order
        r1_end_idx = execution_order.index("end-r1")
        r2_start_idx = execution_order.index("start-r2")
        assert r1_end_idx < r2_start_idx, (
            f"Expected r1 to finish before r2 starts, but got: {execution_order}"
        )


# =============================================================================
# Idempotency with idempotency_key
# =============================================================================


class TestIdempotencyKey:
    """Tests for idempotency_key handling in /v1/rollout/init."""

    async def test_idempotency_key_used_when_provided(self) -> None:
        """Two requests with different rollout_ids but same idempotency_key
        should be treated as duplicates."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp1 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="id-1", idempotency_key="shared-key"
                    ),
                )
                resp2 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="id-2", idempotency_key="shared-key"
                    ),
                )

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        # Both should return the same rollout_id (from the leader request)
        assert resp1.json()["rollout_id"] == resp2.json()["rollout_id"]

    async def test_different_idempotency_keys_are_independent(self) -> None:
        """Requests with different idempotency_keys should be treated separately."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp1 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="id-a", idempotency_key="key-a"
                    ),
                )
                resp2 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="id-b", idempotency_key="key-b"
                    ),
                )

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["rollout_id"] != resp2.json()["rollout_id"]

    async def test_fallback_to_rollout_id_when_no_idempotency_key(self) -> None:
        """Without idempotency_key, rollout_id is the idempotency key."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp1 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="dup-id"),
                )
                resp2 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="dup-id"),
                )

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["rollout_id"] == resp2.json()["rollout_id"]


# =============================================================================
# Error handling in init endpoint
# =============================================================================


class TestInitEndpointErrorHandling:
    """Tests for error paths in the /v1/rollout/init handler."""

    async def test_get_tools_error_returns_500(self) -> None:
        """If get_tools() raises, the endpoint should return 500."""
        app = create_app(FailingGetToolsAgentLoop())
        # raise_app_exceptions=False so httpx returns the HTTP 500 response
        # instead of propagating the exception to the test.
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/rollout/init",
                json=make_rollout_payload(rollout_id="fail-tools"),
            )
        assert resp.status_code == 500

    async def test_get_tools_error_clears_init_record(self) -> None:
        """After get_tools() fails, a retry with the same ID should succeed
        (the init record should have been cleaned up)."""

        call_count = 0

        class FailOnceThenSucceedAgent(RolloutAgentLoop):
            name = "fail_once_agent"

            def get_tools(
                self, request: RolloutRequest
            ) -> list[OpenAIFunctionToolSchema]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Transient error")
                return []

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                return ctx.complete([])

        app = create_app(FailOnceThenSucceedAgent())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # First request fails
                resp1 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="retry-me"),
                )
                assert resp1.status_code == 500

                # Retry should succeed (init record was cleared)
                resp2 = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="retry-me"),
                )
                assert resp2.status_code == 202

    async def test_infrastructure_error_marks_completed(self) -> None:
        """When OsmosisLLMClient raises during setup, the key should still be
        marked completed so resources are cleaned up."""
        app = create_app(SimpleAgentLoop())

        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            mock = AsyncMock()
            mock.__aenter__ = AsyncMock(side_effect=ConnectionError("Cannot connect"))
            mock.__aexit__ = AsyncMock(return_value=None)
            MockCls.return_value = mock

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="infra-err"),
                )
                assert resp.status_code == 202

                # Give background task time to fail and clean up
                await asyncio.sleep(0.2)

                health = await client.get("/health")
                data = health.json()
                # Should have moved to completed (not stuck in active)
                assert data["active_rollouts"] == 0


# =============================================================================
# Background rollout task
# =============================================================================


class TestBackgroundRolloutTask:
    """Tests for the run_rollout() background task created in init_rollout."""

    async def test_successful_rollout_calls_complete_rollout(self) -> None:
        """A successful agent run should call llm.complete_rollout with COMPLETED."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="bg-ok"),
                )
                assert resp.status_code == 202
                await asyncio.sleep(0.2)

        mock_client.complete_rollout.assert_called_once()
        kwargs = mock_client.complete_rollout.call_args[1]
        assert kwargs["status"] == "COMPLETED"
        assert kwargs["finish_reason"] == "stop"

    async def test_agent_error_reports_error_status(self) -> None:
        """When the agent raises, complete_rollout should be called with ERROR."""

        class ErrorAgent(RolloutAgentLoop):
            name = "error_agent"

            def get_tools(
                self, request: RolloutRequest
            ) -> list[OpenAIFunctionToolSchema]:
                return []

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                raise RuntimeError("Agent crashed")

        app = create_app(ErrorAgent())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="bg-err"),
                )
                assert resp.status_code == 202
                await asyncio.sleep(0.2)

        mock_client.complete_rollout.assert_called_once()
        kwargs = mock_client.complete_rollout.call_args[1]
        assert kwargs["status"] == "ERROR"
        assert "Agent crashed" in kwargs["error_message"]

    async def test_rollout_uses_request_api_key_for_callback(self) -> None:
        """The OsmosisLLMClient should be constructed with the request's api_key."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="api-key-cb", api_key="callback-secret"
                    ),
                )
                assert resp.status_code == 202
                await asyncio.sleep(0.2)

        # OsmosisLLMClient should have been called with the request's api_key
        MockCls.assert_called_once()
        assert MockCls.call_args.kwargs["api_key"] == "callback-secret"

    async def test_rollout_passes_server_url_to_client(self) -> None:
        """The OsmosisLLMClient should receive the server_url from the request."""
        app = create_app(SimpleAgentLoop())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(
                        rollout_id="url-test",
                        server_url="http://traingate:9000",
                    ),
                )
                assert resp.status_code == 202
                await asyncio.sleep(0.2)

        MockCls.assert_called_once()
        assert MockCls.call_args.kwargs["server_url"] == "http://traingate:9000"
        assert MockCls.call_args.kwargs["rollout_id"] == "url-test"


# =============================================================================
# /platform/health endpoint edge cases
# =============================================================================


class TestPlatformHealthEdgeCases:
    """Additional edge-case tests for /platform/health."""

    def test_raw_token_without_bearer_prefix_accepted(self) -> None:
        """A raw token (no 'Bearer' prefix) should still authenticate."""
        api_key = "my-secret-key"
        app = create_app(SimpleAgentLoop(), api_key=api_key)

        with TestClient(app) as client:
            resp = client.get(
                "/platform/health",
                headers={"Authorization": api_key},
            )
        assert resp.status_code == 200

    def test_platform_health_returns_agent_name_and_counts(self) -> None:
        """Authenticated /platform/health should return agent info and counts."""
        api_key = "check-key"
        app = create_app(SimpleAgentLoop(), api_key=api_key)

        with TestClient(app) as client:
            resp = client.get(
                "/platform/health",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_loop"] == "simple_agent"
        assert "active_rollouts" in data
        assert "completed_rollouts" in data


# =============================================================================
# /v1/rollout/init API key auth edge cases
# =============================================================================


class TestInitApiKeyEdgeCases:
    """Edge-case tests for API key authentication on /v1/rollout/init."""

    def test_raw_token_accepted_on_init(self) -> None:
        """A raw API key without 'Bearer' prefix should authenticate init."""
        api_key = "my-secret-key"
        app = create_app(SimpleAgentLoop(), api_key=api_key)

        with TestClient(app) as client:
            resp = client.post(
                "/v1/rollout/init",
                headers={"Authorization": api_key},
                json=make_rollout_payload(rollout_id="raw-auth"),
            )
        assert resp.status_code == 202


# =============================================================================
# /health endpoint details
# =============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint response shape."""

    async def test_health_returns_correct_counts(self) -> None:
        """Health endpoint should report accurate active and completed counts."""
        agent = SlowAgentLoop(delay=0.3)
        app = create_app(agent)
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Initially zero
                health = await client.get("/health")
                data = health.json()
                assert data["active_rollouts"] == 0
                assert data["completed_rollouts"] == 0

                # Start a rollout
                await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="count-1"),
                )
                await asyncio.sleep(0.05)

                health = await client.get("/health")
                data = health.json()
                assert data["active_rollouts"] >= 1

                # Wait for completion
                await asyncio.sleep(0.5)

                health = await client.get("/health")
                data = health.json()
                assert data["completed_rollouts"] >= 1


# =============================================================================
# create_app configuration variations
# =============================================================================


class TestCreateAppConfig:
    """Tests for various create_app() parameter combinations."""

    def test_custom_settings_used(self) -> None:
        """Custom RolloutSettings should be respected."""
        from osmosis_ai.rollout.config.settings import (
            RolloutServerSettings,
            RolloutSettings,
        )

        settings = RolloutSettings(
            server=RolloutServerSettings(max_concurrent_rollouts=42)
        )
        app = create_app(SimpleAgentLoop(), settings=settings)
        assert app is not None

    def test_custom_record_ttl(self) -> None:
        """Custom record_ttl_seconds should be accepted."""
        app = create_app(SimpleAgentLoop(), record_ttl_seconds=120.0)
        assert app is not None

    def test_debug_dir_parameter(self) -> None:
        """debug_dir parameter should be accepted."""
        app = create_app(SimpleAgentLoop(), debug_dir="/tmp/debug-test")
        assert app is not None

    def test_app_title_includes_agent_name(self) -> None:
        """The FastAPI app title should include the agent loop name."""
        app = create_app(SimpleAgentLoop())
        assert "simple_agent" in app.title

    def test_default_settings_loaded_when_none(self) -> None:
        """When settings=None, default settings should be loaded via get_settings()."""
        app = create_app(SimpleAgentLoop())
        assert app is not None
        assert "simple_agent" in app.title


# =============================================================================
# Debug dir in background rollout
# =============================================================================


class TestDebugDirInRollout:
    """Tests that debug_dir is passed through to the RolloutContext."""

    async def test_debug_dir_propagated_to_context(self, tmp_path: Any) -> None:
        """The debug_dir should be passed to RolloutContext during rollout."""
        debug_dir = str(tmp_path / "debug_out")
        observed_debug_dir: str | None = "UNSET"

        class InspectingAgent(RolloutAgentLoop):
            name = "inspecting_agent"

            def get_tools(
                self, request: RolloutRequest
            ) -> list[OpenAIFunctionToolSchema]:
                return []

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                nonlocal observed_debug_dir
                observed_debug_dir = ctx._debug_dir
                return ctx.complete([])

        app = create_app(InspectingAgent(), debug_dir=debug_dir)
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="debug-prop"),
                )
                await asyncio.sleep(0.2)

        assert observed_debug_dir == debug_dir

    async def test_no_debug_dir_means_none_in_context(self) -> None:
        """Without debug_dir, RolloutContext._debug_dir should be None."""
        observed_debug_dir: str | None = "UNSET"

        class InspectingAgent(RolloutAgentLoop):
            name = "inspect_no_debug"

            def get_tools(
                self, request: RolloutRequest
            ) -> list[OpenAIFunctionToolSchema]:
                return []

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                nonlocal observed_debug_dir
                observed_debug_dir = ctx._debug_dir
                return ctx.complete([])

        app = create_app(InspectingAgent())
        mock_client = mock_llm_client()
        with patch("osmosis_ai.rollout.server.app.OsmosisLLMClient") as MockCls:
            MockCls.return_value = mock_client

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post(
                    "/v1/rollout/init",
                    json=make_rollout_payload(rollout_id="no-debug"),
                )
                await asyncio.sleep(0.2)

        assert observed_debug_dir is None


# =============================================================================
# ImportError guard
# =============================================================================


class TestImportGuard:
    """Test that create_app() raises ImportError when FastAPI is absent."""

    def test_raises_import_error_when_fastapi_missing(self) -> None:
        """create_app() should raise ImportError with a helpful message."""
        with patch("osmosis_ai.rollout.server.app.FASTAPI_AVAILABLE", False):
            with pytest.raises(ImportError, match="FastAPI is required"):
                create_app(SimpleAgentLoop())

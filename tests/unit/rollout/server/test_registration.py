"""Tests for osmosis_ai.rollout.server.registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.rollout.network import PublicIPDetectionError
from osmosis_ai.rollout.server.registration import (
    RegistrationResult,
    get_public_ip,
    get_report_host,
    print_registration_result,
    register_with_platform,
)

# =============================================================================
# Helper: Create mock WorkspaceCredentials
# =============================================================================


def _make_mock_credentials() -> MagicMock:
    """Create a mock WorkspaceCredentials with a fake access_token."""
    creds = MagicMock()
    creds.access_token = "fake-token-abc123"
    return creds


# =============================================================================
# RegistrationResult Tests
# =============================================================================


class TestRegistrationResult:
    """Tests for the RegistrationResult dataclass."""

    def test_default_values(self) -> None:
        """Verify default field values on a minimal RegistrationResult."""
        result = RegistrationResult(success=True)
        assert result.success is True
        assert result.server_id is None
        assert result.status == "unknown"
        assert result.error is None
        assert result.server_info is None

    def test_is_healthy_when_status_is_healthy(self) -> None:
        """Verify is_healthy returns True only when status is 'healthy'."""
        result = RegistrationResult(success=True, status="healthy")
        assert result.is_healthy is True

    def test_is_healthy_returns_false_for_other_statuses(self) -> None:
        """Verify is_healthy returns False for non-healthy statuses."""
        for status in ("unknown", "error", "unhealthy", "pending", ""):
            result = RegistrationResult(success=True, status=status)
            assert result.is_healthy is False, f"Expected False for status={status!r}"

    def test_full_construction(self) -> None:
        """Verify all fields can be set at construction time."""
        info = {"active_rollouts": 3, "agent_loop": "my_agent"}
        result = RegistrationResult(
            success=True,
            server_id="srv_123",
            status="healthy",
            error=None,
            server_info=info,
        )
        assert result.server_id == "srv_123"
        assert result.server_info == info


# =============================================================================
# get_public_ip Tests
# =============================================================================


class TestGetPublicIP:
    """Tests for the get_public_ip function."""

    @patch("osmosis_ai.rollout.server.registration.detect_public_ip")
    def test_returns_detected_ip(self, mock_detect: MagicMock) -> None:
        """Verify get_public_ip delegates to detect_public_ip and returns its result."""
        mock_detect.return_value = "54.200.100.50"
        assert get_public_ip() == "54.200.100.50"
        mock_detect.assert_called_once()

    @patch("osmosis_ai.rollout.server.registration.detect_public_ip")
    def test_propagates_detection_error(self, mock_detect: MagicMock) -> None:
        """Verify PublicIPDetectionError propagates from get_public_ip."""
        mock_detect.side_effect = PublicIPDetectionError("all methods failed")
        with pytest.raises(PublicIPDetectionError, match="all methods failed"):
            get_public_ip()


# =============================================================================
# get_report_host Tests
# =============================================================================


class TestGetReportHost:
    """Tests for the get_report_host function."""

    @patch("osmosis_ai.rollout.server.registration.get_public_ip")
    def test_returns_public_ip_for_all_interfaces(self, mock_ip: MagicMock) -> None:
        """Verify 0.0.0.0 triggers public IP detection."""
        mock_ip.return_value = "203.0.113.10"
        assert get_report_host("0.0.0.0") == "203.0.113.10"
        mock_ip.assert_called_once()

    def test_returns_host_as_is_for_specific_address(self) -> None:
        """Verify a specific IP/hostname is returned unchanged."""
        assert get_report_host("10.0.0.5") == "10.0.0.5"
        assert get_report_host("my-server.example.com") == "my-server.example.com"
        assert get_report_host("127.0.0.1") == "127.0.0.1"

    @patch("osmosis_ai.rollout.server.registration.get_public_ip")
    def test_propagates_error_for_all_interfaces(self, mock_ip: MagicMock) -> None:
        """Verify PublicIPDetectionError propagates when host is 0.0.0.0."""
        mock_ip.side_effect = PublicIPDetectionError("detection failed")
        with pytest.raises(PublicIPDetectionError, match="detection failed"):
            get_report_host("0.0.0.0")


# =============================================================================
# register_with_platform Tests
# =============================================================================


class TestRegisterWithPlatform:
    """Tests for the register_with_platform function."""

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_healthy_registration(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify a successful healthy registration returns expected result."""
        mock_host.return_value = "54.200.100.1"
        mock_request.return_value = {
            "id": "srv_abc",
            "status": "healthy",
            "health_check_result": {
                "server_info": {"active_rollouts": 0, "agent_loop": "my_agent"},
            },
        }

        result = register_with_platform(
            host="0.0.0.0",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is True
        assert result.is_healthy is True
        assert result.server_id == "srv_abc"
        assert result.server_info == {"active_rollouts": 0, "agent_loop": "my_agent"}
        assert result.error is None

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_registration_succeeded_but_health_check_failed(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify registration success with health check failure returns appropriate result."""
        mock_host.return_value = "54.200.100.1"
        mock_request.return_value = {
            "id": "srv_xyz",
            "status": "unhealthy",
            "health_check_result": {
                "error": "Connection refused",
            },
        }

        result = register_with_platform(
            host="0.0.0.0",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is True
        assert result.is_healthy is False
        assert result.server_id == "srv_xyz"
        assert result.status == "unhealthy"
        assert result.error == "Connection refused"

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_registration_sends_correct_data_without_api_key(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify the correct registration payload is sent when no api_key is given."""
        mock_host.return_value = "10.0.0.5"
        mock_request.return_value = {
            "id": "srv_1",
            "status": "healthy",
            "health_check_result": {},
        }

        creds = _make_mock_credentials()
        register_with_platform(
            host="10.0.0.5",
            port=9090,
            agent_loop_name="test_agent",
            credentials=creds,
        )

        mock_request.assert_called_once_with(
            "/api/cli/rollout-server/register",
            method="POST",
            data={
                "host": "10.0.0.5",
                "port": 9090,
                "agent_loop_name": "test_agent",
            },
            timeout=15.0,
            credentials=creds,
        )

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_registration_sends_api_key_when_provided(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify api_key is included in the registration payload when provided."""
        mock_host.return_value = "10.0.0.5"
        mock_request.return_value = {
            "id": "srv_1",
            "status": "healthy",
            "health_check_result": {},
        }

        creds = _make_mock_credentials()
        register_with_platform(
            host="10.0.0.5",
            port=9090,
            agent_loop_name="test_agent",
            credentials=creds,
            api_key="secret-key-456",
        )

        call_data = mock_request.call_args[1]["data"]
        assert call_data["api_key"] == "secret-key-456"

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_ip_detection_failure_returns_error_result(
        self, mock_host: MagicMock
    ) -> None:
        """Verify IP detection failure returns a failed RegistrationResult."""
        mock_host.side_effect = PublicIPDetectionError("no IP")

        result = register_with_platform(
            host="0.0.0.0",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is False
        assert result.status == "error"
        assert "public IP" in result.error

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_authentication_expired_returns_error_result(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify AuthenticationExpiredError is caught and returned as error result."""
        from osmosis_ai.auth.platform_client import AuthenticationExpiredError

        mock_host.return_value = "10.0.0.5"
        mock_request.side_effect = AuthenticationExpiredError("session expired")

        result = register_with_platform(
            host="10.0.0.5",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is False
        assert result.status == "error"
        assert "session expired" in result.error

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_platform_api_error_returns_error_result(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify PlatformAPIError is caught and returned as error result."""
        from osmosis_ai.auth.platform_client import PlatformAPIError

        mock_host.return_value = "10.0.0.5"
        mock_request.side_effect = PlatformAPIError("HTTP 500", status_code=500)

        result = register_with_platform(
            host="10.0.0.5",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is False
        assert result.status == "error"
        assert "HTTP 500" in result.error

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_unexpected_exception_returns_error_result(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify unexpected exceptions are caught and returned as error result."""
        mock_host.return_value = "10.0.0.5"
        mock_request.side_effect = RuntimeError("unexpected network issue")

        result = register_with_platform(
            host="10.0.0.5",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is False
        assert result.status == "error"
        assert "unexpected network issue" in result.error

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_missing_status_defaults_to_unknown(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify missing status in platform response defaults to 'unknown'."""
        mock_host.return_value = "10.0.0.5"
        mock_request.return_value = {
            "id": "srv_1",
            # No "status" key
            "health_check_result": {},
        }

        result = register_with_platform(
            host="10.0.0.5",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        # Registration succeeded, but status is unknown -> non-healthy branch
        assert result.success is True
        assert result.status == "unknown"
        assert result.is_healthy is False

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    @patch("osmosis_ai.auth.platform_client.platform_request")
    def test_health_check_failed_default_error_message(
        self, mock_request: MagicMock, mock_host: MagicMock
    ) -> None:
        """Verify default error message when health_check_result has no 'error' key."""
        mock_host.return_value = "10.0.0.5"
        mock_request.return_value = {
            "id": "srv_1",
            "status": "unhealthy",
            "health_check_result": {},  # No "error" key
        }

        result = register_with_platform(
            host="10.0.0.5",
            port=8080,
            agent_loop_name="my_agent",
            credentials=_make_mock_credentials(),
        )

        assert result.success is True
        assert result.error == "Health check failed"


# =============================================================================
# print_registration_result Tests
# =============================================================================


class TestPrintRegistrationResult:
    """Tests for the print_registration_result function."""

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_healthy_output(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify output for a healthy registration result."""
        mock_host.return_value = "54.200.100.1"
        result = RegistrationResult(
            success=True,
            server_id="srv_1",
            status="healthy",
            server_info={"active_rollouts": 2},
        )

        print_registration_result(
            result, host="0.0.0.0", port=8080, agent_loop_name="my_agent"
        )

        output = capsys.readouterr().out
        assert "[OK] Registered with Osmosis Platform" in output
        assert "my_agent" in output
        assert "54.200.100.1:8080" in output
        assert "Active rollouts: 2" in output

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_healthy_output_without_server_info(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify output for healthy result when server_info is None."""
        mock_host.return_value = "54.200.100.1"
        result = RegistrationResult(
            success=True,
            server_id="srv_1",
            status="healthy",
            server_info=None,
        )

        print_registration_result(
            result, host="0.0.0.0", port=8080, agent_loop_name="my_agent"
        )

        output = capsys.readouterr().out
        assert "[OK] Registered with Osmosis Platform" in output
        assert "Active rollouts" not in output

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_unhealthy_output(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify output for registration success but health check failure."""
        mock_host.return_value = "10.0.0.5"
        result = RegistrationResult(
            success=True,
            server_id="srv_2",
            status="unhealthy",
            error="Connection refused",
        )

        print_registration_result(
            result, host="10.0.0.5", port=9090, agent_loop_name="test_agent"
        )

        output = capsys.readouterr().out
        assert "[WARNING] Registered but health check failed" in output
        assert "test_agent" in output
        assert "10.0.0.5:9090" in output
        assert "Connection refused" in output
        assert "The server will continue running." in output
        assert "Tip: Use a VM with public IP" in output

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_failed_registration_output(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify output for a completely failed registration."""
        mock_host.return_value = "10.0.0.5"
        result = RegistrationResult(
            success=False,
            status="error",
            error="Authentication expired",
        )

        print_registration_result(
            result, host="10.0.0.5", port=8080, agent_loop_name="agent"
        )

        output = capsys.readouterr().out
        assert "[WARNING] Failed to register with Platform" in output
        assert "Authentication expired" in output
        assert "The server will continue running without registration." in output

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_ip_detection_failure_falls_back_to_original_host(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify print_registration_result falls back to original host on IP detection error."""
        mock_host.side_effect = PublicIPDetectionError("cannot detect IP")
        result = RegistrationResult(
            success=True,
            server_id="srv_1",
            status="healthy",
        )

        print_registration_result(
            result, host="0.0.0.0", port=8080, agent_loop_name="agent"
        )

        output = capsys.readouterr().out
        # Should fall back to original host "0.0.0.0" instead of crashing
        assert "0.0.0.0:8080" in output
        assert "[OK] Registered with Osmosis Platform" in output

    @patch("osmosis_ai.rollout.server.registration.get_report_host")
    def test_healthy_output_with_zero_active_rollouts(
        self, mock_host: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify active rollouts line shows 0 correctly."""
        mock_host.return_value = "10.0.0.5"
        result = RegistrationResult(
            success=True,
            status="healthy",
            server_info={"active_rollouts": 0},
        )

        print_registration_result(
            result, host="10.0.0.5", port=8080, agent_loop_name="agent"
        )

        output = capsys.readouterr().out
        assert "Active rollouts: 0" in output

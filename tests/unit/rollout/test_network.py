"""Tests for osmosis_ai.rollout.network module.

Covers IP validation helpers (pure logic), cloud metadata detection (mocked HTTP),
external service fallback, and the main detect_public_ip orchestrator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from osmosis_ai.rollout.network import (
    PublicIPDetectionError,
    _get_aws_public_ip,
    _get_azure_public_ip,
    _get_gcp_public_ip,
    detect_from_cloud_metadata,
    detect_from_external_services,
    detect_public_ip,
    is_private_ip,
    is_valid_hostname_or_ip,
    validate_ipv4,
)

# =============================================================================
# validate_ipv4 Tests
# =============================================================================


class TestValidateIPv4:
    """Tests for the validate_ipv4 helper."""

    def test_standard_public_ip(self) -> None:
        """Valid public IPv4 addresses should be accepted."""
        assert validate_ipv4("8.8.8.8") is True
        assert validate_ipv4("54.123.45.67") is True
        assert validate_ipv4("1.2.3.4") is True

    def test_private_ip_ranges(self) -> None:
        """Private IPv4 addresses are still valid IPv4."""
        assert validate_ipv4("192.168.1.1") is True
        assert validate_ipv4("10.0.0.1") is True
        assert validate_ipv4("172.16.0.1") is True

    def test_boundary_values(self) -> None:
        """Boundary addresses (0.0.0.0, 255.255.255.255) are valid IPv4."""
        assert validate_ipv4("0.0.0.0") is True
        assert validate_ipv4("255.255.255.255") is True

    def test_loopback(self) -> None:
        """Loopback address is valid IPv4."""
        assert validate_ipv4("127.0.0.1") is True

    def test_octet_out_of_range(self) -> None:
        """Octets > 255 should be rejected."""
        assert validate_ipv4("256.1.1.1") is False
        assert validate_ipv4("1.256.1.1") is False
        assert validate_ipv4("1.1.1.256") is False

    def test_not_an_ip_at_all(self) -> None:
        """Non-IP strings should be rejected."""
        assert validate_ipv4("not-an-ip") is False
        assert validate_ipv4("hello world") is False
        assert validate_ipv4("") is False

    def test_ipv6_is_rejected(self) -> None:
        """IPv6 addresses should be rejected (this function is IPv4-only)."""
        assert validate_ipv4("::1") is False
        assert validate_ipv4("2001:db8::1") is False
        assert validate_ipv4("fe80::1") is False

    def test_too_few_or_too_many_octets(self) -> None:
        """Addresses with wrong number of octets should be rejected."""
        assert validate_ipv4("1.2.3") is False
        assert validate_ipv4("1.2.3.4.5") is False

    def test_leading_zeros(self) -> None:
        """Leading zeros are rejected by Python's ipaddress module."""
        assert validate_ipv4("01.02.03.04") is False

    def test_whitespace_not_stripped(self) -> None:
        """Addresses with surrounding whitespace should be rejected."""
        assert validate_ipv4(" 1.2.3.4") is False
        assert validate_ipv4("1.2.3.4 ") is False


# =============================================================================
# is_private_ip Tests
# =============================================================================


class TestIsPrivateIP:
    """Tests for the is_private_ip helper."""

    def test_rfc1918_class_a(self) -> None:
        """10.0.0.0/8 range should be private."""
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("10.255.255.255") is True

    def test_rfc1918_class_b(self) -> None:
        """172.16.0.0/12 range should be private."""
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("172.31.255.255") is True

    def test_rfc1918_class_c(self) -> None:
        """192.168.0.0/16 range should be private."""
        assert is_private_ip("192.168.0.1") is True
        assert is_private_ip("192.168.255.255") is True

    def test_loopback_is_private(self) -> None:
        """Loopback addresses are considered private by Python ipaddress."""
        assert is_private_ip("127.0.0.1") is True

    def test_public_ips(self) -> None:
        """Public IPs should not be private."""
        assert is_private_ip("54.123.45.67") is False
        assert is_private_ip("8.8.8.8") is False
        assert is_private_ip("1.1.1.1") is False

    def test_invalid_input_returns_false(self) -> None:
        """Invalid IP strings should return False, not raise."""
        assert is_private_ip("not-an-ip") is False
        assert is_private_ip("") is False
        assert is_private_ip("999.999.999.999") is False


# =============================================================================
# is_valid_hostname_or_ip Tests
# =============================================================================


class TestIsValidHostnameOrIP:
    """Tests for the is_valid_hostname_or_ip helper."""

    def test_valid_ipv4_addresses(self) -> None:
        """IPv4 addresses should be accepted."""
        assert is_valid_hostname_or_ip("192.168.1.1") is True
        assert is_valid_hostname_or_ip("8.8.8.8") is True
        assert is_valid_hostname_or_ip("0.0.0.0") is True

    def test_valid_hostnames(self) -> None:
        """Standard hostnames should be accepted."""
        assert is_valid_hostname_or_ip("example.com") is True
        assert is_valid_hostname_or_ip("sub.example.com") is True
        assert is_valid_hostname_or_ip("localhost") is True

    def test_single_character_hostname(self) -> None:
        """Single character hostnames are valid per the regex pattern."""
        assert is_valid_hostname_or_ip("a") is True

    def test_hostname_with_hyphens(self) -> None:
        """Hostnames with internal hyphens should be accepted."""
        assert is_valid_hostname_or_ip("my-host") is True
        assert is_valid_hostname_or_ip("my-long-hostname.example.com") is True

    def test_hostname_with_underscores(self) -> None:
        """Hostnames with underscores should be accepted (practical tolerance)."""
        assert is_valid_hostname_or_ip("my_host") is True
        assert is_valid_hostname_or_ip("my_host.example.com") is True

    def test_host_with_port(self) -> None:
        """host:port format should be accepted."""
        assert is_valid_hostname_or_ip("example.com:8080") is True
        assert is_valid_hostname_or_ip("192.168.1.1:8080") is True
        assert is_valid_hostname_or_ip("localhost:3000") is True

    def test_empty_string_rejected(self) -> None:
        """Empty string should be rejected."""
        assert is_valid_hostname_or_ip("") is False

    def test_string_with_spaces_rejected(self) -> None:
        """Strings containing spaces should be rejected."""
        assert is_valid_hostname_or_ip("has spaces") is False
        assert is_valid_hostname_or_ip("has space.com") is False

    def test_too_long_rejected(self) -> None:
        """Values exceeding max length should be rejected."""
        # Total length > 260 chars
        long_value = "a" * 261
        assert is_valid_hostname_or_ip(long_value) is False

    def test_hostname_label_too_long(self) -> None:
        """Hostname total > 253 chars (without port) should be rejected."""
        # Host part > 253 chars
        long_host = "a" * 254
        assert is_valid_hostname_or_ip(long_host) is False

    def test_colon_with_non_numeric_port_treated_as_hostname(self) -> None:
        """A colon followed by non-digits is not treated as host:port splitting."""
        # "host:abc" - the port is not numeric, so the full string is checked as hostname
        # This should fail because ':' is not in the allowed hostname chars
        assert is_valid_hostname_or_ip("host:abc") is False

    def test_numeric_hostname(self) -> None:
        """Purely numeric strings that are not valid IPs should still be checked."""
        # "1" is a valid single-char hostname
        assert is_valid_hostname_or_ip("1") is True


# =============================================================================
# AWS Metadata Detection Tests
# =============================================================================


class TestGetAWSPublicIP:
    """Tests for _get_aws_public_ip with mocked HTTP."""

    @patch("osmosis_ai.rollout.network.requests")
    def test_successful_detection(self, mock_requests: MagicMock) -> None:
        """AWS IMDSv2 two-step flow returns a valid public IP."""
        mock_token_resp = MagicMock()
        mock_token_resp.text = "test-token-abc"
        mock_token_resp.raise_for_status = MagicMock()

        mock_ip_resp = MagicMock()
        mock_ip_resp.status_code = 200
        mock_ip_resp.text = "54.123.45.67"
        mock_ip_resp.raise_for_status = MagicMock()

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.return_value = mock_ip_resp

        result = _get_aws_public_ip()

        assert result == "54.123.45.67"
        # Verify token request was made with PUT
        mock_requests.put.assert_called_once()
        put_args = mock_requests.put.call_args
        assert "169.254.169.254" in put_args[0][0]
        assert "X-aws-ec2-metadata-token-ttl-seconds" in put_args[1]["headers"]

    @patch("osmosis_ai.rollout.network.requests")
    def test_no_public_ip_returns_none(self, mock_requests: MagicMock) -> None:
        """AWS returns 404 when instance has no public IP (private subnet)."""
        mock_token_resp = MagicMock()
        mock_token_resp.text = "test-token"
        mock_token_resp.raise_for_status = MagicMock()

        mock_ip_resp = MagicMock()
        mock_ip_resp.status_code = 404

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.return_value = mock_ip_resp

        result = _get_aws_public_ip()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_token_request_fails(self, mock_requests: MagicMock) -> None:
        """When token request fails, returns None gracefully."""
        mock_requests.put.side_effect = requests.ConnectionError("timeout")

        result = _get_aws_public_ip()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_invalid_ip_returned(self, mock_requests: MagicMock) -> None:
        """When metadata returns non-IP text, returns None."""
        mock_token_resp = MagicMock()
        mock_token_resp.text = "token"
        mock_token_resp.raise_for_status = MagicMock()

        mock_ip_resp = MagicMock()
        mock_ip_resp.status_code = 200
        mock_ip_resp.text = "not-a-valid-ip"
        mock_ip_resp.raise_for_status = MagicMock()

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.return_value = mock_ip_resp

        result = _get_aws_public_ip()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_empty_ip_response(self, mock_requests: MagicMock) -> None:
        """When metadata returns empty text, returns None."""
        mock_token_resp = MagicMock()
        mock_token_resp.text = "token"
        mock_token_resp.raise_for_status = MagicMock()

        mock_ip_resp = MagicMock()
        mock_ip_resp.status_code = 200
        mock_ip_resp.text = "  "
        mock_ip_resp.raise_for_status = MagicMock()

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.return_value = mock_ip_resp

        result = _get_aws_public_ip()
        assert result is None


# =============================================================================
# GCP Metadata Detection Tests
# =============================================================================


class TestGetGCPPublicIP:
    """Tests for _get_gcp_public_ip with mocked HTTP."""

    @patch("osmosis_ai.rollout.network.requests")
    def test_successful_detection(self, mock_requests: MagicMock) -> None:
        """GCP metadata returns a valid public IP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "35.200.100.50"
        mock_resp.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp

        result = _get_gcp_public_ip()

        assert result == "35.200.100.50"
        get_args = mock_requests.get.call_args
        assert "Metadata-Flavor" in get_args[1]["headers"]
        assert get_args[1]["headers"]["Metadata-Flavor"] == "Google"

    @patch("osmosis_ai.rollout.network.requests")
    def test_no_external_ip_returns_none(self, mock_requests: MagicMock) -> None:
        """GCP returns 404 when instance has no external IP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_requests.get.return_value = mock_resp

        result = _get_gcp_public_ip()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_connection_error_returns_none(self, mock_requests: MagicMock) -> None:
        """When GCP metadata service is unreachable, returns None."""
        mock_requests.get.side_effect = requests.ConnectionError("timeout")

        result = _get_gcp_public_ip()
        assert result is None


# =============================================================================
# Azure Metadata Detection Tests
# =============================================================================


class TestGetAzurePublicIP:
    """Tests for _get_azure_public_ip with mocked HTTP."""

    @patch("osmosis_ai.rollout.network.requests")
    def test_successful_detection(self, mock_requests: MagicMock) -> None:
        """Azure IMDS returns a valid public IP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "20.50.100.200"
        mock_resp.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp

        result = _get_azure_public_ip()

        assert result == "20.50.100.200"
        get_args = mock_requests.get.call_args
        assert get_args[1]["headers"]["Metadata"] == "true"
        assert "api-version" in get_args[1]["params"]

    @patch("osmosis_ai.rollout.network.requests")
    def test_no_public_ip_returns_none(self, mock_requests: MagicMock) -> None:
        """Azure returns 404 when instance has no public IP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_requests.get.return_value = mock_resp

        result = _get_azure_public_ip()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_connection_error_returns_none(self, mock_requests: MagicMock) -> None:
        """When Azure IMDS is unreachable, returns None."""
        mock_requests.get.side_effect = requests.Timeout("timeout")

        result = _get_azure_public_ip()
        assert result is None


# =============================================================================
# detect_from_cloud_metadata Tests
# =============================================================================


class TestDetectFromCloudMetadata:
    """Tests for the parallel cloud metadata detection orchestrator."""

    @patch("osmosis_ai.rollout.network._get_azure_public_ip", return_value=None)
    @patch("osmosis_ai.rollout.network._get_gcp_public_ip", return_value=None)
    @patch(
        "osmosis_ai.rollout.network._get_aws_public_ip",
        return_value="54.123.45.67",
    )
    def test_returns_first_successful_result(
        self, mock_aws: MagicMock, mock_gcp: MagicMock, mock_azure: MagicMock
    ) -> None:
        """When AWS succeeds, returns its IP even if others fail."""
        result = detect_from_cloud_metadata()
        assert result == "54.123.45.67"

    @patch("osmosis_ai.rollout.network._get_azure_public_ip", return_value=None)
    @patch(
        "osmosis_ai.rollout.network._get_gcp_public_ip",
        return_value="35.200.100.50",
    )
    @patch("osmosis_ai.rollout.network._get_aws_public_ip", return_value=None)
    def test_returns_gcp_when_aws_fails(
        self, mock_aws: MagicMock, mock_gcp: MagicMock, mock_azure: MagicMock
    ) -> None:
        """When AWS returns None but GCP succeeds, returns GCP's IP."""
        result = detect_from_cloud_metadata()
        assert result == "35.200.100.50"

    @patch("osmosis_ai.rollout.network._get_azure_public_ip", return_value=None)
    @patch("osmosis_ai.rollout.network._get_gcp_public_ip", return_value=None)
    @patch("osmosis_ai.rollout.network._get_aws_public_ip", return_value=None)
    def test_returns_none_when_all_fail(
        self, mock_aws: MagicMock, mock_gcp: MagicMock, mock_azure: MagicMock
    ) -> None:
        """When all cloud providers return None, returns None."""
        result = detect_from_cloud_metadata()
        assert result is None

    @patch(
        "osmosis_ai.rollout.network._get_azure_public_ip",
        side_effect=Exception("crash"),
    )
    @patch(
        "osmosis_ai.rollout.network._get_gcp_public_ip",
        side_effect=Exception("crash"),
    )
    @patch(
        "osmosis_ai.rollout.network._get_aws_public_ip",
        side_effect=Exception("crash"),
    )
    def test_handles_exceptions_from_all_providers(
        self, mock_aws: MagicMock, mock_gcp: MagicMock, mock_azure: MagicMock
    ) -> None:
        """When all providers raise exceptions, returns None without crashing."""
        result = detect_from_cloud_metadata()
        assert result is None


# =============================================================================
# detect_from_external_services Tests
# =============================================================================


class TestDetectFromExternalServices:
    """Tests for external IP service fallback detection."""

    @patch("osmosis_ai.rollout.network.requests")
    def test_returns_first_valid_ip(self, mock_requests: MagicMock) -> None:
        """When an external service returns a valid IP, it is returned."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "203.0.113.42\n"
        mock_resp.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp

        result = detect_from_external_services()
        assert result == "203.0.113.42"

    @patch("osmosis_ai.rollout.network.requests")
    def test_returns_none_when_all_services_fail(
        self, mock_requests: MagicMock
    ) -> None:
        """When all external services raise errors, returns None."""
        mock_requests.get.side_effect = requests.ConnectionError("unreachable")

        result = detect_from_external_services()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_returns_none_when_services_return_invalid_ip(
        self, mock_requests: MagicMock
    ) -> None:
        """When services return non-IP text, returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<!DOCTYPE html>error page"
        mock_resp.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp

        result = detect_from_external_services()
        assert result is None

    @patch("osmosis_ai.rollout.network.requests")
    def test_returns_none_when_services_return_empty(
        self, mock_requests: MagicMock
    ) -> None:
        """When services return empty text, returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp

        result = detect_from_external_services()
        assert result is None


# =============================================================================
# detect_public_ip (Main Orchestrator) Tests
# =============================================================================


class TestDetectPublicIP:
    """Tests for the main detect_public_ip orchestration function."""

    @patch("osmosis_ai.rollout.network.detect_from_external_services")
    @patch("osmosis_ai.rollout.network.detect_from_cloud_metadata")
    def test_prefers_cloud_metadata_over_external(
        self, mock_cloud: MagicMock, mock_external: MagicMock
    ) -> None:
        """Cloud metadata result is returned without calling external services."""
        mock_cloud.return_value = "54.123.45.67"

        result = detect_public_ip()

        assert result == "54.123.45.67"
        mock_cloud.assert_called_once()
        mock_external.assert_not_called()

    @patch("osmosis_ai.rollout.network.detect_from_external_services")
    @patch("osmosis_ai.rollout.network.detect_from_cloud_metadata")
    def test_falls_back_to_external_when_cloud_fails(
        self, mock_cloud: MagicMock, mock_external: MagicMock
    ) -> None:
        """When cloud metadata returns None, falls back to external services."""
        mock_cloud.return_value = None
        mock_external.return_value = "203.0.113.42"

        result = detect_public_ip()

        assert result == "203.0.113.42"
        mock_cloud.assert_called_once()
        mock_external.assert_called_once()

    @patch("osmosis_ai.rollout.network.detect_from_external_services")
    @patch("osmosis_ai.rollout.network.detect_from_cloud_metadata")
    def test_raises_error_when_all_methods_fail(
        self, mock_cloud: MagicMock, mock_external: MagicMock
    ) -> None:
        """When both cloud and external fail, raises PublicIPDetectionError."""
        mock_cloud.return_value = None
        mock_external.return_value = None

        with pytest.raises(PublicIPDetectionError, match="Failed to detect public IP"):
            detect_public_ip()

    @patch("osmosis_ai.rollout.network.detect_from_external_services")
    @patch("osmosis_ai.rollout.network.detect_from_cloud_metadata")
    def test_error_message_is_informative(
        self, mock_cloud: MagicMock, mock_external: MagicMock
    ) -> None:
        """The error message should mention all detection methods that were tried."""
        mock_cloud.return_value = None
        mock_external.return_value = None

        with pytest.raises(PublicIPDetectionError) as exc_info:
            detect_public_ip()

        error_message = str(exc_info.value)
        assert "Cloud metadata" in error_message
        assert "External IP services" in error_message


# =============================================================================
# PublicIPDetectionError Tests
# =============================================================================


class TestPublicIPDetectionError:
    """Tests for the custom exception class."""

    def test_is_exception_subclass(self) -> None:
        """PublicIPDetectionError should be a proper Exception subclass."""
        assert issubclass(PublicIPDetectionError, Exception)

    def test_can_be_instantiated_with_message(self) -> None:
        """Exception can carry a custom message."""
        err = PublicIPDetectionError("custom message")
        assert str(err) == "custom message"

    def test_can_be_caught_as_exception(self) -> None:
        """PublicIPDetectionError can be caught via pytest.raises."""
        with pytest.raises(PublicIPDetectionError):
            raise PublicIPDetectionError("test")

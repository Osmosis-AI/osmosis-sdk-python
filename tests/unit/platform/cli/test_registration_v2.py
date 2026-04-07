"""Tests for platform CLI registration (v2 serve migration)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osmosis_ai.platform.cli import registration as reg


@pytest.fixture
def mock_public_ip() -> str:
    return "203.0.113.1"


class TestGetReportHost:
    @pytest.mark.parametrize(
        "bind_host",
        ["0.0.0.0", "", "::", "0:0:0:0:0:0:0:0"],
    )
    def test_bind_all_sentinels_use_public_ip(
        self, bind_host: str, mock_public_ip: str
    ) -> None:
        with patch.object(reg, "detect_public_ip", return_value=mock_public_ip):
            assert reg.get_report_host(bind_host) == mock_public_ip

    @pytest.mark.parametrize(
        "specific_host",
        ["192.168.1.100", "127.0.0.1", "my-host.example.com"],
    )
    def test_specific_host_passthrough(
        self, specific_host: str, mock_public_ip: str
    ) -> None:
        with patch.object(reg, "detect_public_ip", return_value=mock_public_ip):
            assert reg.get_report_host(specific_host) == specific_host


class TestProbeHost:
    @pytest.mark.parametrize(
        "bind_host",
        ["0.0.0.0", "", "::", "0:0:0:0:0:0:0:0"],
    )
    def test_bind_all_sentinels_use_loopback(self, bind_host: str) -> None:
        assert reg.probe_host(bind_host) == "127.0.0.1"

    @pytest.mark.parametrize(
        "specific_host",
        ["192.168.1.100", "127.0.0.1", "my-host.example.com"],
    )
    def test_specific_host_passthrough(self, specific_host: str) -> None:
        assert reg.probe_host(specific_host) == specific_host

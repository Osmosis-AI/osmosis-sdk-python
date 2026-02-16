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

"""Tests for osmosis_ai.rollout.config.settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout.config.settings import (
    RolloutClientSettings,
    RolloutServerSettings,
    RolloutSettings,
    configure,
    get_settings,
    reset_settings,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_settings():
    """Reset the global settings singleton before and after each test."""
    reset_settings()
    yield
    reset_settings()


# =============================================================================
# Default Values Tests
# =============================================================================


class TestDefaultValues:
    """Verify sensible default values when no env vars are set."""

    def test_client_timeout_seconds_default(self) -> None:
        """Default client timeout is 300 seconds."""
        settings = RolloutClientSettings()
        assert settings.timeout_seconds == 300.0

    def test_client_max_retries_default(self) -> None:
        """Default max retries is 3."""
        settings = RolloutClientSettings()
        assert settings.max_retries == 3

    def test_client_complete_rollout_retries_default(self) -> None:
        """Default complete_rollout_retries is 2."""
        settings = RolloutClientSettings()
        assert settings.complete_rollout_retries == 2

    def test_client_retry_base_delay_default(self) -> None:
        """Default retry base delay is 1.0."""
        settings = RolloutClientSettings()
        assert settings.retry_base_delay == 1.0

    def test_client_retry_max_delay_default(self) -> None:
        """Default retry max delay is 30.0."""
        settings = RolloutClientSettings()
        assert settings.retry_max_delay == 30.0

    def test_client_max_connections_default(self) -> None:
        """Default max connections is 100."""
        settings = RolloutClientSettings()
        assert settings.max_connections == 100

    def test_client_max_keepalive_connections_default(self) -> None:
        """Default max keepalive connections is 20."""
        settings = RolloutClientSettings()
        assert settings.max_keepalive_connections == 20

    def test_server_max_concurrent_rollouts_default(self) -> None:
        """Default max concurrent rollouts is 100."""
        settings = RolloutServerSettings()
        assert settings.max_concurrent_rollouts == 100

    def test_server_record_ttl_seconds_default(self) -> None:
        """Default record TTL is 3600 seconds (1 hour)."""
        settings = RolloutServerSettings()
        assert settings.record_ttl_seconds == 3600.0

    def test_server_cleanup_interval_seconds_default(self) -> None:
        """Default cleanup interval is 60 seconds."""
        settings = RolloutServerSettings()
        assert settings.cleanup_interval_seconds == 60.0

    def test_server_request_timeout_seconds_default(self) -> None:
        """Default request timeout is 600 seconds."""
        settings = RolloutServerSettings()
        assert settings.request_timeout_seconds == 600.0

    def test_server_registration_readiness_timeout_default(self) -> None:
        """Default registration readiness timeout is 10 seconds."""
        settings = RolloutServerSettings()
        assert settings.registration_readiness_timeout_seconds == 10.0

    def test_server_registration_readiness_poll_interval_default(self) -> None:
        """Default registration readiness poll interval is 0.2 seconds."""
        settings = RolloutServerSettings()
        assert settings.registration_readiness_poll_interval_seconds == 0.2

    def test_server_registration_shutdown_timeout_default(self) -> None:
        """Default registration shutdown timeout is 30 seconds."""
        settings = RolloutServerSettings()
        assert settings.registration_shutdown_timeout_seconds == 30.0

    def test_rollout_max_metadata_size_bytes_default(self) -> None:
        """Default max metadata size is 1MB."""
        settings = RolloutSettings()
        assert settings.max_metadata_size_bytes == 1024 * 1024

    def test_rollout_settings_has_client_and_server(self) -> None:
        """RolloutSettings contains client and server sub-settings."""
        settings = RolloutSettings()
        assert isinstance(settings.client, RolloutClientSettings)
        assert isinstance(settings.server, RolloutServerSettings)


# =============================================================================
# Singleton Pattern Tests (get_settings / configure / reset)
# =============================================================================


class TestSingleton:
    """Tests for the global settings singleton pattern."""

    def test_get_settings_returns_instance(self) -> None:
        """get_settings returns a RolloutSettings instance."""
        settings = get_settings()
        assert isinstance(settings, RolloutSettings)

    def test_get_settings_returns_same_instance(self) -> None:
        """Multiple calls to get_settings return the same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_settings_clears_singleton(self) -> None:
        """reset_settings causes get_settings to create a new instance."""
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_configure_sets_custom_settings(self) -> None:
        """configure() overrides the global singleton."""
        custom = RolloutSettings(
            max_metadata_size_bytes=2048,
        )
        configure(custom)
        s = get_settings()
        assert s is custom
        assert s.max_metadata_size_bytes == 2048

    def test_configure_overrides_previous_singleton(self) -> None:
        """configure() replaces any previously set singleton."""
        s1 = get_settings()
        custom = RolloutSettings()
        configure(custom)
        s2 = get_settings()
        assert s2 is custom
        assert s2 is not s1

    def test_reset_after_configure(self) -> None:
        """reset_settings works after configure."""
        custom = RolloutSettings(max_metadata_size_bytes=5000)
        configure(custom)
        reset_settings()
        s = get_settings()
        # Should be a fresh default instance, not the custom one
        assert s is not custom
        assert s.max_metadata_size_bytes == 1024 * 1024


# =============================================================================
# configure() Override Tests
# =============================================================================


class TestConfigureOverride:
    """Tests for programmatic configuration via configure()."""

    def test_configure_with_custom_client(self) -> None:
        """configure() with custom client settings."""
        custom = RolloutSettings(
            client=RolloutClientSettings(timeout_seconds=120.0, max_retries=5),
        )
        configure(custom)
        s = get_settings()
        assert s.client.timeout_seconds == 120.0
        assert s.client.max_retries == 5

    def test_configure_with_custom_server(self) -> None:
        """configure() with custom server settings."""
        custom = RolloutSettings(
            server=RolloutServerSettings(max_concurrent_rollouts=500),
        )
        configure(custom)
        s = get_settings()
        assert s.server.max_concurrent_rollouts == 500

    def test_configure_with_all_custom_values(self) -> None:
        """configure() with all sub-settings overridden."""
        custom = RolloutSettings(
            client=RolloutClientSettings(
                timeout_seconds=60.0,
                max_retries=1,
                retry_base_delay=0.5,
            ),
            server=RolloutServerSettings(
                max_concurrent_rollouts=200,
                record_ttl_seconds=7200.0,
            ),
            max_metadata_size_bytes=2048,
        )
        configure(custom)
        s = get_settings()
        assert s.client.timeout_seconds == 60.0
        assert s.client.max_retries == 1
        assert s.client.retry_base_delay == 0.5
        assert s.server.max_concurrent_rollouts == 200
        assert s.server.record_ttl_seconds == 7200.0
        assert s.max_metadata_size_bytes == 2048


# =============================================================================
# Field Validation Tests
# =============================================================================


class TestFieldValidation:
    """Tests that invalid values are rejected by Pydantic validation."""

    # --- Client settings validation ---

    def test_client_timeout_too_low(self) -> None:
        """Client timeout below minimum (1.0) is rejected."""
        with pytest.raises(ValidationError, match="timeout_seconds"):
            RolloutClientSettings(timeout_seconds=0.5)

    def test_client_timeout_too_high(self) -> None:
        """Client timeout above maximum (3600.0) is rejected."""
        with pytest.raises(ValidationError, match="timeout_seconds"):
            RolloutClientSettings(timeout_seconds=5000.0)

    def test_client_max_retries_negative(self) -> None:
        """Negative max_retries is rejected."""
        with pytest.raises(ValidationError, match="max_retries"):
            RolloutClientSettings(max_retries=-1)

    def test_client_max_retries_too_high(self) -> None:
        """max_retries above maximum (10) is rejected."""
        with pytest.raises(ValidationError, match="max_retries"):
            RolloutClientSettings(max_retries=20)

    def test_client_retry_base_delay_too_low(self) -> None:
        """retry_base_delay below minimum (0.1) is rejected."""
        with pytest.raises(ValidationError, match="retry_base_delay"):
            RolloutClientSettings(retry_base_delay=0.01)

    def test_client_max_connections_too_low(self) -> None:
        """max_connections below minimum (1) is rejected."""
        with pytest.raises(ValidationError, match="max_connections"):
            RolloutClientSettings(max_connections=0)

    def test_client_max_connections_too_high(self) -> None:
        """max_connections above maximum (1000) is rejected."""
        with pytest.raises(ValidationError, match="max_connections"):
            RolloutClientSettings(max_connections=2000)

    # --- Server settings validation ---

    def test_server_max_concurrent_too_low(self) -> None:
        """max_concurrent_rollouts below minimum (1) is rejected."""
        with pytest.raises(ValidationError, match="max_concurrent_rollouts"):
            RolloutServerSettings(max_concurrent_rollouts=0)

    def test_server_max_concurrent_too_high(self) -> None:
        """max_concurrent_rollouts above maximum (10000) is rejected."""
        with pytest.raises(ValidationError, match="max_concurrent_rollouts"):
            RolloutServerSettings(max_concurrent_rollouts=20000)

    def test_server_record_ttl_too_low(self) -> None:
        """record_ttl_seconds below minimum (60.0) is rejected."""
        with pytest.raises(ValidationError, match="record_ttl_seconds"):
            RolloutServerSettings(record_ttl_seconds=10.0)

    def test_server_record_ttl_too_high(self) -> None:
        """record_ttl_seconds above maximum (86400.0) is rejected."""
        with pytest.raises(ValidationError, match="record_ttl_seconds"):
            RolloutServerSettings(record_ttl_seconds=100000.0)

    def test_server_cleanup_interval_too_low(self) -> None:
        """cleanup_interval_seconds below minimum (10.0) is rejected."""
        with pytest.raises(ValidationError, match="cleanup_interval_seconds"):
            RolloutServerSettings(cleanup_interval_seconds=5.0)

    def test_server_request_timeout_too_low(self) -> None:
        """request_timeout_seconds below minimum (10.0) is rejected."""
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            RolloutServerSettings(request_timeout_seconds=5.0)

    def test_server_registration_readiness_timeout_too_low(self) -> None:
        """registration_readiness_timeout_seconds below minimum (1.0) is rejected."""
        with pytest.raises(
            ValidationError, match="registration_readiness_timeout_seconds"
        ):
            RolloutServerSettings(registration_readiness_timeout_seconds=0.5)

    # --- Global settings validation ---

    def test_max_metadata_size_bytes_too_low(self) -> None:
        """max_metadata_size_bytes below minimum (1024) is rejected."""
        with pytest.raises(ValidationError, match="max_metadata_size_bytes"):
            RolloutSettings(max_metadata_size_bytes=100)

    def test_max_metadata_size_bytes_too_high(self) -> None:
        """max_metadata_size_bytes above maximum (100MB) is rejected."""
        with pytest.raises(ValidationError, match="max_metadata_size_bytes"):
            RolloutSettings(max_metadata_size_bytes=200 * 1024 * 1024)

    # --- Valid boundary values ---

    def test_client_timeout_at_minimum(self) -> None:
        """Client timeout at exact minimum (1.0) is accepted."""
        s = RolloutClientSettings(timeout_seconds=1.0)
        assert s.timeout_seconds == 1.0

    def test_client_timeout_at_maximum(self) -> None:
        """Client timeout at exact maximum (3600.0) is accepted."""
        s = RolloutClientSettings(timeout_seconds=3600.0)
        assert s.timeout_seconds == 3600.0

    def test_server_max_concurrent_at_minimum(self) -> None:
        """max_concurrent_rollouts at exact minimum (1) is accepted."""
        s = RolloutServerSettings(max_concurrent_rollouts=1)
        assert s.max_concurrent_rollouts == 1

    def test_server_max_concurrent_at_maximum(self) -> None:
        """max_concurrent_rollouts at exact maximum (10000) is accepted."""
        s = RolloutServerSettings(max_concurrent_rollouts=10000)
        assert s.max_concurrent_rollouts == 10000


# =============================================================================
# Environment Variable Loading Tests
# =============================================================================


class TestEnvironmentVariables:
    """Tests for loading settings from environment variables."""

    def test_client_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client timeout_seconds loads from OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS", "120.0")
        settings = RolloutClientSettings()
        assert settings.timeout_seconds == 120.0

    def test_client_max_retries_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client max_retries loads from OSMOSIS_ROLLOUT_CLIENT_MAX_RETRIES."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_MAX_RETRIES", "5")
        settings = RolloutClientSettings()
        assert settings.max_retries == 5

    def test_client_retry_base_delay_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client retry_base_delay loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_RETRY_BASE_DELAY", "2.5")
        settings = RolloutClientSettings()
        assert settings.retry_base_delay == 2.5

    def test_client_max_connections_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client max_connections loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_MAX_CONNECTIONS", "50")
        settings = RolloutClientSettings()
        assert settings.max_connections == 50

    def test_server_max_concurrent_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server max_concurrent_rollouts loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS", "200")
        settings = RolloutServerSettings()
        assert settings.max_concurrent_rollouts == 200

    def test_server_record_ttl_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Server record_ttl_seconds loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_SERVER_RECORD_TTL_SECONDS", "7200")
        settings = RolloutServerSettings()
        assert settings.record_ttl_seconds == 7200.0

    def test_server_cleanup_interval_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server cleanup_interval_seconds loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_SERVER_CLEANUP_INTERVAL_SECONDS", "120")
        settings = RolloutServerSettings()
        assert settings.cleanup_interval_seconds == 120.0

    def test_server_request_timeout_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server request_timeout_seconds loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_SERVER_REQUEST_TIMEOUT_SECONDS", "300")
        settings = RolloutServerSettings()
        assert settings.request_timeout_seconds == 300.0

    def test_max_metadata_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Global max_metadata_size_bytes loads from env."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES", "2048")
        settings = RolloutSettings()
        assert settings.max_metadata_size_bytes == 2048

    def test_invalid_env_var_value_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid env var value (not a number) is rejected."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS", "not_a_number")
        with pytest.raises(ValidationError):
            RolloutClientSettings()

    def test_env_var_violating_constraint_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var value that violates constraint (too low) is rejected."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS", "0.1")
        with pytest.raises(ValidationError):
            RolloutClientSettings()

    def test_get_settings_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_settings() singleton also picks up env vars on first call."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES", "4096")
        settings = get_settings()
        assert settings.max_metadata_size_bytes == 4096

    def test_extra_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown env vars with the right prefix are ignored (extra='ignore')."""
        monkeypatch.setenv("OSMOSIS_ROLLOUT_CLIENT_NONEXISTENT_FIELD", "42")
        # Should not raise
        settings = RolloutClientSettings()
        assert not hasattr(settings, "nonexistent_field")


# =============================================================================
# Programmatic Construction Tests
# =============================================================================


class TestProgrammaticConstruction:
    """Tests for constructing settings programmatically."""

    def test_construct_client_with_all_fields(self) -> None:
        """Construct RolloutClientSettings with all fields specified."""
        s = RolloutClientSettings(
            timeout_seconds=60.0,
            max_retries=5,
            complete_rollout_retries=3,
            retry_base_delay=2.0,
            retry_max_delay=60.0,
            max_connections=200,
            max_keepalive_connections=50,
        )
        assert s.timeout_seconds == 60.0
        assert s.max_retries == 5
        assert s.complete_rollout_retries == 3
        assert s.retry_base_delay == 2.0
        assert s.retry_max_delay == 60.0
        assert s.max_connections == 200
        assert s.max_keepalive_connections == 50

    def test_construct_server_with_all_fields(self) -> None:
        """Construct RolloutServerSettings with all fields specified."""
        s = RolloutServerSettings(
            max_concurrent_rollouts=500,
            record_ttl_seconds=1800.0,
            cleanup_interval_seconds=30.0,
            request_timeout_seconds=120.0,
            registration_readiness_timeout_seconds=5.0,
            registration_readiness_poll_interval_seconds=0.5,
            registration_shutdown_timeout_seconds=60.0,
        )
        assert s.max_concurrent_rollouts == 500
        assert s.record_ttl_seconds == 1800.0
        assert s.cleanup_interval_seconds == 30.0
        assert s.request_timeout_seconds == 120.0
        assert s.registration_readiness_timeout_seconds == 5.0
        assert s.registration_readiness_poll_interval_seconds == 0.5
        assert s.registration_shutdown_timeout_seconds == 60.0

    def test_construct_rollout_settings_nested(self) -> None:
        """Construct RolloutSettings with nested sub-settings."""
        s = RolloutSettings(
            client=RolloutClientSettings(timeout_seconds=99.0),
            server=RolloutServerSettings(max_concurrent_rollouts=42),
            max_metadata_size_bytes=8192,
        )
        assert s.client.timeout_seconds == 99.0
        assert s.server.max_concurrent_rollouts == 42
        assert s.max_metadata_size_bytes == 8192


# =============================================================================
# Parametrized Boundary Tests
# =============================================================================


class TestParametrizedBoundaries:
    """Parametrized tests for boundary validation."""

    @pytest.mark.parametrize(
        "value",
        [1.0, 100.0, 3600.0],
        ids=["min", "mid", "max"],
    )
    def test_client_timeout_valid_range(self, value: float) -> None:
        """Client timeout accepts values in valid range [1.0, 3600.0]."""
        s = RolloutClientSettings(timeout_seconds=value)
        assert s.timeout_seconds == value

    @pytest.mark.parametrize(
        "value",
        [0.5, 0.0, -1.0, 3601.0],
        ids=["below_min", "zero", "negative", "above_max"],
    )
    def test_client_timeout_invalid_range(self, value: float) -> None:
        """Client timeout rejects values outside valid range."""
        with pytest.raises(ValidationError):
            RolloutClientSettings(timeout_seconds=value)

    @pytest.mark.parametrize(
        "value",
        [0, 1, 5, 10],
        ids=["zero", "one", "five", "ten"],
    )
    def test_client_max_retries_valid_range(self, value: int) -> None:
        """Client max_retries accepts values in valid range [0, 10]."""
        s = RolloutClientSettings(max_retries=value)
        assert s.max_retries == value

    @pytest.mark.parametrize(
        "value",
        [60.0, 3600.0, 86400.0],
        ids=["min", "mid", "max"],
    )
    def test_server_record_ttl_valid_range(self, value: float) -> None:
        """Server record_ttl_seconds accepts values in valid range."""
        s = RolloutServerSettings(record_ttl_seconds=value)
        assert s.record_ttl_seconds == value

    @pytest.mark.parametrize(
        "value",
        [1, 100, 5000, 10000],
        ids=["min", "default", "mid", "max"],
    )
    def test_server_max_concurrent_valid_range(self, value: int) -> None:
        """Server max_concurrent_rollouts accepts values in valid range."""
        s = RolloutServerSettings(max_concurrent_rollouts=value)
        assert s.max_concurrent_rollouts == value

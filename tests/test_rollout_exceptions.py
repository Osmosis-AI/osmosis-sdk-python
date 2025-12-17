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

"""Tests for osmosis_ai.rollout.exceptions."""

from __future__ import annotations

from osmosis_ai.rollout import (
    AgentLoopNotFoundError,
    OsmosisRolloutError,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
)


def test_osmosis_rollout_error_is_base_exception() -> None:
    """Verify OsmosisRolloutError is the base class."""
    error = OsmosisRolloutError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"


def test_osmosis_transport_error() -> None:
    """Verify OsmosisTransportError inherits from base."""
    error = OsmosisTransportError("connection failed")
    assert isinstance(error, OsmosisRolloutError)
    assert str(error) == "connection failed"


def test_osmosis_server_error_with_status_code() -> None:
    """Verify OsmosisServerError includes status_code."""
    error = OsmosisServerError("internal server error", 500)
    assert isinstance(error, OsmosisRolloutError)
    assert error.status_code == 500
    assert str(error) == "internal server error"


def test_osmosis_server_error_different_status_codes() -> None:
    """Verify OsmosisServerError works with various 5xx codes."""
    for status_code in [500, 502, 503, 504]:
        error = OsmosisServerError(f"error {status_code}", status_code)
        assert error.status_code == status_code


def test_osmosis_validation_error_with_status_code() -> None:
    """Verify OsmosisValidationError includes status_code."""
    error = OsmosisValidationError("bad request", 400)
    assert isinstance(error, OsmosisRolloutError)
    assert error.status_code == 400
    assert str(error) == "bad request"


def test_osmosis_validation_error_different_status_codes() -> None:
    """Verify OsmosisValidationError works with various 4xx codes."""
    for status_code in [400, 401, 403, 404, 422]:
        error = OsmosisValidationError(f"error {status_code}", status_code)
        assert error.status_code == status_code


def test_osmosis_timeout_error() -> None:
    """Verify OsmosisTimeoutError inherits from base."""
    error = OsmosisTimeoutError("request timed out")
    assert isinstance(error, OsmosisRolloutError)
    assert str(error) == "request timed out"


def test_agent_loop_not_found_error_message() -> None:
    """Verify AgentLoopNotFoundError has correct message format."""
    error = AgentLoopNotFoundError("missing_agent", ["agent1", "agent2"])
    assert isinstance(error, OsmosisRolloutError)
    assert "missing_agent" in str(error)
    assert "['agent1', 'agent2']" in str(error)


def test_agent_loop_not_found_error_attributes() -> None:
    """Verify AgentLoopNotFoundError stores name and available list."""
    error = AgentLoopNotFoundError("my_agent", ["foo", "bar", "baz"])
    assert error.name == "my_agent"
    assert error.available == ["foo", "bar", "baz"]


def test_agent_loop_not_found_error_empty_available() -> None:
    """Verify AgentLoopNotFoundError works with empty available list."""
    error = AgentLoopNotFoundError("my_agent", [])
    assert error.name == "my_agent"
    assert error.available == []
    assert "[]" in str(error)


def test_exception_inheritance_hierarchy() -> None:
    """Verify all exceptions inherit from OsmosisRolloutError."""
    assert issubclass(OsmosisTransportError, OsmosisRolloutError)
    assert issubclass(OsmosisServerError, OsmosisRolloutError)
    assert issubclass(OsmosisValidationError, OsmosisRolloutError)
    assert issubclass(OsmosisTimeoutError, OsmosisRolloutError)
    assert issubclass(AgentLoopNotFoundError, OsmosisRolloutError)


def test_exceptions_can_be_caught_by_base() -> None:
    """Verify all exceptions can be caught by base class."""
    exceptions = [
        OsmosisTransportError("test"),
        OsmosisServerError("test", 500),
        OsmosisValidationError("test", 400),
        OsmosisTimeoutError("test"),
        AgentLoopNotFoundError("test", []),
    ]

    for exc in exceptions:
        try:
            raise exc
        except OsmosisRolloutError as e:
            assert e is exc

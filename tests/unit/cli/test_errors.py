"""Tests for the structured CLIError signature."""

from __future__ import annotations

from osmosis_ai.cli.errors import CLIError


def test_positional_message_defaults_to_validation() -> None:
    err = CLIError("Something went wrong.")
    assert err.message == "Something went wrong."
    assert str(err) == "Something went wrong."
    assert err.code == "VALIDATION"
    assert err.details == {}
    assert err.request_id is None


def test_explicit_code_is_preserved() -> None:
    err = CLIError("Not found.", code="NOT_FOUND")
    assert err.code == "NOT_FOUND"


def test_details_coerced_to_dict() -> None:
    err = CLIError("Bad", details={"field": "name"})
    assert err.details == {"field": "name"}


def test_details_default_is_empty_dict() -> None:
    err = CLIError("Bad")
    assert err.details == {}
    err.details["x"] = 1
    assert CLIError("Other").details == {}


def test_request_id_is_optional() -> None:
    err = CLIError("Bad", request_id="req_abc")
    assert err.request_id == "req_abc"


def test_empty_message_is_allowed() -> None:
    err = CLIError(code="INTERACTIVE_REQUIRED")
    assert err.message == ""
    assert err.code == "INTERACTIVE_REQUIRED"

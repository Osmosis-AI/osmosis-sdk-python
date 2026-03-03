"""Tests for project name validation logic.

Covers: empty names, length limits, casing, character restrictions, reserved names,
and regex edge cases.
"""

from __future__ import annotations

import pytest

from osmosis_ai.platform.cli.project import (
    _RESERVED_PROJECT_NAMES,
    validate_project_name,
)

# ---------------------------------------------------------------------------
# Valid names (should return None)
# ---------------------------------------------------------------------------


class TestValidateProjectName:
    """Tests for the validate_project_name helper."""

    # -- valid names --------------------------------------------------------

    def test_valid_simple(self) -> None:
        assert validate_project_name("my-project") is None

    def test_valid_single_char(self) -> None:
        assert validate_project_name("a") is None

    def test_valid_two_chars(self) -> None:
        assert validate_project_name("ab") is None

    def test_valid_digits_only(self) -> None:
        assert validate_project_name("123") is None

    def test_valid_mixed(self) -> None:
        assert validate_project_name("my-project-2") is None

    def test_valid_max_length(self) -> None:
        assert validate_project_name("a" * 64) is None

    # -- invalid names ------------------------------------------------------

    def test_empty_string(self) -> None:
        result = validate_project_name("")
        assert result is not None
        assert "required" in result

    def test_too_long(self) -> None:
        result = validate_project_name("a" * 65)
        assert result is not None
        assert "64 characters" in result

    def test_uppercase(self) -> None:
        result = validate_project_name("MyProject")
        assert result is not None
        assert "lowercase" in result

    def test_starts_with_hyphen(self) -> None:
        result = validate_project_name("-project")
        assert result is not None
        assert "cannot start or end with a hyphen" in result

    def test_ends_with_hyphen(self) -> None:
        result = validate_project_name("project-")
        assert result is not None
        assert "cannot start or end with a hyphen" in result

    def test_special_chars(self) -> None:
        result = validate_project_name("my_project")
        assert result is not None
        assert "lowercase letters, digits, and hyphens" in result

    def test_spaces(self) -> None:
        result = validate_project_name("my project")
        assert result is not None
        assert "lowercase letters, digits, and hyphens" in result

    def test_dots(self) -> None:
        result = validate_project_name("my.project")
        assert result is not None
        assert "lowercase letters, digits, and hyphens" in result

    # -- reserved names (parametrized) --------------------------------------

    @pytest.mark.parametrize(
        "name",
        sorted(_RESERVED_PROJECT_NAMES),
        ids=sorted(_RESERVED_PROJECT_NAMES),
    )
    def test_reserved_names(self, name: str) -> None:
        result = validate_project_name(name)
        assert result is not None
        assert f"'{name}' is a reserved name" in result

    # -- edge cases ---------------------------------------------------------

    def test_single_hyphen(self) -> None:
        result = validate_project_name("-")
        assert result is not None

    def test_double_hyphen_ok(self) -> None:
        assert validate_project_name("a--b") is None

    def test_exactly_64_chars_valid(self) -> None:
        name = "a" + "b" * 62 + "c"
        assert len(name) == 64
        assert validate_project_name(name) is None

    def test_65_chars_invalid(self) -> None:
        name = "a" + "b" * 63 + "c"
        assert len(name) == 65
        result = validate_project_name(name)
        assert result is not None
        assert "64 characters" in result

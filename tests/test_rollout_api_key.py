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

"""Tests for osmosis_ai.rollout.server.api_key."""

from __future__ import annotations

import pytest

from osmosis_ai.rollout.server.api_key import (
    generate_api_key,
    validate_api_key,
    API_KEY_PREFIX,
)


class TestGenerateApiKey:
    """Tests for generate_api_key function."""

    def test_generate_api_key_has_prefix(self) -> None:
        """Verify generated key has correct prefix."""
        key = generate_api_key()
        assert key.startswith(API_KEY_PREFIX)

    def test_generate_api_key_is_unique(self) -> None:
        """Verify each call generates a unique key."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_generate_api_key_is_url_safe(self) -> None:
        """Verify generated key contains only URL-safe characters."""
        key = generate_api_key()
        # URL-safe base64 uses only alphanumeric, '-', and '_'
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        for char in key:
            assert char in allowed, f"Unexpected character: {char}"

    def test_generate_api_key_has_sufficient_length(self) -> None:
        """Verify generated key is long enough for security."""
        key = generate_api_key()
        # Prefix + 43 chars (32 bytes in base64)
        assert len(key) >= len(API_KEY_PREFIX) + 40


class TestValidateApiKey:
    """Tests for validate_api_key function."""

    def test_validate_api_key_matching(self) -> None:
        """Verify matching keys return True."""
        key = "test-key-12345"
        assert validate_api_key(key, key) is True

    def test_validate_api_key_not_matching(self) -> None:
        """Verify non-matching keys return False."""
        assert validate_api_key("key1", "key2") is False

    def test_validate_api_key_none_provided(self) -> None:
        """Verify None provided returns False."""
        assert validate_api_key(None, "expected-key") is False

    def test_validate_api_key_empty_string(self) -> None:
        """Verify empty string doesn't match non-empty key."""
        assert validate_api_key("", "expected-key") is False

    def test_validate_api_key_generated(self) -> None:
        """Verify validation works with generated keys."""
        key = generate_api_key()
        assert validate_api_key(key, key) is True
        assert validate_api_key(key + "x", key) is False

    def test_validate_api_key_case_sensitive(self) -> None:
        """Verify validation is case-sensitive."""
        assert validate_api_key("TestKey", "testkey") is False
        assert validate_api_key("TestKey", "TestKey") is True

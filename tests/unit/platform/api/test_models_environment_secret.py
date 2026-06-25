"""Tests for EnvironmentSecretInfo.scope round-trip from API payloads."""

from __future__ import annotations

from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)


def test_environment_secret_info_parses_scope() -> None:
    info = EnvironmentSecretInfo.from_dict(
        {
            "id": "sec_1",
            "name": "OPENAI_API_KEY",
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-01T00:00:01Z",
            "creator_name": "Ada",
            "scope": "user",
        }
    )
    assert info.scope == "user"


def test_environment_secret_info_scope_defaults_to_none_when_absent() -> None:
    info = EnvironmentSecretInfo.from_dict({"id": "sec_1", "name": "X"})
    assert info.scope is None


def test_paginated_environment_secrets_parses_scope_per_item() -> None:
    page = PaginatedEnvironmentSecrets.from_dict(
        {
            "environment_secrets": [
                {"id": "a", "name": "A", "scope": "workspace"},
                {"id": "b", "name": "B", "scope": "user"},
            ],
            "total_count": 2,
            "has_more": False,
        }
    )
    assert [s.scope for s in page.environment_secrets] == ["workspace", "user"]

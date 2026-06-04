"""Tests for OsmosisClient environment-secret endpoints (set/list/delete)."""

from __future__ import annotations

from typing import Any

import pytest

import osmosis_ai.platform.api.client as client_module
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)

GIT_IDENTITY = "acme/rollouts"
CREDS = object()


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_request(
        endpoint,
        method="GET",
        data=None,
        *,
        credentials=None,
        git_identity=None,
        **kwargs,
    ):
        calls["endpoint"] = endpoint
        calls["method"] = method
        calls["data"] = data
        calls["credentials"] = credentials
        calls["git_identity"] = git_identity
        return {
            "id": "sec_1",
            "name": "OPENAI_API_KEY",
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-01T00:00:01Z",
            "scope": "user",
            "environment_secrets": [
                {"id": "sec_1", "name": "OPENAI_API_KEY", "scope": "user"}
            ],
            "total_count": 1,
            "has_more": False,
        }

    monkeypatch.setattr(client_module, "platform_request", fake_request)
    return calls


def test_set_environment_secret_posts_name_value_scope(captured) -> None:
    result = OsmosisClient().set_environment_secret(
        "OPENAI_API_KEY",
        "sk-x",
        scope="user",
        credentials=CREDS,
        git_identity=GIT_IDENTITY,
    )
    assert captured["endpoint"] == "/api/cli/environment-secrets"
    assert captured["method"] == "POST"
    assert captured["data"] == {
        "name": "OPENAI_API_KEY",
        "value": "sk-x",
        "scope": "user",
    }
    assert captured["credentials"] is CREDS
    assert captured["git_identity"] == GIT_IDENTITY
    assert isinstance(result, EnvironmentSecretInfo)
    assert result.scope == "user"


def test_delete_environment_secret_sends_delete_with_name_scope(captured) -> None:
    OsmosisClient().delete_environment_secret(
        "OPENAI_API_KEY",
        scope="workspace",
        credentials=CREDS,
        git_identity=GIT_IDENTITY,
    )
    assert captured["endpoint"] == "/api/cli/environment-secrets"
    assert captured["method"] == "DELETE"
    assert captured["data"] == {"name": "OPENAI_API_KEY", "scope": "workspace"}


def test_list_environment_secrets_puts_scope_in_query(captured) -> None:
    OsmosisClient().list_environment_secrets(
        limit=10,
        offset=0,
        scope="workspace",
        credentials=CREDS,
        git_identity=GIT_IDENTITY,
    )
    assert captured["endpoint"].startswith("/api/cli/environment-secrets?")
    assert "scope=workspace" in captured["endpoint"]
    assert "limit=10" in captured["endpoint"]
    assert "offset=0" in captured["endpoint"]


def test_list_environment_secrets_default_scope_is_all(captured) -> None:
    OsmosisClient().list_environment_secrets(
        credentials=CREDS,
        git_identity=GIT_IDENTITY,
    )
    assert "scope=all" in captured["endpoint"]


def test_list_environment_secrets_returns_paginated(captured) -> None:
    page = OsmosisClient().list_environment_secrets(
        credentials=CREDS,
        git_identity=GIT_IDENTITY,
    )
    assert isinstance(page, PaginatedEnvironmentSecrets)
    assert page.environment_secrets[0].scope == "user"

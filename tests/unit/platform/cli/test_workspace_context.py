from __future__ import annotations

from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.workspace_context import (
    WorkspaceRefResolutionError,
    list_accessible_workspaces,
    resolve_workspace_ref,
)


class _Creds:
    def is_expired(self) -> bool:
        return False


def test_resolve_workspace_ref_matches_id_before_name() -> None:
    workspaces: list[dict[str, Any]] = [
        {"id": "team-alpha", "name": "other"},
        {"id": "ws_2", "name": "team-alpha"},
    ]

    assert resolve_workspace_ref("team-alpha", workspaces)["id"] == "team-alpha"


def test_resolve_workspace_ref_rejects_ambiguous_case_insensitive_name() -> None:
    workspaces: list[dict[str, Any]] = [
        {"id": "ws_1", "name": "Team"},
        {"id": "ws_2", "name": "team"},
    ]

    with pytest.raises(WorkspaceRefResolutionError, match="ambiguous"):
        resolve_workspace_ref("TEAM", workspaces)


def test_resolve_workspace_ref_matches_exact_name() -> None:
    workspaces: list[dict[str, Any]] = [
        {"id": "ws_1", "name": "team-alpha"},
        {"id": "ws_2", "name": "team-beta"},
    ]

    assert resolve_workspace_ref("team-beta", workspaces)["id"] == "ws_2"


def test_resolve_workspace_ref_matches_case_insensitive_name() -> None:
    workspaces: list[dict[str, Any]] = [
        {"id": "ws_1", "name": "Team Alpha"},
        {"id": "ws_2", "name": "team-beta"},
    ]

    assert resolve_workspace_ref("TEAM ALPHA", workspaces)["id"] == "ws_1"


def test_resolve_workspace_ref_does_not_coerce_non_string_names() -> None:
    workspaces: list[dict[str, Any]] = [{"id": "ws_1", "name": 123}]

    with pytest.raises(WorkspaceRefResolutionError, match="not found"):
        resolve_workspace_ref("123", workspaces)


def test_resolve_workspace_ref_reports_not_found() -> None:
    workspaces: list[dict[str, Any]] = [{"id": "ws_1", "name": "team-alpha"}]

    with pytest.raises(WorkspaceRefResolutionError, match="not found"):
        resolve_workspace_ref("missing", workspaces)


def test_list_accessible_workspaces_returns_list_and_forwards_request_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    workspaces = [{"id": "ws_1", "name": "team-alpha"}]

    def fake_platform_request(path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"path": path, **kwargs})
        return {"workspaces": workspaces}

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.platform_request",
        fake_platform_request,
    )

    assert list_accessible_workspaces(credentials=_Creds()) == workspaces
    assert calls == [
        {
            "path": "/api/cli/workspaces",
            "credentials": calls[0]["credentials"],
            "require_workspace": False,
            "cleanup_on_401": False,
        }
    ]


def test_list_accessible_workspaces_rejects_malformed_workspaces_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.platform_request",
        lambda *args, **kwargs: {"workspaces": {"id": "ws_1"}},
    )

    with pytest.raises(CLIError, match="Invalid workspaces response"):
        list_accessible_workspaces(credentials=_Creds())


def test_list_accessible_workspaces_rejects_malformed_workspace_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.platform_request",
        lambda *args, **kwargs: {"workspaces": ["ws_1"]},
    )

    with pytest.raises(CLIError, match="Invalid workspace entry"):
        list_accessible_workspaces(credentials=_Creds())

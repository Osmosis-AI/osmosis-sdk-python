from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.project_mapping import (
    ProjectLinkRecord,
    ProjectMappingStore,
)
from osmosis_ai.platform.cli.workspace_context import (
    WorkspaceRefResolutionError,
    list_accessible_workspaces,
    refresh_workspace_by_id,
    resolve_linked_workspace_context,
    resolve_workspace_ref,
)


class _Creds:
    def is_expired(self) -> bool:
        return False


def _make_project(root: Path) -> Path:
    for rel_path in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel_path).mkdir(parents=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".osmosis" / "program.md").write_text("# Program\n", encoding="utf-8")
    return root


def _make_minimal_project(root: Path) -> Path:
    (root / ".osmosis").mkdir(parents=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    return root


def test_resolve_linked_workspace_context_reads_current_platform_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "project")
    store_path = tmp_path / "config.json"
    store = ProjectMappingStore(
        config_file=store_path, platform_url="https://platform.osmosis.ai"
    )
    store.link(
        ProjectLinkRecord(
            str(project.resolve()),
            "ws_1",
            "team-alpha",
            None,
            "2026-05-03T00:00:00+00:00",
        )
    )
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.CONFIG_FILE", store_path
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.load_credentials", lambda: _Creds()
    )

    ctx = resolve_linked_workspace_context()

    assert ctx.project_root == project.resolve()
    assert ctx.workspace_id == "ws_1"
    assert ctx.workspace_name == "team-alpha"


def test_unlinked_project_reports_project_link_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.CONFIG_FILE",
        tmp_path / "config.json",
    )

    with pytest.raises(CLIError, match="This project is not linked"):
        resolve_linked_workspace_context()


def test_linked_project_with_missing_contract_paths_reports_contract_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_minimal_project(tmp_path / "project")
    store_path = tmp_path / "config.json"
    ProjectMappingStore(
        config_file=store_path, platform_url="https://platform.osmosis.ai"
    ).link(
        ProjectLinkRecord(
            str(project.resolve()),
            "ws_1",
            "team-alpha",
            None,
            "2026-05-03T00:00:00+00:00",
        )
    )
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.CONFIG_FILE", store_path
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.load_credentials", lambda: _Creds()
    )

    with pytest.raises(CLIError, match="Project is missing required Osmosis paths"):
        resolve_linked_workspace_context()


def test_nested_unlinked_project_does_not_use_parent_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _make_project(tmp_path / "parent")
    child = _make_project(parent / "child")
    store_path = tmp_path / "config.json"
    ProjectMappingStore(
        config_file=store_path, platform_url="https://platform.osmosis.ai"
    ).link(
        ProjectLinkRecord(
            str(parent.resolve()),
            "ws_parent",
            "parent",
            None,
            "2026-05-03T00:00:00+00:00",
        )
    )
    monkeypatch.chdir(child)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.CONFIG_FILE", store_path
    )

    with pytest.raises(CLIError, match="This project is not linked"):
        resolve_linked_workspace_context()


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


def test_refresh_workspace_by_id_returns_matching_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspaces = [
        {"id": "ws_1", "name": "team-alpha"},
        {"id": "ws_2", "name": "team-beta"},
    ]
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.list_accessible_workspaces",
        lambda *, credentials: workspaces,
    )

    assert refresh_workspace_by_id(workspace_id="ws_2", credentials=_Creds()) == {
        "id": "ws_2",
        "name": "team-beta",
    }


def test_refresh_workspace_by_id_reports_stale_or_inaccessible_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_context.list_accessible_workspaces",
        lambda *, credentials: [{"id": "ws_1", "name": "team-alpha"}],
    )

    with pytest.raises(CLIError, match="no longer accessible"):
        refresh_workspace_by_id(workspace_id="ws_missing", credentials=_Creds())

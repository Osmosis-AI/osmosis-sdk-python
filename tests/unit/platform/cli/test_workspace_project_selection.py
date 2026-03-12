"""Regression tests for workspace-specific project selection."""

from __future__ import annotations

import osmosis_ai.platform.cli.project as project_module
from osmosis_ai.platform.auth.platform_client import PlatformAPIError


def test_select_project_interactive_refreshes_selected_workspace_projects(
    monkeypatch,
) -> None:
    calls: dict[str, str] = {}
    target_project = {"id": "proj-b", "project_name": "target-project"}

    monkeypatch.setattr(project_module, "is_interactive", lambda: False)

    def fake_refresh_projects(*, workspace_name: str) -> list[dict[str, str]]:
        calls["workspace_name"] = workspace_name
        return [target_project]

    monkeypatch.setattr(project_module, "_refresh_projects", fake_refresh_projects)

    result = project_module.select_project_interactive("ws-b")

    assert calls["workspace_name"] == "ws-b"
    assert result == target_project


def test_select_project_interactive_uses_selected_workspace_cache_on_refresh_failure(
    monkeypatch,
) -> None:
    calls: dict[str, str | float] = {}
    target_project = {"id": "proj-b", "project_name": "target-project"}

    monkeypatch.setattr(project_module, "is_interactive", lambda: False)

    def fake_refresh_projects(*, workspace_name: str) -> list[dict]:
        raise PlatformAPIError(f"boom for {workspace_name}")

    def fake_get_cached_projects(
        *,
        workspace_name: str,
        max_age: float | None = project_module.CACHE_TTL_SECONDS,
    ) -> list[dict[str, str]]:
        calls["workspace_name"] = workspace_name
        calls["max_age"] = max_age
        return [target_project]

    monkeypatch.setattr(project_module, "_refresh_projects", fake_refresh_projects)
    monkeypatch.setattr(
        project_module, "_get_cached_projects", fake_get_cached_projects
    )

    result = project_module.select_project_interactive("ws-b")

    assert calls["workspace_name"] == "ws-b"
    assert calls["max_age"] is None
    assert result == target_project


def test_get_cached_projects_reads_selected_workspace_cache(monkeypatch) -> None:
    target_projects = [{"id": "proj-b", "project_name": "target-project"}]

    def fake_load_workspace_projects(
        workspace_name: str,
    ) -> tuple[list[dict[str, str]], float | None]:
        if workspace_name == "ws-b":
            return target_projects, 123.0
        raise AssertionError(f"unexpected workspace: {workspace_name}")

    monkeypatch.setattr(
        project_module, "load_workspace_projects", fake_load_workspace_projects
    )

    projects = project_module._get_cached_projects(workspace_name="ws-b", max_age=None)

    assert projects == target_projects


def test_refresh_projects_uses_selected_workspace_credentials_and_cache(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    target_projects = [{"id": "proj-b", "project_name": "target-project"}]

    class FakeCredentials:
        def is_expired(self) -> bool:
            return False

    fake_credentials = FakeCredentials()

    monkeypatch.setattr(
        project_module,
        "load_workspace_credentials",
        lambda workspace_name: fake_credentials if workspace_name == "ws-b" else None,
        raising=False,
    )

    class FakeClient:
        def refresh_workspace_info(self, *, credentials=None) -> dict:
            calls["credentials"] = credentials
            return {"projects": target_projects, "has_subscription": True}

    monkeypatch.setattr(project_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(
        project_module,
        "save_workspace_projects",
        lambda workspace_name, projects: calls.setdefault("project_saves", []).append(
            (workspace_name, projects)
        ),
    )
    monkeypatch.setattr(
        project_module,
        "save_subscription_status",
        lambda workspace_name, has_subscription: calls.setdefault(
            "subscription_saves", []
        ).append((workspace_name, has_subscription)),
    )

    projects = project_module._refresh_projects(workspace_name="ws-b")

    assert calls["credentials"] is fake_credentials
    assert calls["project_saves"] == [("ws-b", target_projects)]
    assert calls["subscription_saves"] == [("ws-b", True)]
    assert projects == target_projects


def test_resolve_project_uses_selected_workspace_default_and_cache(
    monkeypatch,
) -> None:
    calls: dict[str, str | float] = {}
    target_project = {"id": "proj-b", "project_name": "target-project"}
    monkeypatch.setattr(
        project_module,
        "get_default_project",
        lambda workspace_name: (
            {"project_name": "active-project"}
            if workspace_name == "ws-a"
            else {"project_name": "target-project"}
        ),
    )

    def fake_get_cached_projects(
        *,
        workspace_name: str,
        max_age: float | None = project_module.CACHE_TTL_SECONDS,
    ) -> list[dict[str, str]]:
        calls["workspace_name"] = workspace_name
        calls["max_age"] = max_age
        return [target_project]

    monkeypatch.setattr(
        project_module, "_get_cached_projects", fake_get_cached_projects
    )

    result = project_module._resolve_project(None, workspace_name="ws-b")

    assert calls["workspace_name"] == "ws-b"
    assert calls["max_age"] == project_module.CACHE_TTL_SECONDS
    assert result == target_project


def test_require_subscription_refreshes_selected_workspace(monkeypatch) -> None:
    load_calls: list[tuple[str, float | None]] = []
    refresh_calls: list[str] = []

    call_count = 0

    def fake_load_subscription_status(
        workspace_name: str,
        max_age: float | None = None,
    ) -> bool | None:
        nonlocal call_count
        call_count += 1
        load_calls.append((workspace_name, max_age))
        # First call: cache miss/expired; second call: refresh wrote fresh data
        return None if call_count == 1 else True

    def fake_refresh_projects(*, workspace_name: str) -> list[dict]:
        refresh_calls.append(workspace_name)
        return []

    monkeypatch.setattr(
        project_module, "load_subscription_status", fake_load_subscription_status
    )
    monkeypatch.setattr(project_module, "_refresh_projects", fake_refresh_projects)

    project_module._require_subscription(workspace_name="ws-b")

    assert load_calls == [
        ("ws-b", project_module.CACHE_TTL_SECONDS),
        ("ws-b", project_module.CACHE_TTL_SECONDS),
    ]
    assert refresh_calls == ["ws-b"]

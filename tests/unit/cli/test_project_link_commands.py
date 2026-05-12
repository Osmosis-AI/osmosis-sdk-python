from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.cli.main import main
from osmosis_ai.platform.cli.project_mapping import (
    ProjectLinkRecord,
    ProjectMappingStore,
)

_CONNECTED_REPO = "https://github.com/acme/project.git"


class _Creds:
    def is_expired(self) -> bool:
        return False


def _make_project(root: Path) -> Path:
    for rel in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test Program\n", encoding="utf-8"
    )
    return root


def _workspace(
    *,
    workspace_id: str = "ws_1",
    name: str = "team-alpha",
    repo_url: str | None = _CONNECTED_REPO,
) -> dict[str, object]:
    return {
        "id": workspace_id,
        "name": name,
        "connected_repo": {"repo_url": repo_url} if repo_url is not None else None,
    }


def _allow_matching_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.get_local_git_remote_url",
        lambda project_root: _CONNECTED_REPO,
    )


@pytest.fixture
def isolated_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.json"
    monkeypatch.setattr("osmosis_ai.platform.cli.project_mapping.CONFIG_FILE", path)
    monkeypatch.setattr("osmosis_ai.platform.cli.workspace_context.CONFIG_FILE", path)
    monkeypatch.setattr("osmosis_ai.platform.cli.project.CONFIG_FILE", path)
    return path


def _seed_link(
    config_file: Path,
    project: Path,
    *,
    workspace_id: str = "ws_1",
    workspace_name: str = "team-alpha",
    repo_url: str | None = None,
    platform_url: str = "https://platform.osmosis.ai",
) -> None:
    ProjectMappingStore(
        config_file=config_file,
        platform_url=platform_url,
    ).link(
        ProjectLinkRecord(
            project_path=str(project.resolve()),
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            repo_url=repo_url,
            linked_at="2026-05-03T00:00:00+00:00",
        )
    )


def test_project_link_by_workspace_id_writes_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace()],
    )
    _allow_matching_origin(monkeypatch)

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert "data" not in payload
    assert payload["resource"]["workspace"]["id"] == "ws_1"
    raw = json.loads(isolated_mapping.read_text(encoding="utf-8"))
    assert (
        raw["platforms"]["https://platform.osmosis.ai"]["projects"][
            str(project.resolve())
        ]["workspaceId"]
        == "ws_1"
    )


def test_project_link_non_interactive_requires_workspace_and_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "link"])

    assert rc == 1
    assert "INTERACTIVE_REQUIRED" in capsys.readouterr().err


def test_project_link_no_accessible_workspaces_points_to_platform_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [],
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "No accessible Osmosis workspaces" in error["message"]
    assert "Platform" in error["message"]
    assert "Git Sync" in error["message"]
    assert "osmosis workspace" not in error["message"]
    assert "workspace create" not in error["message"]
    assert "workspace list" not in error["message"]
    assert not isolated_mapping.exists()


def test_project_link_workspace_not_found_points_to_platform_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace(workspace_id="ws_other", name="other")],
    )

    rc = main(["--json", "project", "link", "--workspace", "missing", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Workspace 'missing' not found" in error["message"]
    assert "Platform" in error["message"]
    assert "Git Sync" in error["message"]
    assert "osmosis workspace" not in error["message"]
    assert "workspace create" not in error["message"]
    assert "workspace list" not in error["message"]
    assert not isolated_mapping.exists()


def test_project_unlink_is_idempotent_without_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "unlink"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "data" not in payload
    assert payload["resource"]["linked"] is False


def test_project_unlink_yes_removes_existing_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    _seed_link(isolated_mapping, project)
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "unlink", "--yes"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "data" not in payload
    assert payload["resource"]["linked"] is False
    assert payload["resource"]["workspace"]["id"] == "ws_1"
    raw = json.loads(isolated_mapping.read_text(encoding="utf-8"))
    assert (
        str(project.resolve())
        not in raw["platforms"]["https://platform.osmosis.ai"]["projects"]
    )


def test_project_unlink_non_interactive_existing_mapping_requires_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    _seed_link(isolated_mapping, project)
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "unlink"])

    assert rc == 1
    assert "INTERACTIVE_REQUIRED" in capsys.readouterr().err


def test_project_link_rejects_malformed_workspace_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [{"id": "ws_1", "connected_repo": None}],
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "VALIDATION"
    assert "Invalid workspace record" in error["message"]


def test_project_link_requires_connected_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace(repo_url=None)],
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "has no Git Sync connected repository" in error["message"]
    assert "https://platform.osmosis.ai/team-alpha/integrations/git" in error["message"]
    assert not isolated_mapping.exists()


def test_project_link_invalid_connected_repo_url_mentions_git_sync_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace(repo_url="not-a-git-url")],
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Platform/Git Sync connected repository URL" in error["message"]
    assert "Update Git Sync settings" in error["message"]
    assert "https://platform.osmosis.ai/team-alpha/integrations/git" in error["message"]
    assert not isolated_mapping.exists()


def test_project_link_requires_origin_for_git_sync_repo_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace()],
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.get_local_git_remote_url",
        lambda project_root: None,
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Platform/Git Sync connected repository" in error["message"]
    assert "Git Sync repo" in error["message"]
    assert "has no `origin` remote" in error["message"]
    assert _CONNECTED_REPO in error["message"]
    assert not isolated_mapping.exists()


def test_project_link_requires_matching_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [_workspace()],
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.get_local_git_remote_url",
        lambda project_root: "https://github.com/acme/other.git",
    )

    rc = main(["--json", "project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert "must be run from a clone" in error["message"]
    assert "Platform/Git Sync connected repository" in error["message"]
    assert "Git Sync repo" in error["message"]
    assert _CONNECTED_REPO in error["message"]
    assert "https://github.com/acme/other.git" in error["message"]
    assert not isolated_mapping.exists()


def test_project_link_rich_output_escapes_workspace_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.list_accessible_workspaces",
        lambda *, credentials: [
            {
                "id": "ws_1",
                "name": "[red]team[/red]",
                "connected_repo": {"repo_url": _CONNECTED_REPO},
            }
        ],
    )
    _allow_matching_origin(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.console.print",
        lambda message, **_: messages.append(str(message)),
    )

    rc = main(["project", "link", "--workspace", "ws_1", "--yes"])

    assert rc == 0
    assert messages == ["Linked project to workspace: \\[red]team\\[/red]"]


def test_project_unlink_rich_output_escapes_workspace_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    _seed_link(isolated_mapping, project, workspace_name="[red]team[/red]")
    monkeypatch.chdir(project)
    messages: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.console.print",
        lambda message, **_: messages.append(str(message)),
    )

    rc = main(["project", "unlink", "--yes"])

    assert rc == 0
    assert messages == ["Unlinked project from workspace: \\[red]team\\[/red]"]

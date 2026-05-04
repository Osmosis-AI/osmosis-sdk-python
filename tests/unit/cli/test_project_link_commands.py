from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.cli.main import main
from osmosis_ai.platform.cli.project_mapping import (
    ProjectLinkRecord,
    ProjectMappingStore,
)


class _Creds:
    def is_expired(self) -> bool:
        return False


def _make_project(root: Path) -> Path:
    for rel in (
        ".osmosis",
        "rollouts",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".osmosis" / "program.md").write_text("# Test Program\n", encoding="utf-8")
    return root


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
) -> None:
    ProjectMappingStore(
        config_file=config_file,
        platform_url="https://platform.osmosis.ai",
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
        lambda *, credentials: [
            {"id": "ws_1", "name": "team-alpha", "connected_repo": None}
        ],
    )

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


def test_link_alias_matches_project_link(
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
        lambda *, credentials: [
            {"id": "ws_1", "name": "team-alpha", "connected_repo": None}
        ],
    )

    assert main(["--json", "link", "--workspace", "ws_1", "--yes"]) == 0
    payload = json.loads(capsys.readouterr().out)
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


def test_project_info_refresh_updates_cached_workspace_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    _seed_link(
        isolated_mapping,
        project,
        workspace_name="old-name",
        repo_url="https://github.com/acme/old",
    )
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.load_credentials", lambda: _Creds()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project.refresh_workspace_by_id",
        lambda *, workspace_id, credentials: {
            "id": workspace_id,
            "name": "new-name",
            "connected_repo": {
                "repo_url": "https://user:token@github.com/acme/new.git?secret=x#main"
            },
        },
    )

    rc = main(["--json", "project", "info", "--refresh"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["workspace"]["name"] == "new-name"
    assert payload["data"]["workspace"]["repo_url"] == "https://github.com/acme/new.git"
    raw = json.loads(isolated_mapping.read_text(encoding="utf-8"))
    stored = raw["platforms"]["https://platform.osmosis.ai"]["projects"][
        str(project.resolve())
    ]
    assert stored["workspaceName"] == "new-name"
    assert stored["repoUrl"] == "https://github.com/acme/new.git"


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
            {"id": "ws_1", "name": "[red]team[/red]", "connected_repo": None}
        ],
    )
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


def test_project_info_shows_unlinked_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_mapping: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "info"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["project_root"] == str(project.resolve())
    assert payload["data"]["linked"] is False

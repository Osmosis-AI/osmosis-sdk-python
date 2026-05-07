from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.project_mapping import (
    MappingConflictError,
    ProjectLinkRecord,
    ProjectMappingStore,
    normalize_platform_key,
    sanitize_repo_url,
)


def test_normalize_platform_key_keeps_scheme_host_and_explicit_port() -> None:
    assert (
        normalize_platform_key("https://platform.osmosis.ai/app?x=1")
        == "https://platform.osmosis.ai"
    )
    assert (
        normalize_platform_key("http://localhost:3000/foo/") == "http://localhost:3000"
    )


def test_normalize_platform_key_strips_default_ports_and_userinfo() -> None:
    assert (
        normalize_platform_key("https://platform.osmosis.ai:443/app")
        == "https://platform.osmosis.ai"
    )
    assert (
        normalize_platform_key("http://platform.osmosis.ai:80/app")
        == "http://platform.osmosis.ai"
    )
    assert (
        normalize_platform_key("https://user:token@platform.osmosis.ai/app")
        == "https://platform.osmosis.ai"
    )


def test_normalize_platform_key_brackets_ipv6_when_rebuilding_netloc() -> None:
    assert normalize_platform_key("http://[::1]:3000/path") == "http://[::1]:3000"


def test_sanitize_repo_url_removes_https_secret_query_and_fragment() -> None:
    assert (
        sanitize_repo_url("https://user:pat@github.com/acme/repo.git?token=x#main")
        == "https://github.com/acme/repo.git"
    )


def test_sanitize_repo_url_accepts_scp_ssh_form_without_userinfo_secret() -> None:
    assert (
        sanitize_repo_url("git@github.com:acme/repo.git")
        == "git@github.com:acme/repo.git"
    )


def test_sanitize_repo_url_rejects_non_git_scp_user() -> None:
    assert sanitize_repo_url("alice@github.com:acme/repo.git") is None


@pytest.mark.parametrize(
    "repo_url",
    [
        "git@exa mple.com:acme/repo.git",
        "git@.com:acme/repo.git",
    ],
)
def test_sanitize_repo_url_rejects_scp_invalid_hostname(repo_url: str) -> None:
    assert sanitize_repo_url(repo_url) is None


@pytest.mark.parametrize(
    "repo_url",
    [
        "ssh://git@github.com/acme/repo.git",
        "ftp://user:pw@example.com/path",
    ],
)
def test_sanitize_repo_url_rejects_non_https_url_schemes(repo_url: str) -> None:
    assert sanitize_repo_url(repo_url) is None


def test_normalize_platform_key_rejects_invalid_port() -> None:
    with pytest.raises(CLIError):
        normalize_platform_key("https://example.com:bad/path")


@pytest.mark.parametrize(
    "platform_url",
    [
        "https://exa mple.com/path",
        "https://.com",
    ],
)
def test_normalize_platform_key_rejects_invalid_hostname(platform_url: str) -> None:
    with pytest.raises(CLIError):
        normalize_platform_key(platform_url)


@pytest.mark.parametrize(
    "repo_url",
    [
        "https://github.com:bad/acme/repo.git",
        "https://[::1/acme/repo.git",
    ],
)
def test_sanitize_repo_url_returns_none_for_malformed_https_urls(repo_url: str) -> None:
    assert sanitize_repo_url(repo_url) is None


@pytest.mark.parametrize(
    "repo_url",
    [
        "https://exa mple.com/acme/repo.git",
        "https://.com/acme/repo.git",
    ],
)
def test_sanitize_repo_url_rejects_https_invalid_hostname(repo_url: str) -> None:
    assert sanitize_repo_url(repo_url) is None


def test_sanitize_repo_url_brackets_ipv6_when_rebuilding_netloc() -> None:
    assert (
        sanitize_repo_url("https://user:pat@[::1]:8443/acme/repo.git?token=x#main")
        == "https://[::1]:8443/acme/repo.git"
    )


def test_store_writes_owner_only_file_and_round_trips(tmp_path: Path) -> None:
    store = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://platform.osmosis.ai"
    )
    record = ProjectLinkRecord(
        project_path=str((tmp_path / "project").resolve()),
        workspace_id="ws_1",
        workspace_name="team-alpha",
        repo_url="https://github.com/acme/repo",
        linked_at="2026-05-03T00:00:00+00:00",
    )

    store.link(record)

    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600
    assert store.get_project(record.project_path) == record
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    bucket = raw["platforms"]["https://platform.osmosis.ai"]
    assert bucket["workspaceToProject"]["ws_1"] == record.project_path


@pytest.mark.parametrize(
    "field,value",
    [
        ("projectPath", ""),
        ("workspaceId", 123),
        ("workspaceName", None),
        ("linkedAt", ""),
    ],
)
def test_project_link_record_from_json_rejects_malformed_required_fields(
    field: str, value: object
) -> None:
    data = {
        "projectPath": "/tmp/project",
        "workspaceId": "ws_1",
        "workspaceName": "team-alpha",
        "linkedAt": "2026-05-03T00:00:00+00:00",
    }
    data[field] = value

    with pytest.raises(CLIError, match="Invalid project link record"):
        ProjectLinkRecord.from_json(data)


def test_link_returns_sanitized_stored_record(tmp_path: Path) -> None:
    store = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://platform.osmosis.ai"
    )
    record = ProjectLinkRecord(
        project_path=str((tmp_path / "project").resolve()),
        workspace_id="ws_1",
        workspace_name="team-alpha",
        repo_url="https://user:pat@github.com/acme/repo.git?token=x#main",
        linked_at="2026-05-03T00:00:00+00:00",
    )

    linked = store.link(record)

    assert linked.repo_url == "https://github.com/acme/repo.git"
    assert linked == store.get_project(record.project_path)


def test_relink_same_workspace_refreshes_stored_metadata(tmp_path: Path) -> None:
    store = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://platform.osmosis.ai"
    )
    project_path = str((tmp_path / "project").resolve())
    store.link(
        ProjectLinkRecord(
            project_path=project_path,
            workspace_id="ws_1",
            workspace_name="old-name",
            repo_url="https://github.com/acme/old.git",
            linked_at="2026-05-03T00:00:00+00:00",
        )
    )
    refreshed = ProjectLinkRecord(
        project_path=project_path,
        workspace_id="ws_1",
        workspace_name="new-name",
        repo_url="https://user:pat@github.com/acme/new.git?token=x#main",
        linked_at="2026-05-06T00:00:00+00:00",
    )

    linked = store.link(refreshed)

    assert linked == ProjectLinkRecord(
        project_path=project_path,
        workspace_id="ws_1",
        workspace_name="new-name",
        repo_url="https://github.com/acme/new.git",
        linked_at="2026-05-06T00:00:00+00:00",
    )
    assert store.get_project(project_path) == linked


def test_platform_buckets_are_isolated(tmp_path: Path) -> None:
    prod = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://platform.osmosis.ai"
    )
    staging = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://staging.osmosis.ai"
    )
    record = ProjectLinkRecord(
        project_path=str((tmp_path / "project").resolve()),
        workspace_id="ws_1",
        workspace_name="team-alpha",
        repo_url=None,
        linked_at="2026-05-03T00:00:00+00:00",
    )

    prod.link(record)

    assert prod.get_project(record.project_path) == record
    assert staging.get_project(record.project_path) is None


def test_same_workspace_on_different_path_raises_conflict(tmp_path: Path) -> None:
    store = ProjectMappingStore(
        config_file=tmp_path / "config.json", platform_url="https://platform.osmosis.ai"
    )
    first = ProjectLinkRecord(
        str((tmp_path / "one").resolve()),
        "ws_1",
        "team-alpha",
        None,
        "2026-05-03T00:00:00+00:00",
    )
    second = ProjectLinkRecord(
        str((tmp_path / "two").resolve()),
        "ws_1",
        "team-alpha",
        None,
        "2026-05-03T00:00:00+00:00",
    )
    store.link(first)

    with pytest.raises(MappingConflictError) as exc:
        store.link(second)

    assert str((tmp_path / "one").resolve()) in str(exc.value)


def test_forward_schema_version_raises_upgrade_message(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 2, "platforms": {}}), encoding="utf-8")
    store = ProjectMappingStore(
        config_file=path, platform_url="https://platform.osmosis.ai"
    )

    with pytest.raises(CLIError, match="upgrade"):
        store.get_project(str(tmp_path.resolve()))


def test_malformed_project_record_raises_cli_error(tmp_path: Path) -> None:
    project_path = str((tmp_path / "project").resolve())
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "platforms": {
                    "https://platform.osmosis.ai": {
                        "projects": {
                            project_path: {
                                "projectPath": project_path,
                                "workspaceName": "team-alpha",
                                "linkedAt": "2026-05-03T00:00:00+00:00",
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = ProjectMappingStore(
        config_file=path, platform_url="https://platform.osmosis.ai"
    )

    with pytest.raises(CLIError, match="Invalid project link record"):
        store.get_project(project_path)

"""Tests for `osmosis auth whoami` across rich/json/plain."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import VerifyResult


@pytest.fixture
def fake_credentials() -> Credentials:
    return Credentials(
        access_token="token",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=30),
        created_at=datetime.now(UTC),
        user=UserInfo(id="u1", email="brian@example.com", name="Brian"),
        token_id="tok_1",
    )


def _patch_auth(monkeypatch, creds: Credentials | None) -> list[str]:
    verify_calls: list[str] = []
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: creds)
    if creds is not None:
        verified = VerifyResult(
            user=creds.user,
            expires_at=creds.expires_at,
            token_id=creds.token_id,
        )
        monkeypatch.setattr(
            "osmosis_ai.platform.auth.verify_token",
            lambda token: verify_calls.append(token) or verified,
        )
    return verify_calls


def _write_legacy_active_workspace(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        '{"active_workspace":{"id":"legacy-ws","name":"legacy-workspace"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.CONFIG_FILE",
        config_file,
    )


def _create_canonical_project(project_root: Path) -> None:
    for relative in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (project_root / relative).mkdir(parents=True, exist_ok=True)
    (project_root / ".osmosis" / "project.toml").write_text(
        '[project]\nname = "demo"\n',
        encoding="utf-8",
    )
    (project_root / ".osmosis" / "research" / "program.md").write_text(
        "# Program\n",
        encoding="utf-8",
    )


def _link_project_mapping(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_file: Path,
    project_root: Path,
    workspace_id: str = "ws_1",
    workspace_name: str = "default",
) -> None:
    from osmosis_ai.platform.cli.project_mapping import (
        ProjectLinkRecord,
        ProjectMappingStore,
        now_linked_at,
    )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_mapping.CONFIG_FILE",
        config_file,
    )
    ProjectMappingStore(config_file=config_file).link(
        ProjectLinkRecord(
            project_path=str(project_root.resolve()),
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            repo_url=None,
            linked_at=now_linked_at(),
        )
    )


def _assert_no_project_context(data: dict) -> None:
    assert "workspace" not in data
    assert "linked_project" not in data
    assert "local_linked_project" not in data


@pytest.fixture
def fake_verify_result() -> VerifyResult:
    return VerifyResult(
        user=UserInfo(id="env_u1", email="env@example.com", name="Env User"),
        expires_at=datetime.now(UTC) + timedelta(days=30),
        token_id="env_tok_1",
    )


def test_whoami_json_outside_linked_project_has_no_workspace(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    data = payload["data"]
    assert data["email"] == "brian@example.com"
    assert data["name"] == "Brian"
    _assert_no_project_context(data)
    assert data["account"]["email"] == "brian@example.com"
    assert data["account"]["source"] == "credentials"
    assert data["source"] == "credentials"
    assert "expires_at" in data


def test_whoami_json_inside_linked_project_stays_auth_only(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    project_root = tmp_path / "project"
    _create_canonical_project(project_root)
    _link_project_mapping(
        monkeypatch,
        config_file=tmp_path / "home" / ".osmosis" / "config.json",
        project_root=project_root,
        workspace_id="ws_linked",
        workspace_name="team-alpha",
    )
    monkeypatch.chdir(project_root)
    _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    _assert_no_project_context(data)


def test_whoami_json_with_stored_credentials_verifies_token(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    monkeypatch.chdir(tmp_path)
    verify_calls = _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 0
    data = json.loads(captured.out)["data"]
    assert data["account"]["source"] == "credentials"
    assert data["account"]["email"] == "brian@example.com"
    assert verify_calls == ["token"]


def test_whoami_json_with_env_token_uses_verified_identity(
    monkeypatch, capsys, tmp_path, fake_verify_result
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    verify_calls = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: verify_calls.append(token) or fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("env token whoami should not use stored credentials"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "env token whoami should not fetch workspace list"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    assert data["name"] == "Env User"
    _assert_no_project_context(data)
    assert data["account"]["email"] == "env@example.com"
    assert data["account"]["source"] == "environment"
    assert data["source"] == "environment"
    assert verify_calls == ["env-token"]


def test_whoami_json_with_env_token_ignores_cached_workspace_resolution(
    monkeypatch, capsys, tmp_path, fake_verify_result
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    _write_legacy_active_workspace(monkeypatch, tmp_path / "legacy-config.json")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "env token whoami should not fetch workspace list"
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("env token whoami should not use stored credentials"),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    _assert_no_project_context(data)
    assert data["source"] == "environment"


def test_whoami_json_with_env_token_stays_auth_only_inside_linked_project(
    monkeypatch, capsys, tmp_path, fake_verify_result
) -> None:
    project_root = tmp_path / "project"
    _create_canonical_project(project_root)
    _link_project_mapping(
        monkeypatch,
        config_file=tmp_path / "home" / ".osmosis" / "config.json",
        project_root=project_root,
        workspace_id="ws_linked",
        workspace_name="team-alpha",
    )
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "env token whoami should not fetch workspace list"
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("env token whoami should not use stored credentials"),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)["data"]
    assert data["account"]["email"] == "env@example.com"
    _assert_no_project_context(data)


def test_whoami_json_with_env_token_ignores_mismatched_local_workspace(
    monkeypatch, capsys, tmp_path, fake_verify_result
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    _write_legacy_active_workspace(monkeypatch, tmp_path / "legacy-config.json")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "env token whoami should not fetch workspace list"
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("env token whoami should not use stored credentials"),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    _assert_no_project_context(data)
    assert data["source"] == "environment"


def test_whoami_json_with_env_token_verify_failure_emits_auth_required(
    monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth import LoginError

    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Authentication failed.", status_code=401)
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail(
            "env token whoami should not fall back to stored credentials"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"


def test_whoami_json_with_env_token_malformed_response_emits_platform_error(
    monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth import LoginError

    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Invalid response from platform")
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail(
            "env token whoami should not fall back to stored credentials"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "PLATFORM_ERROR"


def test_whoami_json_when_logged_out_emits_auth_required(monkeypatch, capsys) -> None:
    _patch_auth(monkeypatch, None)
    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"


def test_whoami_json_with_expired_stored_credentials_emits_auth_required(
    monkeypatch, capsys, fake_credentials
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    expired_credentials = Credentials(
        access_token=fake_credentials.access_token,
        token_type=fake_credentials.token_type,
        expires_at=datetime.now(UTC) - timedelta(days=1),
        created_at=fake_credentials.created_at,
        user=fake_credentials.user,
        token_id=fake_credentials.token_id,
    )
    _patch_auth(monkeypatch, expired_credentials)

    exit_code = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"
    assert "session has expired" in envelope["error"]["message"]
    assert "osmosis auth login" in envelope["error"]["message"]


def test_whoami_json_with_revoked_stored_credentials_emits_auth_required(
    monkeypatch, capsys, fake_credentials
) -> None:
    from osmosis_ai.platform.auth import LoginError

    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    _patch_auth(monkeypatch, fake_credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Token has been revoked.", code="TOKEN_REVOKED", status_code=401)
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"
    assert "osmosis auth login" in envelope["error"]["message"]


def test_whoami_plain_renders_label_value_lines(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["--plain", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.splitlines()
    assert "Email: brian@example.com" in lines
    assert not any(line.startswith("Workspace:") for line in lines)


def test_whoami_plain_inside_linked_project_stays_auth_only(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    project_root = tmp_path / "project"
    _create_canonical_project(project_root)
    _link_project_mapping(
        monkeypatch,
        config_file=tmp_path / "home" / ".osmosis" / "config.json",
        project_root=project_root,
        workspace_id="ws_linked",
        workspace_name="team-alpha",
    )
    monkeypatch.chdir(project_root)
    _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["--plain", "auth", "whoami"])

    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.splitlines()
    assert not any(line.startswith("Local project root:") for line in lines)
    assert not any(line.startswith("Local linked workspace:") for line in lines)


def test_whoami_rich_still_uses_table(
    monkeypatch, capsys, tmp_path, fake_credentials
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_auth(monkeypatch, fake_credentials)

    exit_code = cli.main(["auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Email" in captured.out
    assert "brian@example.com" in captured.out

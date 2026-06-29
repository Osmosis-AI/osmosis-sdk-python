from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.main import main


@pytest.fixture(autouse=True)
def _logged_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep doctor's best-effort workspace lookup off the network."""
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: None)


def _write_workspace_template(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "AGENTS.md").write_text("template agents\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("template claude\n", encoding="utf-8")
    (root / "configs" / "AGENTS.md").write_text(
        "template config agents\n", encoding="utf-8"
    )
    (root / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _write_workspace_template(tmp_path / "workspace-template")
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(root))
    return root


def _make_workspace_directory(root: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    return root


def test_project_doctor_dry_run_reports_missing_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "rollouts/" in payload["resource"]["missing"]
    assert "configs/training/" in payload["resource"]["missing"]
    assert ".osmosis/cache/" not in payload["resource"]["missing"]
    assert "rollouts/.gitkeep" not in payload["resource"]["missing"]
    assert "AGENTS.md" in payload["resource"]["missing"]
    assert payload["resource"]["fixed"] is False
    assert payload["resource"]["valid"] is False
    assert payload["resource"]["required_paths"] == [
        "rollouts/",
        "configs/training/",
        "configs/eval/",
        "data/",
    ]


def test_project_doctor_reports_git_context(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "remote",
            "add",
            "origin",
            "git@github.com:Acme/Rollouts.git",
        ],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis/cache",
        "rollouts",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (project / rel_path).mkdir(parents=True, exist_ok=True)
    (project / "configs" / "AGENTS.md").write_text("config agents\n", encoding="utf-8")
    (project / ".claude").mkdir()
    (project / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("agents\n", encoding="utf-8")
    (project / "CLAUDE.md").write_text("claude\n", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "success"
    assert payload["resource"]["valid"] is True
    assert payload["resource"]["workspace"] is None
    assert payload["resource"]["git"]["identity"] == "acme/rollouts"
    assert (
        payload["resource"]["git"]["remote_url"]
        == "ssh://git@github.com/Acme/Rollouts.git"
    )
    assert "warning" not in payload["resource"]["git"]


def test_project_doctor_reports_invalid_git_origin_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "remote",
            "add",
            "origin",
            "https://gitlab.com/acme/rollouts.git",
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["resource"]["git"]["identity"] is None
    assert payload["resource"]["git"]["remote_url"] is None
    assert "hosted on github.com" in payload["resource"]["git"]["warning"]


def test_project_doctor_does_not_report_missing_gitkeep_for_existing_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    (project / "rollouts").mkdir()
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "rollouts/" not in payload["resource"]["missing"]
    assert "rollouts/.gitkeep" not in payload["resource"]["missing"]


def test_project_doctor_plain_reports_actionable_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--plain", "doctor"])

    assert rc == 1
    output = capsys.readouterr().out
    assert "Workspace doctor found missing scaffold paths." in output
    assert f"Workspace directory: {project}" in output
    assert "Missing scaffold paths:" in output
    assert "rollouts/" in output
    assert ".gitkeep" not in output
    assert "Run `osmosis doctor --fix` to create missing scaffold paths." in output


def test_project_doctor_dry_run_does_not_require_workspace_template(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from osmosis_ai.templates import source

    def fail_download(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("dry-run doctor must not fetch workspace-template")

    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.delenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", raising=False)
    monkeypatch.setattr(source, "_download_workspace_template", fail_download)

    rc = main(["--json", "doctor"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "AGENTS.md" in payload["resource"]["missing"]


def test_project_doctor_rejects_yes_option(tmp_path: Path, monkeypatch, capsys) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor", "--yes"])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "VALIDATION"
    assert "--yes" in error["message"]


def test_project_doctor_fix_creates_missing_paths(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "doctor", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "success"
    assert (project / "configs" / "training").is_dir()
    assert (project / "AGENTS.md").is_file()
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "template agents\n"
    assert payload["resource"]["missing"] == []
    assert not (project / ".osmosis" / "project.toml").exists()


def test_project_doctor_fix_outside_project_does_not_create_project(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "doctor", "--fix"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert not (tmp_path / ".osmosis" / "project.toml").exists()
    error = json.loads(captured.err)["error"]
    assert error["code"] == "WORKSPACE_REQUIRED"
    assert "Osmosis workspace directory" in error["message"]
    assert "osmosis init" not in error["message"]


def _fake_credentials():
    from datetime import UTC, datetime, timedelta

    from osmosis_ai.platform.auth.credentials import Credentials, UserInfo

    return Credentials(
        access_token="token",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=30),
        created_at=datetime.now(UTC),
        user=UserInfo(id="u1", email="brian@example.com", name="Brian"),
        token_id="tok_1",
    )


def _add_origin_remote(project: Path) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "remote",
            "add",
            "origin",
            "git@github.com:Acme/Rollouts.git",
        ],
        check=True,
        capture_output=True,
    )


def test_project_doctor_reports_linked_workspace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth.flow import VerifiedWorkspace, VerifyResult

    project = _make_workspace_directory(tmp_path / "project")
    _add_origin_remote(project)
    monkeypatch.chdir(project)
    credentials = _fake_credentials()
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda: credentials
    )
    verify_calls: list[dict[str, str | None]] = []
    verified = VerifyResult(
        user=credentials.user,
        expires_at=credentials.expires_at,
        token_id=credentials.token_id,
        workspace=VerifiedWorkspace(id="ws_1", name="Acme Workspace", role="admin"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (
            verify_calls.append({"token": token, "git_identity": git_identity})
            or verified
        ),
    )

    rc = main(["--json", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1  # scaffold is still missing — workspace lookup is orthogonal
    assert verify_calls == [{"token": "token", "git_identity": "acme/rollouts"}]
    assert payload["resource"]["workspace"] == {
        "id": "ws_1",
        "name": "Acme Workspace",
        "role": "admin",
    }


def test_project_doctor_plain_shows_linked_workspace_line(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth.flow import VerifiedWorkspace, VerifyResult

    project = _make_workspace_directory(tmp_path / "project")
    _add_origin_remote(project)
    monkeypatch.chdir(project)
    credentials = _fake_credentials()
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda: credentials
    )
    verified = VerifyResult(
        user=credentials.user,
        expires_at=credentials.expires_at,
        token_id=credentials.token_id,
        workspace=VerifiedWorkspace(id="ws_1", name="Acme Workspace", role="admin"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: verified,
    )

    rc = main(["--plain", "doctor"])

    assert rc == 1
    assert "Linked workspace: Acme Workspace" in capsys.readouterr().out


def test_project_doctor_degrades_when_workspace_lookup_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth import LoginError

    project = _make_workspace_directory(tmp_path / "project")
    _add_origin_remote(project)
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda: _fake_credentials()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (_ for _ in ()).throw(
            LoginError("Could not connect to platform: offline")
        ),
    )

    rc = main(["--json", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["resource"]["workspace"] is None


def test_project_doctor_skips_workspace_lookup_without_git_identity(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda: _fake_credentials()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: pytest.fail(
            "doctor must not call verify without a git identity"
        ),
    )

    rc = main(["--json", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["resource"]["workspace"] is None

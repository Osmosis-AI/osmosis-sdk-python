from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.main import main


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


def _make_project(root: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    return root


def test_project_doctor_dry_run_reports_missing_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "rollouts/" in payload["resource"]["missing"]
    assert "configs/training/" in payload["resource"]["missing"]
    assert "rollouts/.gitkeep" not in payload["resource"]["missing"]
    assert "AGENTS.md" in payload["resource"]["missing"]
    assert payload["resource"]["fixed"] is False
    assert payload["resource"]["updates_checked"] is False
    assert payload["resource"]["valid"] is False
    assert payload["resource"]["required_paths"] == [
        "rollouts/",
        "configs/training/",
        "configs/eval/",
        "data/",
    ]
    assert not (project / "research" / "program.md").exists()


def test_project_doctor_reports_git_context(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
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

    rc = main(["--json", "project", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["resource"]["valid"] is True
    assert payload["resource"]["git"]["identity"] == "acme/rollouts"
    assert (
        payload["resource"]["git"]["remote_url"]
        == "ssh://git@github.com/Acme/Rollouts.git"
    )
    assert "warning" not in payload["resource"]["git"]


def test_project_doctor_reports_invalid_git_origin_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
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

    rc = main(["--json", "project", "doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["resource"]["git"]["identity"] is None
    assert payload["resource"]["git"]["remote_url"] is None
    assert "hosted on github.com" in payload["resource"]["git"]["warning"]


def test_project_doctor_does_not_report_missing_gitkeep_for_existing_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    (project / "rollouts").mkdir()
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "rollouts/" not in payload["resource"]["missing"]
    assert "rollouts/.gitkeep" not in payload["resource"]["missing"]


def test_project_doctor_plain_reports_actionable_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--plain", "project", "doctor"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Project doctor completed." in output
    assert f"Project root: {project}" in output
    assert "Missing scaffold paths:" in output
    assert "rollouts/" in output
    assert ".gitkeep" not in output
    assert (
        "Run `osmosis project doctor --fix` to create missing scaffold paths." in output
    )


def test_project_doctor_dry_run_does_not_require_workspace_template(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from osmosis_ai.templates import source

    def fail_download(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("dry-run doctor must not fetch workspace-template")

    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)
    monkeypatch.delenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", raising=False)
    monkeypatch.setattr(source, "_download_workspace_template", fail_download)

    rc = main(["--json", "project", "doctor"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "AGENTS.md" in payload["resource"]["missing"]


def test_project_doctor_rejects_yes_option(tmp_path: Path, monkeypatch, capsys) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor", "--yes"])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "VALIDATION"
    assert "--yes" in error["message"]


def test_project_doctor_fix_creates_missing_paths(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert not (project / ".osmosis" / "research" / "program.md").exists()
    assert (project / "configs" / "training").is_dir()
    assert (project / "AGENTS.md").is_file()
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "template agents\n"
    assert payload["resource"]["missing"] == []
    assert payload["resource"]["updates_checked"] is True
    assert not (project / ".osmosis" / "project.toml").exists()


def test_project_doctor_fix_outside_project_does_not_create_project(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "project", "doctor", "--fix"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert not (tmp_path / ".osmosis" / "project.toml").exists()
    message = json.loads(captured.err)["error"]["message"]
    assert "cloned Osmosis repository" in message
    assert "osmosis init" not in message


def test_project_doctor_fix_preserves_existing_research_program(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    program = project / "research" / "program.md"
    program.parent.mkdir(parents=True)
    program.write_text("# Research Brief\n\nKeep this content.\n", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor", "--fix"])

    capsys.readouterr()
    assert rc == 0
    assert (
        program.read_text(encoding="utf-8")
        == "# Research Brief\n\nKeep this content.\n"
    )


def test_project_doctor_reports_agent_updates_without_overwriting(
    tmp_path: Path, monkeypatch, workspace_template: Path
) -> None:
    from osmosis_ai.cli.output import OutputFormat, override_output_context
    from osmosis_ai.platform.cli.project import doctor_project

    project = _make_project(tmp_path / "project")
    (project / "AGENTS.md").write_text("custom agents", encoding="utf-8")
    monkeypatch.chdir(project)

    with override_output_context(format=OutputFormat.rich, interactive=True):
        result = doctor_project(fix=True)

    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "custom agents"
    assert (project / "configs" / "training").is_dir()
    assert result.resource["updates_available"] == ["AGENTS.md"]
    assert result.display_next_steps


def test_project_refresh_agents_refuses_local_edits_without_force(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    (project / "AGENTS.md").write_text("custom agents", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "refresh-agents"])

    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["error"]["code"] == "CONFLICT"
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "custom agents"


def test_project_refresh_agents_force_overwrites_local_edits(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    (project / "AGENTS.md").write_text("custom agents", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "refresh-agents", "--force"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["resource"]["refreshed"] == ["AGENTS.md"]
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "template agents\n"


def test_project_refresh_agents_plain_reports_changed_files(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    (project / "AGENTS.md").write_text("custom agents", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--plain", "project", "refresh-agents", "--force"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Project agent scaffold refresh completed." in captured.out
    assert "Added: CLAUDE.md, configs/AGENTS.md, .claude/settings.json" in captured.out
    assert "Refreshed: AGENTS.md" in captured.out

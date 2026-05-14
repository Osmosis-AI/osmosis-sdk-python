from __future__ import annotations

import json
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
    (root / ".osmosis").mkdir(parents=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    return root


def test_project_doctor_dry_run_reports_missing_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "rollouts/.gitkeep" in payload["resource"]["missing"]
    assert "AGENTS.md" in payload["resource"]["missing"]
    assert payload["resource"]["fixed"] is False
    assert not (project / ".osmosis" / "research" / "program.md").exists()


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
    assert "Not in an Osmosis project" in message
    assert "existing Osmosis project" in message
    assert "osmosis init" not in message


def test_project_doctor_fix_preserves_existing_research_program(
    tmp_path: Path, monkeypatch, capsys, workspace_template: Path
) -> None:
    project = _make_project(tmp_path / "project")
    program = project / ".osmosis" / "research" / "program.md"
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

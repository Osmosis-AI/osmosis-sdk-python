from __future__ import annotations

import json
import subprocess
from pathlib import Path

from osmosis_ai.cli.main import main


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
    assert payload["resource"]["fixed"] is False
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
    for rel_path in ("rollouts", "configs/training", "configs/eval", "data"):
        (project / rel_path).mkdir(parents=True, exist_ok=True)
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


def test_project_doctor_fix_creates_missing_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor", "--fix", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert (project / "research" / "program.md").is_file()
    assert (project / "configs" / "training").is_dir()
    assert payload["resource"]["missing"] == []
    assert not (project / ".osmosis" / "project.toml").exists()


def test_project_doctor_fix_outside_project_does_not_create_project(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "project", "doctor", "--fix", "--yes"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert not (tmp_path / ".osmosis" / "project.toml").exists()
    message = json.loads(captured.err)["error"]["message"]
    assert "cloned Osmosis repository" in message
    assert "osmosis init" not in message


def test_project_doctor_fix_preserves_existing_research_program(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _make_project(tmp_path / "project")
    program = project / "research" / "program.md"
    program.parent.mkdir(parents=True)
    program.write_text("# Research Brief\n\nKeep this content.\n", encoding="utf-8")
    monkeypatch.chdir(project)

    rc = main(["--json", "project", "doctor", "--fix", "--yes"])

    capsys.readouterr()
    assert rc == 0
    assert (
        program.read_text(encoding="utf-8")
        == "# Research Brief\n\nKeep this content.\n"
    )


def test_project_doctor_declining_refresh_preserves_agents_but_repairs_missing_paths(
    tmp_path: Path, monkeypatch
) -> None:
    from osmosis_ai.cli.output import OutputFormat, override_output_context
    from osmosis_ai.platform.cli.project import doctor_project

    project = _make_project(tmp_path / "project")
    (project / "AGENTS.md").write_text("custom agents", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setattr("osmosis_ai.cli.prompts.confirm", lambda *a, **kw: False)

    with override_output_context(format=OutputFormat.rich, interactive=True):
        result = doctor_project(fix=True, yes=False)

    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "custom agents"
    assert (project / "research" / "program.md").is_file()
    assert (project / "configs" / "training").is_dir()
    assert result.resource["refreshed"] == []

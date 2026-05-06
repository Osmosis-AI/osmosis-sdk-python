from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli.main import main


def _make_project(root: Path) -> Path:
    for rel_path in (
        ".osmosis",
        "rollouts",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n",
        encoding="utf-8",
    )
    return root


def test_project_validate_success(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)

    rc = main(["project", "validate", str(project_root)])

    capsys.readouterr()
    assert rc == 0


def test_project_validate_reports_missing_required_path(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)
    (project_root / "configs" / "eval").rmdir()

    rc = main(["project", "validate", str(project_root)])

    captured = capsys.readouterr()
    assert rc != 0
    assert "configs/eval" in captured.err


def test_project_validate_does_not_require_training_brief(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)

    rc = main(["project", "validate", str(project_root)])

    capsys.readouterr()
    assert rc == 0

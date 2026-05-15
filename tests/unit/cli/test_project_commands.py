from __future__ import annotations

import subprocess
from pathlib import Path

from osmosis_ai.cli.main import main


def _make_project(root: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        "rollouts",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    return root


def test_project_doctor_accepts_project_path(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)

    rc = main(["project", "doctor", str(project_root)])

    capsys.readouterr()
    assert rc == 0


def test_project_validate_is_not_registered(tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path)

    rc = main(["project", "validate", str(project_root)])

    captured = capsys.readouterr()
    assert rc != 0
    assert "No such command" in captured.err


def test_project_link_and_unlink_are_not_registered(capfd) -> None:
    rc = main(["project", "--help"])
    output = capfd.readouterr().out

    assert rc == 0
    assert "validate" not in output
    assert "doctor" in output
    assert "link" not in output
    assert "unlink" not in output

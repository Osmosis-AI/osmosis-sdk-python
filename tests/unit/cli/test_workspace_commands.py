from __future__ import annotations

import subprocess
from pathlib import Path

from osmosis_ai.cli.main import main


def _make_workspace_directory(root: Path) -> Path:
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


def test_doctor_accepts_workspace_directory_path(tmp_path, capsys) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)

    rc = main(["doctor", str(workspace_directory)])

    capsys.readouterr()
    assert rc == 0


def test_project_validate_is_not_registered(tmp_path, capsys) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)

    rc = main(["project", "validate", str(workspace_directory)])

    captured = capsys.readouterr()
    assert rc != 0
    assert "No such command" in captured.err


def test_workspace_group_is_not_registered(capfd) -> None:
    rc = main(["workspace", "--help"])
    captured = capfd.readouterr()

    assert rc != 0
    assert "No such command" in captured.err

"""Project command JSON/plain contracts (local project)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from osmosis_ai.cli import main as cli


def _project_root(root: Path, *, origin: str | None = None) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    if origin is not None:
        subprocess.run(
            ["git", "-C", str(root), "remote", "add", "origin", origin],
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


def test_project_doctor_json_returns_diagnostic_resource(tmp_path, capsys) -> None:
    project_root = _project_root(
        tmp_path,
        origin="https://github.com/Acme/Rollouts.git",
    )

    exit_code = cli.main(["--json", "project", "doctor", str(project_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["resource"]["valid"] is True
    assert payload["resource"]["project_root"] == str(project_root)
    assert payload["resource"]["git"]["identity"] == "acme/rollouts"
    assert (
        payload["resource"]["git"]["remote_url"]
        == "https://github.com/Acme/Rollouts.git"
    )
    assert payload["resource"]["missing"] == []
    assert "workspace" not in payload["resource"]


def test_project_doctor_json_reports_missing_paths_without_error(
    tmp_path, capsys
) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    exit_code = cli.main(["--json", "project", "doctor", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["resource"]["valid"] is False
    assert payload["resource"]["missing"] == [
        "rollouts/",
        "configs/training/",
        "configs/eval/",
        "data/",
    ]

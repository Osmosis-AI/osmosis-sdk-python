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


def test_project_validate_json_returns_detail_result(tmp_path, capsys) -> None:
    project_root = _project_root(
        tmp_path,
        origin="https://github.com/Acme/Rollouts.git",
    )

    exit_code = cli.main(["--json", "project", "validate", str(project_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is True
    assert payload["data"]["project_root"] == str(project_root)
    assert payload["data"]["git"]["identity"] == "acme/rollouts"
    assert (
        payload["data"]["git"]["remote_url"] == "https://github.com/Acme/Rollouts.git"
    )
    assert "workspace" not in payload["data"]


def test_project_validate_json_reports_missing_paths_without_error(
    tmp_path, capsys
) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    exit_code = cli.main(["--json", "project", "validate", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is False
    assert payload["data"]["missing_paths"] == [
        "rollouts/",
        "configs/training/",
        "configs/eval/",
        "data/",
    ]

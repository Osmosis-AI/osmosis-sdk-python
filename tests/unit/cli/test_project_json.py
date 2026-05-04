"""Project command JSON/plain contracts (local project)."""

from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.cli import main as cli


def _project_root(root: Path) -> Path:
    for rel_path in (
        ".osmosis",
        "rollouts",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".osmosis" / "program.md").write_text("# Test Program\n", encoding="utf-8")
    return root


def test_project_validate_json_returns_detail_result(tmp_path, capsys) -> None:
    project_root = _project_root(tmp_path)

    exit_code = cli.main(["--json", "project", "validate", str(project_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is True
    assert payload["data"]["root"] == str(project_root)

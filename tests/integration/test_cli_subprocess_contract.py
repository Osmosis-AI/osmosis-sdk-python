"""Subprocess guardrails for JSON-mode CLI output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write_workspace_template(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "AGENTS.md").write_text("template agents\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("template claude\n", encoding="utf-8")
    (root / "configs" / "AGENTS.md").write_text(
        "template config agents\n", encoding="utf-8"
    )
    return root


def test_project_doctor_fix_in_git_repo_creates_runtime_free_scaffold(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subprocess.run(
        ["git", "init", "-b", "main", str(project)],
        check=True,
        capture_output=True,
    )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *(path for path in [env.get("PYTHONPATH")] if path)]
    )
    env["OSMOSIS_WORKSPACE_TEMPLATE_PATH"] = str(
        _write_workspace_template(tmp_path / "workspace-template")
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "osmosis_ai.cli.main",
            "--json",
            "doctor",
            "--fix",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert not (project / ".osmosis" / "project.toml").exists()
    assert not (project / ".osmosis" / "cache").exists()
    assert (project / "rollouts").is_dir()

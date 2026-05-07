"""Subprocess guardrails for JSON-mode CLI output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _make_project(root: Path) -> Path:
    for rel_path in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test\n", encoding="utf-8"
    )
    return root


def _run_json_command(
    args: list[str], tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    project = _make_project(tmp_path / "project")
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *(path for path in [env.get("PYTHONPATH")] if path)]
    )
    return subprocess.run(
        [sys.executable, "-m", "osmosis_ai.cli.main", "--json", *args],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_eval_cache_dir_json_is_single_stdout_envelope(tmp_path: Path) -> None:
    proc = _run_json_command(["eval", "cache", "dir"], tmp_path)

    assert proc.returncode == 0
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert (
        Path(payload["data"]["cache_root"])
        == (tmp_path / "project" / ".osmosis" / "cache" / "eval").resolve()
    )


def test_eval_cache_ls_json_is_single_stdout_envelope(tmp_path: Path) -> None:
    proc = _run_json_command(["eval", "cache", "ls"], tmp_path)

    assert proc.returncode == 0
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert payload["items"] == []


def test_eval_run_json_error_is_structured_stderr(tmp_path: Path) -> None:
    proc = _run_json_command(["eval", "run", "configs/eval/missing.toml"], tmp_path)

    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "VALIDATION"
    assert "Config file not found" in payload["error"]["message"]

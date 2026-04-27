"""Subprocess guardrails for JSON-mode CLI output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_json_command(
    args: list[str], tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "OSMOSIS_CACHE_DIR": str(tmp_path / "cache"),
    }
    return subprocess.run(
        [sys.executable, "-m", "osmosis_ai.cli.main", "--json", *args],
        cwd=ROOT,
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
    assert payload["data"]["cache_root"].endswith("/eval")


def test_eval_cache_ls_json_is_single_stdout_envelope(tmp_path: Path) -> None:
    proc = _run_json_command(["eval", "cache", "ls"], tmp_path)

    assert proc.returncode == 0
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert payload["items"] == []


def test_eval_run_json_error_is_structured_stderr(tmp_path: Path) -> None:
    proc = _run_json_command(["eval", "run", str(tmp_path / "missing.toml")], tmp_path)

    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "VALIDATION"
    assert "Config file not found" in payload["error"]["message"]
